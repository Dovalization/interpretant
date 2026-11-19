"""Where the transformer's time goes on real target sentences, step by step (`docs/performance.md`).

The usages stage annotates every target sentence with the transformer pipeline. This script runs
one part's sentences the way the stage does (sentences sorted by length, batches capped by documents
and tokens, the stage's model settings, one process on the GPU) and times each pipeline step over
every batch, synchronising the GPU around each step so its work is counted where it happens, plus
building the documents and applying the lemma exceptions. Run from the repository root on the GPU
machine, with no other GPU work: `python scripts/evidence/transformer_time.py PART OUT.json`.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import spacy
import torch
from spacy.tokens import Doc
from thinc.api import require_gpu, set_gpu_allocator

from interpretant.catalog.config import read_catalog
from interpretant.catalog.lemma_exceptions import load_lemma_exceptions
from interpretant.catalog.spellings import load_spellings
from interpretant.catalog.words import load_words
from interpretant.corpus.lemma_exceptions import lemma_exception_ruler
from interpretant.pipelines.build_usages import (
    UsagesSettings,
    _batches,
    _match_part,
    _PartJob,
    _StagedUsages,
)
from interpretant.usages.entities import load_transformer


def main() -> None:
    """Time every step over one part's batches and write the totals."""
    part, out = sys.argv[1], Path(sys.argv[2])
    catalog = read_catalog()
    job = _PartJob(
        catalog,
        UsagesSettings(),
        tuple(load_words(catalog)),
        {spelling.form: spelling.standard for spelling in load_spellings(catalog)},
        tuple(load_lemma_exceptions(catalog)),
    )
    sink = _StagedUsages()
    _match_part(part, job, spacy.blank("en").vocab, sink)
    sentences = sorted(sink.sentences.values(), key=lambda sentence: len(sentence[0]))
    set_gpu_allocator("pytorch")
    require_gpu(0)
    transformer = load_transformer()
    exceptions = lemma_exception_ruler(transformer.vocab, job.exceptions)
    seconds: dict[str, float] = defaultdict(float)
    started = time.perf_counter()
    for positions in _batches([len(words) for words, _ in sentences]):
        clock = time.perf_counter()
        docs = [
            Doc(transformer.vocab, words=sentences[i][0], spaces=sentences[i][1]) for i in positions
        ]
        seconds["building documents"] += time.perf_counter() - clock
        for name, step in transformer.pipeline:
            torch.cuda.synchronize()
            clock = time.perf_counter()
            if hasattr(step, "pipe"):
                docs = list(step.pipe(docs, batch_size=len(docs)))
            else:
                docs = [step(doc) for doc in docs]
            torch.cuda.synchronize()
            seconds[name] += time.perf_counter() - clock
        clock = time.perf_counter()
        docs = [exceptions(doc) for doc in docs]
        seconds["lemma exceptions"] += time.perf_counter() - clock
    total = time.perf_counter() - started
    ranked = sorted(seconds.items(), key=lambda item: -item[1])
    report = {
        "part": part,
        "sentences": len(sentences),
        "tokens": sum(len(words) for words, _ in sentences),
        "seconds": round(total, 2),
        "sentences_per_second": round(len(sentences) / total),
        "steps": {name: round(value, 2) for name, value in ranked},
        "shares": {name: round(value / total, 3) for name, value in ranked},
        "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3),
    }
    out.write_text(json.dumps(report, indent=1) + "\n")
    sys.stdout.write(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
