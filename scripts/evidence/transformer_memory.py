"""GPU memory of the transformer on real target sentences, batch by batch (`docs/performance.md`).

The full transformer run of the usages stage ran out of GPU memory with four processes sharing the
card: memory filled within two minutes and stayed at the card's limit. This script runs one part's
target sentences the way the stage does (one process, sentences sorted by length, batches of 64
documents, the stage's model settings) and records after every batch: the documents, their tokens,
the longest one, and PyTorch's allocated memory (live tensors), reserved memory (held by its cache)
and the batch's peak. Run from the repository root on the GPU machine, with no other GPU work:
`python scripts/evidence/transformer_memory.py PART [PART...] OUT.jsonl`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import spacy
import torch
from spacy.tokens import Doc
from thinc.api import require_gpu, set_gpu_allocator

from interpretant.catalog.config import read_catalog
from interpretant.catalog.lemma_exceptions import load_lemma_exceptions
from interpretant.catalog.spellings import load_spellings
from interpretant.catalog.words import load_words
from interpretant.pipelines.build_usages import (
    _TRANSFORMER_BATCH_SIZE,
    UsagesSettings,
    _match_part,
    _PartJob,
    _StagedUsages,
)
from interpretant.usages.entities import load_transformer

GIGABYTE = 2**30


def target_sentences(part: str, job: _PartJob) -> list[tuple[list[str], list[bool]]]:
    """One part's target sentences, as the stage stages them."""
    sink = _StagedUsages()
    _match_part(part, job, spacy.blank("en").vocab, sink)
    return list(sink.sentences.values())


def main() -> None:
    """Annotate each part's sentences in the stage's order and log memory after every batch."""
    *parts, out = sys.argv[1:]
    catalog = read_catalog()
    job = _PartJob(
        catalog,
        UsagesSettings(),
        tuple(load_words(catalog)),
        {spelling.form: spelling.standard for spelling in load_spellings(catalog)},
        tuple(load_lemma_exceptions(catalog)),
    )
    set_gpu_allocator("pytorch")
    require_gpu(0)
    transformer = load_transformer()
    torch.cuda.synchronize()
    loaded = torch.cuda.memory_allocated() / GIGABYTE
    sys.stdout.write(f"model loaded: {loaded:.2f} GB allocated\n")
    with Path(out).open("w") as handle:
        for part in parts:
            sentences = sorted(target_sentences(part, job), key=lambda sentence: len(sentence[0]))
            sys.stdout.write(f"{part}: {len(sentences)} sentences\n")
            for start in range(0, len(sentences), _TRANSFORMER_BATCH_SIZE):
                batch = sentences[start : start + _TRANSFORMER_BATCH_SIZE]
                docs = [Doc(transformer.vocab, words=w, spaces=s) for w, s in batch]
                torch.cuda.reset_peak_memory_stats()
                began = time.perf_counter()
                list(transformer.pipe(docs, batch_size=len(docs)))
                torch.cuda.synchronize()
                record = {
                    "part": part,
                    "batch": start // _TRANSFORMER_BATCH_SIZE,
                    "documents": len(docs),
                    "tokens": sum(len(words) for words, _ in batch),
                    "longest": max(len(words) for words, _ in batch),
                    "allocated_gb": round(torch.cuda.memory_allocated() / GIGABYTE, 3),
                    "reserved_gb": round(torch.cuda.memory_reserved() / GIGABYTE, 3),
                    "peak_gb": round(torch.cuda.max_memory_allocated() / GIGABYTE, 3),
                    "seconds": round(time.perf_counter() - began, 3),
                }
                handle.write(json.dumps(record) + "\n")
            sys.stdout.write(
                f"{part}: done, reserved {torch.cuda.memory_reserved() / GIGABYTE:.2f} GB\n"
            )


if __name__ == "__main__":
    main()
