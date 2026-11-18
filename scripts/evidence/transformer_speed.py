"""How fast, and how faithfully, the transformer annotates target sentences on the GPU.

Step 6 may run `en_core_web_trf` on every target sentence (agenda 4.9). Its curated transformer has
a `mixed_precision` switch, off in the shipped config, that thinc's PyTorch shim turns into
`torch.autocast("cuda")` at prediction, so float16 matrix products run on the GPU's tensor cores
(the vector store's encoding ran 3.3 times faster that way, `docs/performance.md`). This script
takes real target sentences from cached documents and times the pipeline (without its parser) at
full precision, with TF32 matrix products, and with float16 autocast, at several batch sizes, with
the sentences in their order or sorted by length; then it compares float16 against full precision
token by token (tag, lemma, entity), so a faster setting is chosen only if it changes no annotation
that matters. Run from the repository root on the GPU machine:
`python scripts/evidence/transformer_speed.py OUT.json [SENTENCES]`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import spacy
import torch
from spacy.language import Language
from spacy.tokens import Doc
from thinc.api import require_gpu, set_gpu_allocator

from interpretant.catalog.annotations import load_annotations
from interpretant.catalog.config import read_catalog
from interpretant.catalog.spellings import load_spellings
from interpretant.catalog.words import load_words
from interpretant.usages.target_matching import build_target_matcher, match_targets

PARTS = ("ai_2010", "psychology_1970", "philosophy_1950", "hci_2020", "sociology_1990")
BATCH_SIZES = (64, 256, 1024)
SPANS_PER_FORWARD = 1024


def target_sentences(limit: int) -> list[tuple[list[str], list[bool]]]:
    """The words and spaces of up to ``limit`` cached sentences holding a target, over the parts."""
    catalog = read_catalog()
    matcher = build_target_matcher(
        load_words(catalog), {s.form: s.standard for s in load_spellings(catalog)}
    )
    vocab = spacy.blank("en").vocab
    found: list[tuple[list[str], list[bool]]] = []
    per_part = limit // len(PARTS)
    for part in PARTS:
        taken = 0
        for doc in load_annotations(catalog, part, vocab):
            heads = {mention.token_index for mention in match_targets(doc, matcher)}
            for sentence in doc.sents:
                if any(sentence.start <= head < sentence.end for head in heads):
                    found.append(
                        ([t.text for t in sentence], [bool(t.whitespace_) for t in sentence])
                    )
                    taken += 1
            if taken >= per_part:
                break
    return found[:limit]


def load(*, mixed_precision: bool) -> Language:
    """The transformer without its parser, its span batches widened, float16 or not."""
    return spacy.load(
        "en_core_web_trf",
        exclude=["parser"],
        config={
            "components": {
                "transformer": {
                    "model": {
                        "mixed_precision": mixed_precision,
                        "with_spans": {"batch_size": SPANS_PER_FORWARD},
                    }
                }
            }
        },
    )


def run(
    nlp: Language, sentences: list[tuple[list[str], list[bool]]], batch_size: int
) -> tuple[float, list[Doc]]:
    """The seconds the pipeline takes over the sentences, and its documents in input order."""
    docs = [Doc(nlp.vocab, words=words, spaces=spaces) for words, spaces in sentences]
    torch.cuda.synchronize()
    started = time.perf_counter()
    done = list(nlp.pipe(docs, batch_size=batch_size))
    torch.cuda.synchronize()
    return time.perf_counter() - started, done


def agreement(first: list[Doc], second: list[Doc]) -> dict[str, float]:
    """The share of tokens with the same tag, lemma and entity in two runs."""
    same = {"tag": 0, "lemma": 0, "entity": 0}
    tokens = 0
    for one, two in zip(first, second, strict=True):
        for a, b in zip(one, two, strict=True):
            tokens += 1
            same["tag"] += a.tag_ == b.tag_
            same["lemma"] += a.lemma_ == b.lemma_
            same["entity"] += (a.ent_iob_, a.ent_type_) == (b.ent_iob_, b.ent_type_)
    return {name: count / tokens for name, count in same.items()} | {"tokens": tokens}


def main() -> None:
    """Time every setting, compare float16 with full precision, and write the results."""
    out = Path(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    set_gpu_allocator("pytorch")
    require_gpu(0)
    sentences = target_sentences(limit)
    by_length = sorted(sentences, key=lambda sentence: len(sentence[0]))
    words = sum(len(words) for words, _ in sentences)
    sys.stdout.write(f"{len(sentences)} target sentences, {words} words\n")
    results: list[dict[str, Any]] = []
    outputs: dict[str, list[Doc]] = {}
    for precision in ("float32", "tf32", "float16"):
        torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
        nlp = load(mixed_precision=precision == "float16")
        run(nlp, sentences[:256], 256)
        for batch_size in BATCH_SIZES:
            for order, data in (("as found", sentences), ("by length", by_length)):
                seconds, docs = run(nlp, data, batch_size)
                if order == "as found" and batch_size == 256:
                    outputs[precision] = docs
                row = {
                    "precision": precision,
                    "batch_size": batch_size,
                    "order": order,
                    "seconds": round(seconds, 2),
                    "sentences_per_second": round(len(data) / seconds),
                    "words_per_second": round(words / seconds),
                    "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                }
                results.append(row)
                sys.stdout.write(json.dumps(row) + "\n")
        del nlp
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    compared = {
        "float16 against float32": agreement(outputs["float32"], outputs["float16"]),
        "tf32 against float32": agreement(outputs["float32"], outputs["tf32"]),
    }
    sys.stdout.write(json.dumps(compared, indent=1) + "\n")
    out.write_text(json.dumps({"runs": results, "agreement": compared}, indent=1) + "\n")


if __name__ == "__main__":
    main()
