"""Measure spaCy's transformer model on the GPU against the small model on a seeded sample.

Samples up to PER_CELL cleaned abstracts from every raw cell, counts lemma votes with the small
model on one CPU process and with the transformer on the GPU, times both, and compares the merge
decision each set of votes gives for every word type with enough readings in both. Prints and
writes a summary of counts and word types only, never abstract text.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import polars as pl

from interpretant.corpus.lemmas import MIN_VOTES, lemma_votes, voted_lemma
from interpretant.drift.senses import cased_abstracts

PER_CELL = 52
SEED = 2026
RAW_DIRS = (Path("data/raw/openalex"), Path("data/raw/openalex_full"))


def as_readings(votes):
    readings = {}
    for (form, tag, lemma), count in votes.items():
        readings.setdefault(form, []).append((tag, lemma, count))
    return readings


def main() -> None:
    out = Path(sys.argv[1])
    rng = random.Random(SEED)
    sample = []
    for raw_dir in RAW_DIRS:
        for path in sorted(raw_dir.glob("*/*.jsonl.gz")):
            abstracts = [text for _, text in cased_abstracts(raw_dir, path.parent.name, int(path.name.split(".")[0]))]
            sample += rng.sample(abstracts, min(PER_CELL, len(abstracts)))
    print(f"sample: {len(sample)} abstracts", flush=True)

    start = time.perf_counter()
    small = lemma_votes(sample, workers=1, model="en_core_web_sm")
    small_seconds = time.perf_counter() - start
    start = time.perf_counter()
    trf = lemma_votes(sample, model="en_core_web_trf", gpu=True)
    trf_seconds = time.perf_counter() - start
    tokens = sum(small.values())

    small_r, trf_r = as_readings(small), as_readings(trf)
    both = [
        form for form in small_r
        if sum(c for *_, c in small_r[form]) >= MIN_VOTES and sum(c for *_, c in trf_r.get(form, ())) >= MIN_VOTES
    ]
    decisions = {form: (voted_lemma(form, small_r), voted_lemma(form, trf_r)) for form in both}
    differ = {form: pair for form, pair in decisions.items() if pair[0] != pair[1]}
    merged_small = sum(1 for a, _ in decisions.values() if a)
    merged_trf = sum(1 for _, b in decisions.values() if b)
    frequency = {form: sum(c for *_, c in small_r[form]) for form in differ}

    full_tokens = pl.read_parquet("data/processed/lemma_votes.parquet")["count"].sum()
    summary = {
        "abstracts": len(sample),
        "tokens": tokens,
        "small_cpu_one_process_tokens_per_second": round(tokens / small_seconds),
        "trf_gpu_tokens_per_second": round(tokens / trf_seconds),
        "full_corpus_tokens": int(full_tokens),
        "estimated_full_trf_gpu_hours": round(full_tokens / (tokens / trf_seconds) / 3600, 2),
        "types_compared": len(both),
        "merged_by_small": merged_small,
        "merged_by_trf": merged_trf,
        "decisions_differ": len(differ),
        "differences": sorted(
            ({"type": f, "small": differ[f][0], "trf": differ[f][1], "readings": frequency[f]} for f in differ),
            key=lambda row: -row["readings"],
        ),
    }
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "differences"}, indent=1))
    for row in summary["differences"][:40]:
        print(row)


if __name__ == "__main__":
    main()
