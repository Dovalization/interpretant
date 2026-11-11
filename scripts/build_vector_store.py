"""The shared vector store: every usage the next rounds and scans need, encoded once with
XL-LEXEME (ROADMAP item 8d, 2026-11-11).

    uv run python scripts/build_vector_store.py export   # on the Mac: usages to encode
    uv run python scripts/build_vector_store.py encode   # on the GPU machine: resumable shards
    uv run python scripts/build_vector_store.py index    # back on the Mac: key -> shard, row

Scope (quality rules version 2 applied to every cell; one occurrence per abstract, with the
regular plural, tidied as every round tidies usages):

- the grid's every abstract containing a full word: the target words, round A′'s words, the
  control words and the shortlist;
- the grid's long-list words, up to 300 abstracts per cell (seed 1), for the XL-LEXEME scan;
- every word-targeted cell of the full words under ``data/raw/openalex_words``.

Vectors are stored as float16 under ``data/store/``, which is not committed; the manifest gives
each usage's word, field, decade, year, source and key (``key_of`` as in the scorers).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from interpretant.corpus import quality
from interpretant.drift.classifier import tidy_usage
from interpretant.drift.pairs import Usage
from interpretant.drift.senses import Sampling, cased_abstracts, sample_occurrences

GRID = Path("data/raw/openalex")
WORDS = Path("data/raw/openalex_words")
STORE = Path("data/store")
MANIFEST = STORE / "usages.jsonl.gz"
SHARDS = STORE / "vectors"
INDEX = STORE / "index.json"
LONGLIST = Path("data/discovery/longlist.json")
SHARED_CACHE = Path("data/processed/openalex/shared_sentences.json")
SHARD_SIZE = 50_000
FULL_WORDS = (
    ["attention", "prompt", "platform", "network", "alignment", "component", "having"]
    + ["hallucination", "transformer", "cloud", "embedding"]
    + [
        line.strip()
        for line in Path("data/control_words_rung3.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
)
UNCAPPED = Sampling(cap=10**7, plural=True, quality=True)
LONGLIST_SAMPLING = Sampling(cap=300, plural=True, quality=True)


def key_of(text: str, start: int, end: int) -> str:
    return hashlib.sha256(f"{start}:{end}:{text}".encode()).hexdigest()[:20]


def years(root: Path, field: str, decade: int) -> dict[str, int]:
    with gzip.open(root / field / f"{decade}.jsonl.gz", "rt", encoding="utf-8") as cell:
        return {
            str(record["id"]): int(record.get("year") or decade) for record in map(json.loads, cell)
        }


def cells(root: Path) -> list[tuple[str, int]]:
    return sorted((p.parent.name, int(p.name[:4])) for p in root.glob("*/*.jsonl.gz"))


def export() -> None:
    """Write the manifest: every usage in scope, once per (source, word, abstract)."""
    shared = quality.grid_shared_sentences(GRID, SHARED_CACHE)
    longlist = json.loads(LONGLIST.read_text(encoding="utf-8"))
    full = sorted(set(FULL_WORDS) | set(longlist["shortlist"]))
    sampled = sorted(set(longlist["all"]) - set(full))
    STORE.mkdir(parents=True, exist_ok=True)
    written, seen = 0, set()
    with gzip.open(MANIFEST, "wt", encoding="utf-8") as out:

        def emit(root: Path, field: str, decade: int, plan: list[tuple[str, Sampling]]) -> None:
            nonlocal written
            documents = cased_abstracts(root, field, decade)
            cleaned = quality.clean_cells({(field, decade): documents}, shared)
            year_of = years(root, field, decade)
            for word, sampling in plan:
                for occurrence in sample_occurrences(cleaned, word, seed=1, sampling=sampling):
                    usage = tidy_usage(Usage(occurrence.context, occurrence.start, occurrence.end))
                    key = key_of(usage.text, usage.start, usage.end)
                    row = {
                        "key": key,
                        "word": word,
                        "source": "grid" if root == GRID else f"words/{root.name}",
                        "field": field,
                        "decade": decade,
                        "year": year_of.get(occurrence.work_id, decade),
                        "work_id": occurrence.work_id,
                        "text": usage.text,
                        "start": usage.start,
                        "end": usage.end,
                        "encode": key not in seen,
                    }
                    seen.add(key)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1

        grid_plan = [(w, UNCAPPED) for w in full] + [(w, LONGLIST_SAMPLING) for w in sampled]
        for field, decade in cells(GRID):
            emit(GRID, field, decade, grid_plan)
            print(f"grid {field} {decade}s: {written:,} usages so far", flush=True)
        for word_dir in sorted(p for p in WORDS.iterdir() if p.is_dir() and p.name in full):
            for field, decade in cells(word_dir):
                emit(word_dir, field, decade, [(word_dir.name, UNCAPPED)])
            print(f"words/{word_dir.name}: {written:,} usages so far", flush=True)
    print(f"{written:,} usages, {len(seen):,} distinct to encode")


def encode() -> None:
    """Encode every distinct usage in shards of ``SHARD_SIZE``; finished shards are kept."""
    from interpretant.drift.pairs import XLLexeme  # noqa: PLC0415

    with gzip.open(MANIFEST, "rt", encoding="utf-8") as manifest:
        todo = [row for row in map(json.loads, manifest) if row["encode"]]
    SHARDS.mkdir(parents=True, exist_ok=True)
    encoder = XLLexeme(batch_size=128)
    for first in range(0, len(todo), SHARD_SIZE):
        path = SHARDS / f"shard_{first // SHARD_SIZE:04d}.npz"
        if path.exists():
            continue
        rows = todo[first : first + SHARD_SIZE]
        vectors = encoder.encode([Usage(r["text"], r["start"], r["end"]) for r in rows])
        partial = path.with_name(path.stem + ".partial.npz")
        np.savez(
            partial, keys=np.array([r["key"] for r in rows]), vectors=vectors.astype(np.float16)
        )
        partial.rename(path)
        print(f"{path.name}: {first + len(rows):,} of {len(todo):,}", flush=True)


def index() -> None:
    """Map every key to its shard and row."""
    found: dict[str, list[object]] = {}
    for path in sorted(SHARDS.glob("shard_*.npz")):
        for row, key in enumerate(np.load(path)["keys"]):
            found[str(key)] = [path.name, row]
    INDEX.write_text(json.dumps(found), encoding="utf-8")
    print(f"{len(found):,} vectors indexed")


if __name__ == "__main__":
    {"export": export, "encode": encode, "index": index}[sys.argv[1]]()
