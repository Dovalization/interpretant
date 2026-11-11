"""The shared vector store: every usage the next rounds and scans need, encoded once with
XL-LEXEME (ROADMAP item 8d, 2026-11-11).

    uv run python scripts/build_vector_store.py export [PART]  # usages to encode, by part
    uv run python scripts/build_vector_store.py merge          # one manifest, duplicates marked
    uv run python scripts/build_vector_store.py encode         # on the GPU: resumable shards
    uv run python scripts/build_vector_store.py index          # key -> shard, row

A part is a grid field or "words" (the word-targeted cells); parts run in parallel.

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
from interpretant.drift.senses import Sampling, cased_abstracts, sample_many

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


def parts() -> list[str]:
    return sorted({field for field, _ in cells(GRID)}) + ["words"]


def export(part: str) -> None:
    """Write one part of the manifest: every usage in scope in a grid field or the word cells."""
    shared = quality.grid_shared_sentences(GRID, SHARED_CACHE)
    longlist = json.loads(LONGLIST.read_text(encoding="utf-8"))
    full = sorted(set(FULL_WORDS) | set(longlist["shortlist"]))
    sampled = sorted(set(longlist["all"]) - set(full))
    STORE.mkdir(parents=True, exist_ok=True)
    written = 0
    with gzip.open(STORE / f"part_{part}.jsonl.gz", "wt", encoding="utf-8") as out:

        def emit(root: Path, field: str, decade: int, plan: list[tuple[str, Sampling]]) -> None:
            nonlocal written
            documents = cased_abstracts(root, field, decade)
            cleaned = quality.clean_cells({(field, decade): documents}, shared)
            year_of = years(root, field, decade)
            drawn = sample_many(cleaned, plan, seed=1)
            for word, _ in plan:
                for occurrence in drawn[word]:
                    usage = tidy_usage(Usage(occurrence.context, occurrence.start, occurrence.end))
                    row = {
                        "key": key_of(usage.text, usage.start, usage.end),
                        "word": word,
                        "source": "grid" if root == GRID else f"words/{root.name}",
                        "field": field,
                        "decade": decade,
                        "year": year_of.get(occurrence.work_id, decade),
                        "work_id": occurrence.work_id,
                        "text": usage.text,
                        "start": usage.start,
                        "end": usage.end,
                    }
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1

        if part == "words":
            for word_dir in sorted(p for p in WORDS.iterdir() if p.is_dir() and p.name in full):
                for field, decade in cells(word_dir):
                    emit(word_dir, field, decade, [(word_dir.name, UNCAPPED)])
        else:
            plan = [(w, UNCAPPED) for w in full] + [(w, LONGLIST_SAMPLING) for w in sampled]
            for field, decade in cells(GRID):
                if field == part:
                    emit(GRID, field, decade, plan)
    print(f"part {part}: {written:,} usages", flush=True)


def merge() -> None:
    """Join the parts into one manifest, marking each distinct usage to be encoded once."""
    seen: set[str] = set()
    total = 0
    with gzip.open(MANIFEST, "wt", encoding="utf-8") as out:
        for part in parts():
            with gzip.open(STORE / f"part_{part}.jsonl.gz", "rt", encoding="utf-8") as source:
                for row in map(json.loads, source):
                    row["encode"] = row["key"] not in seen
                    seen.add(row["key"])
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    total += 1
    print(f"{total:,} usages, {len(seen):,} distinct to encode")


TOKENIZER_WORKERS = 4
_TOKENIZER: object = None


def _start_tokenizer() -> None:
    """Load XL-LEXEME's tokenizer once in a worker process."""
    global _TOKENIZER  # noqa: PLW0603
    from transformers import AutoTokenizer  # noqa: PLC0415

    from interpretant.drift.pairs import XL_LEXEME, XL_LEXEME_REVISION  # noqa: PLC0415

    _TOKENIZER = AutoTokenizer.from_pretrained(XL_LEXEME, revision=XL_LEXEME_REVISION)  # type: ignore[no-untyped-call]


def _input_ids(rows: list[tuple[str, int, int]]) -> list[list[int]]:
    from interpretant.drift.pairs import usage_input_ids  # noqa: PLC0415

    return [usage_input_ids(_TOKENIZER, Usage(text, start, end)) for text, start, end in rows]


def encode() -> None:
    """Encode every distinct usage in shards of ``SHARD_SIZE``; finished shards are kept. While
    the GPU encodes one shard, worker processes build the next shard's token ids."""
    from concurrent.futures import Future, ProcessPoolExecutor  # noqa: PLC0415

    from interpretant.drift.pairs import XLLexeme  # noqa: PLC0415

    with gzip.open(MANIFEST, "rt", encoding="utf-8") as manifest:
        todo = [row for row in map(json.loads, manifest) if row["encode"]]
    SHARDS.mkdir(parents=True, exist_ok=True)
    pending = [
        (first, SHARDS / f"shard_{first // SHARD_SIZE:04d}.npz")
        for first in range(0, len(todo), SHARD_SIZE)
    ]
    pending = [(first, path) for first, path in pending if not path.exists()]
    with ProcessPoolExecutor(TOKENIZER_WORKERS, initializer=_start_tokenizer) as pool:

        def submit(first: int) -> list[Future[list[list[int]]]]:
            rows = [(r["text"], r["start"], r["end"]) for r in todo[first : first + SHARD_SIZE]]
            size = -(-len(rows) // TOKENIZER_WORKERS)
            return [pool.submit(_input_ids, rows[k : k + size]) for k in range(0, len(rows), size)]

        upcoming = submit(pending[0][0]) if pending else []
        encoder = XLLexeme(batch_size=128)
        for position, (first, path) in enumerate(pending):
            current = upcoming
            if position + 1 < len(pending):
                upcoming = submit(pending[position + 1][0])
            ids = [row for future in current for row in future.result()]
            vectors = encoder.encode_ids(ids, by_length=True)
            if not np.isfinite(vectors.astype(np.float16)).all():
                raise ValueError(f"{path.name}: vectors overflow float16")
            keys = np.array([r["key"] for r in todo[first : first + SHARD_SIZE]])
            partial = path.with_name(path.stem + ".partial.npz")
            np.savez(partial, keys=keys, vectors=vectors.astype(np.float16))
            partial.rename(path)
            print(f"{path.name}: {first + len(keys):,} of {len(todo):,}", flush=True)


def index() -> None:
    """Map every key to its shard and row."""
    found: dict[str, list[object]] = {}
    for path in sorted(SHARDS.glob("shard_*.npz")):
        for row, key in enumerate(np.load(path)["keys"]):
            found[str(key)] = [path.name, row]
    INDEX.write_text(json.dumps(found), encoding="utf-8")
    print(f"{len(found):,} vectors indexed")


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export(sys.argv[2])
    elif sys.argv[1] == "parts":
        print("\n".join(parts()))
    else:
        {"merge": merge, "encode": encode, "index": index}[sys.argv[1]]()
