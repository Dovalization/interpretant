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
MAX_BATCH_TOKENS = 16_384  # padded tokens per GPU batch: 128 sentences of 128 tokens
HALF = True  # float16 autocast on CUDA; every shard of one store uses the same setting
ENCODERS = ("xl-lexeme", "deberta")
DEV_MANIFEST = STORE / "dev_usages.jsonl.gz"
_TOKENIZER: object = None
_KIND = ""


def store_paths(kind: str, which: str) -> tuple[Path, Path, Path]:
    """Manifest, shard folder and index for an encoder ("xl-lexeme" or "deberta") and a
    manifest ("main" or "dev"); the first store keeps its original names."""
    manifest = MANIFEST if which == "main" else DEV_MANIFEST
    if (kind, which) == ("xl-lexeme", "main"):
        return manifest, SHARDS, INDEX
    return manifest, STORE / f"vectors_{kind}_{which}", STORE / f"index_{kind}_{which}.json"


def _start_tokenizer(kind: str) -> None:
    """Load the encoder's tokenizer once in a worker process."""
    global _TOKENIZER, _KIND  # noqa: PLW0603
    from transformers import AutoTokenizer  # noqa: PLC0415

    from interpretant.drift import pairs  # noqa: PLC0415

    name, revision = (
        (pairs.XL_LEXEME, pairs.XL_LEXEME_REVISION)
        if kind == "xl-lexeme"
        else (pairs.DEBERTA, pairs.DEBERTA_REVISION)
    )
    _TOKENIZER = AutoTokenizer.from_pretrained(name, revision=revision)  # type: ignore[no-untyped-call]
    _KIND = kind


def _prepare(rows: list[tuple[str, int, int]]) -> list[object]:
    """Token ids (XL-LEXEME) or ids with the target's token span (DeBERTa) for each row."""
    from interpretant.drift import pairs  # noqa: PLC0415

    usages = [Usage(text, start, end) for text, start, end in rows]
    if _KIND == "xl-lexeme":
        return [pairs.usage_input_ids(_TOKENIZER, usage) for usage in usages]
    prepare = pairs.TargetTokenEncoder.prepared
    holder = type("Holder", (), {"tokenizer": _TOKENIZER})()
    return [prepare(holder, usage) for usage in usages]


def save_shard(path: Path, keys: np.ndarray, vectors: np.ndarray, done: int, total: int) -> None:
    """Check and write one shard (on a background thread, while the GPU encodes the next)."""
    half = vectors.astype(np.float16)
    if not np.isfinite(half).all():
        raise ValueError(f"{path.name}: vectors overflow float16")
    partial = path.with_name(path.stem + ".partial.npz")
    np.savez(partial, keys=keys, vectors=half)
    partial.rename(path)
    print(f"{path.name}: {done:,} of {total:,}", flush=True)


def encode(kind: str = "xl-lexeme", which: str = "main") -> None:
    """Encode every distinct usage in shards of ``SHARD_SIZE``; finished shards are kept. While
    the GPU encodes one shard, worker processes build the next shard's token ids. DeBERTa on
    the main manifest encodes the full words only (round A′ and the dev data need no more)."""
    from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor  # noqa: PLC0415

    from interpretant.drift.pairs import TargetTokenEncoder, XLLexeme  # noqa: PLC0415

    manifest, shards, _ = store_paths(kind, which)
    full = set(FULL_WORDS)
    with gzip.open(manifest, "rt", encoding="utf-8") as source:
        todo = [
            row
            for row in map(json.loads, source)
            if row["encode"] and (kind == "xl-lexeme" or which == "dev" or row["word"] in full)
        ]
    shards.mkdir(parents=True, exist_ok=True)
    pending = [
        (first, shards / f"shard_{first // SHARD_SIZE:04d}.npz")
        for first in range(0, len(todo), SHARD_SIZE)
    ]
    pending = [(first, path) for first, path in pending if not path.exists()]
    with ProcessPoolExecutor(
        TOKENIZER_WORKERS, initializer=_start_tokenizer, initargs=(kind,)
    ) as pool:

        def submit(first: int) -> list[Future[list[object]]]:
            rows = [(r["text"], r["start"], r["end"]) for r in todo[first : first + SHARD_SIZE]]
            size = -(-len(rows) // TOKENIZER_WORKERS)
            return [pool.submit(_prepare, rows[k : k + size]) for k in range(0, len(rows), size)]

        upcoming = submit(pending[0][0]) if pending else []
        writer = ThreadPoolExecutor(1)
        saving: Future[None] | None = None
        xl = XLLexeme(batch_size=128, half=HALF) if kind == "xl-lexeme" else None
        deberta = TargetTokenEncoder(batch_size=64, half=HALF) if kind == "deberta" else None
        for position, (first, path) in enumerate(pending):
            current = upcoming
            if position + 1 < len(pending):
                upcoming = submit(pending[position + 1][0])
            items = [row for future in current for row in future.result()]
            if xl is not None:
                vectors = xl.encode_ids(items, by_length=True, max_tokens=MAX_BATCH_TOKENS)  # type: ignore[arg-type]
            else:
                assert deberta is not None
                vectors = deberta.encode_prepared(items, max_tokens=MAX_BATCH_TOKENS)  # type: ignore[arg-type]
            keys = np.array([r["key"] for r in todo[first : first + SHARD_SIZE]])
            if saving is not None:
                saving.result()
            saving = writer.submit(save_shard, path, keys, vectors, first + len(keys), len(todo))
        if saving is not None:
            saving.result()
        writer.shutdown(wait=True)


def index(kind: str = "xl-lexeme", which: str = "main") -> None:
    """Map every key to its shard and row."""
    _, shards, index_path = store_paths(kind, which)
    found: dict[str, list[object]] = {}
    for path in sorted(shards.glob("shard_*.npz")):
        for row, key in enumerate(np.load(path)["keys"]):
            found[str(key)] = [path.name, row]
    index_path.write_text(json.dumps(found), encoding="utf-8")
    print(f"{len(found):,} vectors indexed")


def dev() -> None:
    """Write the dev manifest: every usage the development data scores, tidied: check 2b's 186
    (round T), round U's 240, Claude's 300 development labels and round A's target-word cells."""
    sys.path.insert(0, str(Path(__file__).parent))
    import pilot_labellers as pilot  # type: ignore[import-not-found]  # noqa: PLC0415

    rows: list[dict[str, object]] = []
    for uid, (word, usage) in pilot.usages().items():
        rows.append({"set": "T", "id": uid, "word": word, **_usage_row(usage)})
    for line in Path("data/labels/round_u/raw/test.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows.append(
            {
                "set": "U",
                "id": r["id"],
                "word": r["word"],
                **_usage_row(Usage(r["text"], r["start"], r["end"])),
            }
        )
    for r in json.loads(Path("data/labels/round_u/raw/dev_set.json").read_text(encoding="utf-8")):
        rows.append(
            {
                "set": "mine",
                "id": r["id"],
                "word": r["word"],
                **_usage_row(Usage(r["text"], r["start"], r["end"])),
            }
        )
    for line in (
        Path("data/labels/known/raw/measured.jsonl").read_text(encoding="utf-8").splitlines()
    ):
        r = json.loads(line)
        if r["set"] == "case":
            rows.append(
                {
                    "set": "A",
                    "id": r["id"],
                    "word": r["word"],
                    **_usage_row(Usage(r["text"], r["start"], r["end"])),
                }
            )
    seen: set[str] = set()
    with gzip.open(DEV_MANIFEST, "wt", encoding="utf-8") as out:
        for row in rows:
            row["encode"] = row["key"] not in seen
            seen.add(str(row["key"]))
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows):,} dev usages, {len(seen):,} distinct")


def _usage_row(usage: Usage) -> dict[str, object]:
    tidied = tidy_usage(usage)
    return {
        "key": key_of(tidied.text, tidied.start, tidied.end),
        "text": tidied.text,
        "start": tidied.start,
        "end": tidied.end,
    }


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "export":
        export(sys.argv[2])
    elif command == "parts":
        print("\n".join(parts()))
    elif command in ("encode", "index"):
        kind = sys.argv[2] if len(sys.argv) > 2 else "xl-lexeme"
        which = sys.argv[3] if len(sys.argv) > 3 else "main"
        {"encode": encode, "index": index}[command](kind, which)
    else:
        {"merge": merge, "dev": dev}[command]()
