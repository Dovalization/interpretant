"""The spelling audit's items: each pair of the sample and the census with corpus sentences.

`docs/spellings.md` fixes the audit. Every pair of the sample and of the census (each form once)
becomes a blind item: the form, the standard, and up to two sentences from kept works for each,
drawn by the smallest sha256 of (seed, work id, offset), so the draw does not depend on part
order. The items are shuffled (seed 20261119) into batches for the agent judges; key.json maps
each blind id to its pair and its sets; calibration.json holds the 30 sample pairs the author
judges (seed 20261121). Run from the repository root on the run machine:
`python scripts/review/spelling_audit.py REVIEW_DIR OUT`, REVIEW_DIR holding the proposal's
sample.parquet and census.parquet.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

from interpretant.catalog.abstracts import load_abstracts
from interpretant.catalog.config import read_catalog
from interpretant.catalog.static_view import load_static_view, static_view_parts

SEED = 20261119
CALIBRATION_SEED = 20261121
CALIBRATION = 30
SENTENCES = 2
BATCH = 125


def key(work_id: str, start: int) -> str:
    """The sentence draw's key."""
    return hashlib.sha256(f"{SEED}:{work_id}:{start}".encode()).hexdigest()


def main() -> None:
    """Build the items, the batches, the key and the calibration set."""
    review, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    sample = pl.read_parquet(review / "sample.parquet")
    census = pl.read_parquet(review / "census.parquet")
    pairs: dict[str, dict[str, object]] = {}
    for row in sample.iter_rows(named=True):
        pairs[row["form"]] = {
            "form": row["form"],
            "standard": row["standard"],
            "sets": ["sample"],
            "drawn": row["drawn"],
        }
    for row in census.iter_rows(named=True):
        entry = pairs.setdefault(
            row["form"],
            {"form": row["form"], "standard": row["standard"], "sets": [], "drawn": None},
        )
        entry["sets"] = [*entry["sets"], f"census:{row['reason']}"]  # type: ignore[misc]
    needed = {str(p["form"]) for p in pairs.values()} | {str(p["standard"]) for p in pairs.values()}

    catalog = read_catalog()
    kept = (
        load_abstracts(catalog)
        .select("work_id", "text", "sentences", "dropped_reason")
        .filter(pl.col("dropped_reason").is_null())
        .drop("dropped_reason")
        .collect()
    )
    kept_ids = frozenset(kept["work_id"])
    best: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for done, part in enumerate(static_view_parts(catalog), start=1):
        rows = (
            load_static_view(catalog, part, columns=["work_id", "tokens", "starts"])
            .explode("tokens", "starts")
            .filter(pl.col("tokens").is_in(sorted(needed)))
        )
        for work_id, token, start in rows.iter_rows():
            if work_id in kept_ids:
                best[token].append((key(work_id, start), work_id, start))
                if len(best[token]) > 8 * SENTENCES:
                    best[token] = sorted(best[token])[:SENTENCES]
        sys.stdout.write(f"finding sentences: {done}/88 parts\n")
        sys.stdout.flush()
    chosen = {token: sorted(found)[:SENTENCES] for token, found in best.items()}
    wanted = {work_id for found in chosen.values() for _, work_id, _ in found}
    texts = {
        row[0]: (row[1], row[2])
        for row in kept.filter(pl.col("work_id").is_in(sorted(wanted))).iter_rows()
    }

    def sentences_of(token: str) -> list[str]:
        found = []
        for _, work_id, start in chosen.get(token, []):
            text, bounds = texts[work_id]
            sentence = next((s for s in bounds if s["start"] <= start < s["end"]), None)
            if sentence is not None:
                found.append(text[sentence["start"] : sentence["end"]].strip()[:300])
        return found

    rng = random.Random(SEED)
    order = sorted(pairs)
    rng.shuffle(order)
    items, key_map = [], {}
    for index, form in enumerate(order):
        entry = pairs[form]
        item_id = f"S{index:03d}"
        key_map[item_id] = entry
        items.append(
            {
                "id": item_id,
                "form": form,
                "standard": entry["standard"],
                "form_sentences": sentences_of(form),
                "standard_sentences": sentences_of(str(entry["standard"])),
            }
        )
    for batch, start in enumerate(range(0, len(items), BATCH)):
        with (out / f"batch_{batch:02d}.jsonl").open("w", encoding="utf-8") as handle:
            for item in items[start : start + BATCH]:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (out / "key.json").write_text(json.dumps(key_map, indent=1, ensure_ascii=False))
    sample_ids = sorted(item_id for item_id, entry in key_map.items() if "sample" in entry["sets"])  # type: ignore[operator]
    calibration = random.Random(CALIBRATION_SEED).sample(sample_ids, CALIBRATION)
    by_id = {item["id"]: item for item in items}
    (out / "calibration.json").write_text(
        json.dumps(
            [by_id[item_id] for item_id in sorted(calibration)], indent=1, ensure_ascii=False
        )
    )
    missing = sum(1 for item in items if not item["form_sentences"])
    sys.stdout.write(
        f"{len(items)} items in {(len(items) + BATCH - 1) // BATCH} batches; "
        f"{len(sample_ids)} from the sample; {CALIBRATION} for calibration; "
        f"{missing} forms without a sentence\n"
    )


if __name__ == "__main__":
    main()
