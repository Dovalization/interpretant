"""The matching audit's items: kept, left out and missed usages (`docs/usages_audit.md`).

Draws, with seed 20261304, 20 drawn in-scope usages per word spread over decades (precision), and
10 usages per word the table left out of scope. Worker processes run an independent
case-insensitive search over every kept work's text outside removed ranges: each word, its
plurals, its other spellings from the spelling map and theirs, alone or as a hyphen part (phrases
with a space or a hyphen between their words). A hit no mention of the word overlaps is a
candidate miss; 20 per word are drawn. Writes the items in shuffled batches (they quote corpus
sentences and stay out of git), the key, the calibration set (seed 20261305: 30 kept, 15 left out,
15 missed) and `counts.json`, each word's in-scope usages and candidate misses for the missed-share
estimate. Held-out words are left out of the items and counted, never named. Run from the
repository root: `python scripts/review/usages_audit.py OUT [PROCESSES]`.
"""

from __future__ import annotations

import json
import random
import re
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from rich.console import Console

from interpretant.catalog.abstracts import load_abstracts_part
from interpretant.catalog.config import CatalogConfig, read_catalog
from interpretant.catalog.parts import part_path
from interpretant.catalog.sentence_view import load_sentence_view_part
from interpretant.catalog.spellings import load_spellings
from interpretant.catalog.words import Word, load_words
from interpretant.cli.progress import LiveProgress
from interpretant.corpus.sentence_view import merge_ranges
from interpretant.pipelines.workers import map_parts
from interpretant.usages.identifiers import build_draw_key

AUDIT_SEED = 20261304
CALIBRATION_SEED = 20261305
KEPT_PER_WORD = 20
LEFT_OUT_PER_WORD = 10
MISSED_PER_WORD = 20
CALIBRATION = {"kept": 30, "left_out": 15, "missed": 15}
BATCH = 50
WIDTH = 600
COLUMNS = [
    "work_id",
    "word",
    "content_key",
    "sentence_start",
    "sentence_end",
    "target_start",
    "target_end",
    "position",
    "pos",
    "entity",
    "in_scope",
    "drawn",
]


def allocate(available: dict[int, int], total: int) -> dict[int, int]:
    """An equal share per decade; a decade short of its share gives all it has, the rest spread."""
    shares = {}
    remaining = total
    by_size = sorted(available, key=lambda decade: (available[decade], decade))
    for index, decade in enumerate(by_size):
        share = min(available[decade], remaining // (len(by_size) - index))
        shares[decade] = share
        remaining -= share
    return shares


def audit_key(work_id: str, word: str, place: str) -> str:
    """The seeded key ordering one usage or hit in the draws."""
    return build_draw_key(AUDIT_SEED, work_id, word, place)


def forms_of(word: Word, spellings: dict[str, str]) -> list[str]:
    """The written forms the search looks for: the word, its other spellings, and their plurals."""
    if word.match == "form":
        return [word.lemma]
    bases = {word.lemma} | {form for form, standard in spellings.items() if standard == word.lemma}
    forms = set(bases)
    for base in bases:
        forms |= {base + "s", base + "es"}
        if base.endswith("y"):
            forms.add(base[:-1] + "ies")
    return sorted(forms, key=len, reverse=True)


def search_pattern(word: Word, spellings: dict[str, str]) -> re.Pattern[str]:
    """The case-insensitive search for one word, alone or as a hyphen part."""
    if word.is_phrase:
        modifier, head = word.lemma.split(" ")
        heads = "|".join(re.escape(form) for form in forms_of(_single(head), spellings))
        body = rf"{re.escape(modifier)}[\s-]+(?:{heads})"
    else:
        body = "|".join(re.escape(form) for form in forms_of(word, spellings))
    return re.compile(rf"(?<![^\W_])(?:{body})(?![^\W_])", re.IGNORECASE)


def _single(lemma: str) -> Word:
    """A phrase's head as a word matched by lemma, for its forms."""
    return Word(lemma=lemma, pos=("NOUN",), match="lemma", held_out=False)


@dataclass(frozen=True)
class SearchJob:
    """What a search worker needs (picklable)."""

    catalog: CatalogConfig
    patterns: dict[str, re.Pattern[str]]


def search_part(part: str, job: SearchJob) -> dict[str, Any]:
    """One part's candidate misses per word: their count and the smallest-keyed few."""
    abstracts = load_abstracts_part(
        job.catalog, part, ["work_id", "text", "removals", "dropped_reason"]
    ).filter(pl.col("dropped_reason").is_null())
    usages = pl.read_parquet(
        part_path(job.catalog.usages.path, part, ".parquet"),
        columns=["work_id", "word", "target_start", "target_end"],
    )
    mentions: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for work_id, word, start, end in usages.iter_rows():
        mentions[(work_id, word)].append((start, end))
    sentences: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for work_id, start, end in (
        load_sentence_view_part(job.catalog, part)
        .sort("work_id", "start")
        .select("work_id", "start", "end")
        .iter_rows()
    ):
        sentences[work_id].append((start, end))
    counts: Counter[str] = Counter()
    kept: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for work_id, text, ranges, _ in abstracts.iter_rows():
        removed = merge_ranges((r["start"], r["end"]) for r in ranges)
        removed_starts = [start for start, _ in removed]
        bounds = sentences[work_id]
        sentence_starts = [start for start, _ in bounds]
        for word, pattern in job.patterns.items():
            found = mentions.get((work_id, word), [])
            for hit in pattern.finditer(text):
                start, end = hit.span()
                inside = bisect_right(removed_starts, start) - 1
                if inside >= 0 and start < removed[inside][1]:
                    continue
                if any(low < end and start < high for low, high in found):
                    continue
                index = bisect_right(sentence_starts, start) - 1
                if index < 0 or end > bounds[index][1]:
                    continue
                counts[word] += 1
                kept[word].append(
                    {
                        "work_id": work_id,
                        "word": word,
                        "target_start": start,
                        "target_end": end,
                        "sentence_start": bounds[index][0],
                        "sentence_end": bounds[index][1],
                        "part": part,
                        "key": audit_key(work_id, word, f"{start}:{end}"),
                    }
                )
    smallest = {
        word: sorted(hits, key=lambda hit: hit["key"])[:MISSED_PER_WORD]
        for word, hits in kept.items()
    }
    return {"counts": counts, "smallest": smallest}


def load_usages_with_decades(catalog: CatalogConfig) -> pl.DataFrame:
    """The working usages table's audit columns, with each row's part and decade."""
    return pl.concat(
        [
            pl.read_parquet(path, columns=COLUMNS).with_columns(
                pl.lit(path.stem).alias("part"),
                pl.lit(int(path.stem.rsplit("_", 1)[1])).alias("decade"),
            )
            for path in sorted(catalog.usages.path.glob("*.parquet"))
        ]
    )


def with_keys(usages: pl.DataFrame) -> pl.DataFrame:
    """The usages with their audit keys."""
    return usages.with_columns(
        pl.struct("work_id", "word", "content_key")
        .map_elements(
            lambda row: audit_key(row["work_id"], row["word"], row["content_key"]),
            return_dtype=pl.String,
        )
        .alias("key")
    )


def draw_kept(usages: pl.DataFrame) -> pl.DataFrame:
    """20 drawn in-scope usages per word, spread over decades, the smallest keys in each."""
    drawn = with_keys(usages.filter(pl.col("drawn") & pl.col("in_scope")))
    chosen = []
    for (_,), rows in drawn.group_by("word"):
        counts = dict(rows.group_by("decade").len().iter_rows())
        for decade, share in allocate(counts, KEPT_PER_WORD).items():
            chosen.append(rows.filter(pl.col("decade") == decade).sort("key").head(share))
    return pl.concat(chosen)


def draw_left_out(usages: pl.DataFrame) -> pl.DataFrame:
    """10 usages per word the table left out of scope, the smallest keys."""
    left_out = with_keys(usages.filter(~pl.col("in_scope")))
    return left_out.sort("key").group_by("word", maintain_order=True).head(LEFT_OUT_PER_WORD)


def marked(text: str, row: dict[str, Any]) -> str:
    """The sentence, or a window of it, with the target between ⟨t⟩ and ⟨/t⟩."""
    sentence = text[row["sentence_start"] : row["sentence_end"]]
    start = row["target_start"] - row["sentence_start"]
    end = row["target_end"] - row["sentence_start"]
    shown = f"{sentence[:start]}⟨t⟩{sentence[start:end]}⟨/t⟩{sentence[end:]}".strip()
    if len(shown) <= WIDTH:
        return shown
    middle = shown.index("⟨t⟩")
    begin = max(0, min(middle - WIDTH // 2, len(shown) - WIDTH))
    prefix = "…" if begin else ""
    suffix = "…" if begin + WIDTH < len(shown) else ""
    return prefix + shown[begin : begin + WIDTH] + suffix


def left_out_reason(row: dict[str, Any], word: Word) -> str:
    """Why the table left a usage out: its name, or its part of speech."""
    if row["entity"] is not None and (row["position"] is not None or row["pos"] in word.pos):
        return f"inside a name ({row['entity']})"
    return f"part of speech {row['pos']}"


def main() -> None:
    """Draw the three samples, build and shuffle the items, and write them with their key."""
    out = Path(sys.argv[1])
    processes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    out.mkdir(parents=True, exist_ok=True)
    catalog = read_catalog()
    words = {word.lemma: word for word in load_words(catalog)}
    held_out = {lemma for lemma, word in words.items() if word.held_out}
    spellings = {spelling.form: spelling.standard for spelling in load_spellings(catalog)}
    usages = load_usages_with_decades(catalog).filter(~pl.col("word").is_in(held_out))
    kept = draw_kept(usages)
    left_out = draw_left_out(usages)
    patterns = {
        lemma: search_pattern(word, spellings)
        for lemma, word in words.items()
        if lemma not in held_out
    }
    parts = sorted(usages["part"].unique())
    with LiveProgress(Console()) as live:
        live.stage("searching for missed mentions", total=len(parts))
        results = map_parts(
            search_part,
            dict.fromkeys(parts, SearchJob(catalog, patterns)),
            process_count=processes,
            progress=live,
        )
    candidates: Counter[str] = Counter()
    missed_pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results.values():
        candidates.update(result["counts"])
        for word, hits in result["smallest"].items():
            missed_pool[word].extend(hits)
    missed = [
        hit
        for hits in missed_pool.values()
        for hit in sorted(hits, key=lambda h: h["key"])[:MISSED_PER_WORD]
    ]

    samples = (
        [("kept", row) for row in kept.to_dicts()]
        + [("left_out", row) for row in left_out.to_dicts()]
        + [("missed", row) for row in missed]
    )
    texts: dict[str, str] = {}
    by_part: dict[str, set[str]] = defaultdict(set)
    for _, row in samples:
        by_part[row["part"]].add(row["work_id"])
    for part, ids in by_part.items():
        frame = load_abstracts_part(catalog, part, ["work_id", "text"])
        texts |= dict(frame.filter(pl.col("work_id").is_in(sorted(ids))).iter_rows())

    rng = random.Random(AUDIT_SEED)
    rng.shuffle(samples)
    items, key_map = [], {}
    for index, (kind, row) in enumerate(samples):
        item_id = f"U{index:04d}"
        word = words[row["word"]]
        item = {
            "id": item_id,
            "kind": kind,
            "word": row["word"],
            "parts_of_speech": list(word.pos),
            "sentence": marked(texts[row["work_id"]], row),
        }
        if kind == "left_out":
            item["reason"] = left_out_reason(row, word)
        items.append(item)
        key_map[item_id] = {
            "kind": kind,
            "work_id": row["work_id"],
            "word": row["word"],
            "target_start": row["target_start"],
            "target_end": row["target_end"],
            "part": row["part"],
        }
    for batch, start in enumerate(range(0, len(items), BATCH)):
        with (out / f"batch_{batch:02d}.jsonl").open("w", encoding="utf-8") as handle:
            for item in items[start : start + BATCH]:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (out / "key.json").write_text(json.dumps(key_map, indent=1) + "\n")
    calibration_rng = random.Random(CALIBRATION_SEED)
    calibration = []
    for kind, size in CALIBRATION.items():
        of_kind = sorted(item_id for item_id, entry in key_map.items() if entry["kind"] == kind)
        calibration += calibration_rng.sample(of_kind, min(size, len(of_kind)))
    (out / "calibration.json").write_text(json.dumps(sorted(calibration), indent=1) + "\n")
    in_scope = dict(usages.filter("in_scope").group_by("word").len().sort("word").iter_rows())
    counts = {
        "in_scope_usages": in_scope,
        "candidate_misses": dict(sorted(candidates.items())),
        "items": dict(Counter(entry["kind"] for entry in key_map.values())),
        "held_out_words": len(held_out),
    }
    (out / "counts.json").write_text(json.dumps(counts, indent=1) + "\n")
    sys.stdout.write(
        f"{len(items)} items {counts['items']} in {(len(items) + BATCH - 1) // BATCH} batches; "
        f"{sum(candidates.values())} candidate misses over {len(candidates)} words\n"
    )


if __name__ == "__main__":
    main()
