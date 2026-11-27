"""Drawing the matching audit's items from the usages table (`docs/usages_audit.md`).

Worker processes run the recall search part by part: an independent case-insensitive search over
every kept work's text outside removed ranges, for each word's forms alone or as a hyphen part. The
draws themselves are ``review.usages_audit``'s. Writes the items in shuffled batches of 50
(``batch_NN.jsonl``: they quote corpus sentences and stay out of git), ``key.json``,
``calibration.json`` and ``counts.json``. A redraw after a failed bar takes its own seed and skips
the calibration, keeping the calibration of record.
"""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from interpretant.catalog.abstracts import load_abstracts_part
from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.parts import part_path
from interpretant.catalog.sentence_view import load_sentence_view_part
from interpretant.catalog.spellings import load_spellings
from interpretant.catalog.words import load_words
from interpretant.corpus.sentence_view import merge_ranges
from interpretant.pipelines.progress import SILENT, Progress
from interpretant.pipelines.workers import map_parts
from interpretant.review.usages_audit import (
    BATCH,
    COLUMNS,
    MISSED_PER_WORD,
    Sample,
    build_audit_key,
    build_items,
    build_search_pattern,
    count_usages,
    draw_calibration,
    draw_kept,
    draw_left_out,
    draw_missed,
)


@dataclass(frozen=True)
class AuditDraw:
    """What a draw wrote: its items by kind, its batches, and the candidate misses found."""

    items_by_kind: dict[str, int]
    batch_count: int
    candidate_count: int
    candidate_word_count: int

    @property
    def item_count(self) -> int:
        """All the items drawn."""
        return sum(self.items_by_kind.values())


@dataclass(frozen=True)
class SearchJob:
    """What a search worker needs (picklable)."""

    catalog: CatalogConfig
    patterns: dict[str, re.Pattern[str]]
    seed: int


@dataclass(frozen=True)
class PartMisses:
    """One part's candidate misses per word: their count and the smallest-keyed few."""

    counts: Counter[str]
    smallest: dict[str, list[dict[str, Any]]]


def draw_usages_audit(
    catalog: CatalogConfig,
    out: Path,
    *,
    seed: int,
    skip_calibration: bool = False,
    process_count: int,
    progress: Progress = SILENT,
) -> AuditDraw:
    """Draw the three samples and write their items, key, calibration set and counts to ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    words = {word.lemma: word for word in load_words(catalog)}
    spellings = {spelling.form: spelling.standard for spelling in load_spellings(catalog)}
    usages = load_usages_with_decades(catalog)
    kept = draw_kept(usages, seed)
    left_out = draw_left_out(usages, seed)
    patterns = {lemma: build_search_pattern(word, spellings) for lemma, word in words.items()}
    parts = sorted(usages["part"].unique())
    progress.stage("searching for missed mentions", total=len(parts))
    results = map_parts(
        match_missed_in_part,
        dict.fromkeys(parts, SearchJob(catalog, patterns, seed)),
        process_count=process_count,
        progress=progress,
    )
    candidates: Counter[str] = Counter()
    for result in results.values():
        candidates.update(result.counts)
    missed = draw_missed(result.smallest for result in results.values())
    samples: list[Sample] = (
        [("kept", row) for row in kept.to_dicts()]
        + [("left_out", row) for row in left_out.to_dicts()]
        + [("missed", row) for row in missed]
    )
    items, key = build_items(samples, load_texts(catalog, samples), words, seed)
    batch_count = save_batches(items, out)
    (out / "key.json").write_text(json.dumps(key, indent=1) + "\n")
    if not skip_calibration:
        (out / "calibration.json").write_text(json.dumps(draw_calibration(key), indent=1) + "\n")
    counts = count_usages(usages, candidates, key, words)
    (out / "counts.json").write_text(json.dumps(counts, indent=1) + "\n")
    return AuditDraw(
        items_by_kind=counts["items"],
        batch_count=batch_count,
        candidate_count=sum(candidates.values()),
        candidate_word_count=len(candidates),
    )


def load_usages_with_decades(
    catalog: CatalogConfig, columns: Sequence[str] = COLUMNS
) -> pl.DataFrame:
    """The usages table's ``columns`` (the audit's by default), with each row's part and decade."""
    return pl.concat(
        [
            pl.read_parquet(path, columns=list(columns)).with_columns(
                pl.lit(path.stem).alias("part"),
                pl.lit(int(path.stem.rsplit("_", 1)[1])).alias("decade"),
            )
            for path in sorted(catalog.usages.path.glob("*.parquet"))
        ]
    )


def load_texts(catalog: CatalogConfig, samples: list[Sample]) -> dict[str, str]:
    """The text of every work a sample comes from, by work id."""
    ids_by_part: dict[str, set[str]] = defaultdict(set)
    for _, row in samples:
        ids_by_part[row["part"]].add(row["work_id"])
    texts: dict[str, str] = {}
    for part, ids in ids_by_part.items():
        frame = load_abstracts_part(catalog, part, ["work_id", "text"])
        texts |= dict(frame.filter(pl.col("work_id").is_in(sorted(ids))).iter_rows())
    return texts


def save_batches(items: list[dict[str, Any]], out: Path) -> int:
    """Write the items as ``batch_NN.jsonl`` files of 50; how many were written."""
    batch_count = 0
    for batch_count, start in enumerate(range(0, len(items), BATCH), start=1):
        with (out / f"batch_{batch_count - 1:02d}.jsonl").open("w", encoding="utf-8") as handle:
            for item in items[start : start + BATCH]:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return batch_count


def match_missed_in_part(part: str, job: SearchJob) -> PartMisses:
    """One part's hits of each word that no mention of it overlaps, inside a kept sentence."""
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
    hits_by_word: dict[str, list[dict[str, Any]]] = defaultdict(list)
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
                hits_by_word[word].append(
                    {
                        "work_id": work_id,
                        "word": word,
                        "target_start": start,
                        "target_end": end,
                        "sentence_start": bounds[index][0],
                        "sentence_end": bounds[index][1],
                        "part": part,
                        "key": build_audit_key(job.seed, work_id, word, f"{start}:{end}"),
                    }
                )
    smallest = {
        word: sorted(hits, key=lambda hit: hit["key"])[:MISSED_PER_WORD]
        for word, hits in hits_by_word.items()
    }
    return PartMisses(counts, smallest)
