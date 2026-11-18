"""The sentence view audit's items (`docs/sentence_view.md`): calibration, precision, boundaries.

- **Whole abstracts:** the 50 abstracts of run 3's sentence-break review, rebuilt from run 3's
  table exactly as the author saw them, shuffled together with the 50 drawn for cut precision, so
  the judges cannot tell the calibration from the sample. The cut-precision abstracts are kept works
  with at least one cut: 6 per decade, then the 2 smallest keys among the rest, by the smallest
  sha256 of (seed 20261127, work id).
- **Target sentences:** 150 sentences with no exclusion that hold a single-word target outside
  removed ranges, by the static view's entries: 18 per decade, then the 6 smallest keys among the
  rest, by the smallest sha256 of (seed 20261128, work id, sentence number). Each item has the
  sentence before and the sentence after, as cut.

The items quote corpus text, so they stay out of git; key.json maps each blind id to its source.
Run from the repository root on the run machine:
`python scripts/review/sentence_audit.py RUN3_ABSTRACTS_DIR OUT [PROCESSES]`.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from rich.console import Console

from interpretant.catalog.abstracts import abstracts_parts, load_abstracts_part
from interpretant.catalog.config import CatalogConfig, read_catalog
from interpretant.catalog.parts import part_path
from interpretant.catalog.static_view import load_static_view
from interpretant.catalog.words import load_words
from interpretant.cli.progress import LiveProgress
from interpretant.corpus.sentence_view import merge_ranges
from interpretant.pipelines.workers import map_parts

PRECISION_SEED = 20261127
BOUNDARY_SEED = 20261128
SHUFFLE_SEED = 20261129
PRECISION = (6, 2)
BOUNDARIES = (18, 6)
KEEP_PER_PART = 40
RUN3_VERDICTS = Path("data/review/run3/verdicts.jsonl")


@dataclass(frozen=True)
class PartJob:
    """What a worker needs for one part (picklable)."""

    catalog: CatalogConfig
    targets: frozenset[str]


def draw_key(seed: int, *parts: object) -> str:
    """A draw key: the sha256 of the seed and the parts."""
    return hashlib.sha256(":".join(str(part) for part in (seed, *parts)).encode()).hexdigest()


def find_candidates(part: str, job: PartJob) -> dict[str, list[tuple[str, int, str, int]]]:
    """The part's smallest-keyed candidates per decade: (key, decade, work id, sentence number)."""
    abstracts = load_abstracts_part(
        job.catalog, part, ["work_id", "decade", "removals", "sentences", "dropped_reason"]
    ).filter(pl.col("dropped_reason").is_null())
    view = load_static_view(job.catalog, part, columns=["work_id", "tokens", "starts"])
    entries = {work_id: (tokens, starts) for work_id, tokens, starts in view.iter_rows()}
    sentences = pl.read_parquet(
        part_path(job.catalog.sentence_view.path, part, ".parquet"),
        columns=["work_id", "sentence_number", "start", "end", "exclusion"],
    )
    by_work: dict[str, list[tuple[int, int, int, str | None]]] = {}
    for work_id, number, start, end, exclusion in sentences.iter_rows():
        by_work.setdefault(work_id, []).append((start, end, number, exclusion))
    abstracts_found: list[tuple[str, int, str, int]] = []
    targets_found: list[tuple[str, int, str, int]] = []
    for work_id, decade, removals, work_sentences, _ in abstracts.iter_rows():
        if len(work_sentences) > 1:
            abstracts_found.append((draw_key(PRECISION_SEED, work_id), decade, work_id, -1))
        merged = merge_ranges((removal["start"], removal["end"]) for removal in removals)
        lows = [low for low, _ in merged]
        spans = sorted(by_work.get(work_id, []))
        span_starts = [start for start, _, _, _ in spans]
        tokens, starts = entries.get(work_id, ([], []))
        held: set[int] = set()
        for entry, start in zip(tokens, starts, strict=True):
            if not job.targets.intersection(entry.split()):
                continue
            inside = bisect_right(lows, start) - 1
            if inside >= 0 and start < merged[inside][1]:
                continue
            index = bisect_right(span_starts, start) - 1
            if index >= 0 and start < spans[index][1] and spans[index][3] is None:
                held.add(spans[index][2])
        targets_found.extend(
            (draw_key(BOUNDARY_SEED, work_id, number), decade, work_id, number) for number in held
        )
    return {
        "abstracts": smallest_per_decade(abstracts_found, KEEP_PER_PART),
        "targets": smallest_per_decade(targets_found, KEEP_PER_PART),
    }


def smallest_per_decade(
    found: list[tuple[str, int, str, int]], count: int
) -> list[tuple[str, int, str, int]]:
    """The ``count`` smallest keys of each decade."""
    kept: list[tuple[str, int, str, int]] = []
    for decade in sorted({row[1] for row in found}):
        kept.extend(sorted(row for row in found if row[1] == decade)[:count])
    return kept


def draw(
    found: list[tuple[str, int, str, int]], per_decade: int, extra: int
) -> list[tuple[str, int, str, int]]:
    """``per_decade`` smallest keys of each decade, then the ``extra`` smallest among the rest."""
    chosen = smallest_per_decade(found, per_decade)
    rest = sorted(set(found) - set(chosen))
    return sorted(chosen + rest[:extra])


def main() -> None:
    """Draw the samples, rebuild the calibration items, and write the items and the key."""
    run3, out = Path(sys.argv[1]), Path(sys.argv[2])
    processes = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    out.mkdir(parents=True, exist_ok=True)
    catalog = read_catalog()
    targets = frozenset(word.lemma for word in load_words(catalog) if not word.is_phrase)
    jobs = {part: PartJob(catalog, targets) for part in abstracts_parts(catalog)}
    with LiveProgress(Console()) as live:
        live.stage("finding candidates", total=len(jobs))
        results = map_parts(find_candidates, jobs, process_count=processes, progress=live)
    precision = draw([row for r in results.values() for row in r["abstracts"]], *PRECISION)
    boundaries = draw([row for r in results.values() for row in r["targets"]], *BOUNDARIES)
    wanted = {row[2] for row in precision + boundaries}
    current = (
        pl.scan_parquet(sorted(catalog.abstracts.path.glob("*.parquet")))
        .filter(pl.col("work_id").is_in(sorted(wanted)))
        .select("work_id", "text", "sentences")
        .collect()
    )
    texts = {work_id: (text, sentences) for work_id, text, sentences in current.iter_rows()}
    calibration = [
        json.loads(line)
        for line in RUN3_VERDICTS.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["part"] == "sentence_breaks"
    ]
    old = (
        pl.scan_parquet(sorted(run3.glob("*.parquet")))
        .filter(pl.col("work_id").is_in([row["work_id"] for row in calibration]))
        .select("work_id", "text", "sentences")
        .collect()
    )
    old_texts = {work_id: (text, sentences) for work_id, text, sentences in old.iter_rows()}

    def shown(source: dict[str, tuple[str, list[dict[str, Any]]]], work_id: str) -> list[str]:
        text, sentences = source[work_id]
        return [text[sentence["start"] : sentence["end"]] for sentence in sentences]

    whole: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in calibration:
        sentences = shown(old_texts, row["work_id"])
        if len(sentences) - 1 != row["cuts"]:
            raise SystemExit(f"{row['id']}: run 3's table no longer matches the review")
        whole.append(({"sentences": sentences}, {"set": "calibration", "review_id": row["id"]}))
    for _, decade, work_id, _ in precision:
        source = {"set": "precision", "work_id": work_id, "decade": decade}
        whole.append(({"sentences": shown(texts, work_id)}, source))
    random.Random(SHUFFLE_SEED).shuffle(whole)
    key: dict[str, Any] = {}
    with (out / "abstracts.jsonl").open("w", encoding="utf-8") as handle:
        for number, (item, source) in enumerate(whole):
            item_id = f"A{number:03d}"
            key[item_id] = source | {"cuts": len(item["sentences"]) - 1}
            handle.write(json.dumps({"id": item_id} | item, ensure_ascii=False) + "\n")
    with (out / "sentences.jsonl").open("w", encoding="utf-8") as handle:
        for number, (_, decade, work_id, sentence_number) in enumerate(boundaries):
            item_id = f"T{number:03d}"
            sentences = shown(texts, work_id)
            key[item_id] = {
                "set": "boundaries",
                "work_id": work_id,
                "decade": decade,
                "sentence_number": sentence_number,
            }
            item = {
                "id": item_id,
                "before": sentences[sentence_number - 1] if sentence_number > 0 else None,
                "target": sentences[sentence_number],
                "after": sentences[sentence_number + 1]
                if sentence_number + 1 < len(sentences)
                else None,
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (out / "key.json").write_text(json.dumps(key, indent=1) + "\n")
    cuts = sum(entry["cuts"] for entry in key.values() if entry.get("set") == "precision")
    sys.stdout.write(
        f"{len(whole)} whole abstracts ({len(calibration)} calibration, {len(precision)} for "
        f"precision, {cuts} cuts to judge); {len(boundaries)} target sentences\n"
    )


if __name__ == "__main__":
    main()
