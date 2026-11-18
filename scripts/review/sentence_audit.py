"""The fresh sentence view audit's items, after the merge rules (`docs/sentence_view.md`).

- **Whole abstracts:** 30 for the agents' recalibration (seed 20261201: 3 per decade, then the 6
  smallest keys among the rest) and 50 for cut precision (seed 20261202: 6 per decade, then 2), kept
  works with at least one cut in the view, the two sets apart. They are shuffled together (seed
  20261205), so the agents cannot tell the recalibration from the sample; the author judges the 30.
- **Joined cuts:** the 80 joins with the smallest sha256 of (seed 20261203, work id, sentence
  number, join), from every rule but the no-letters one. Each shows the two pieces and the joined
  sentence.
- **Target sentences:** 150 sentences with no exclusion that hold a single-word target outside
  removed ranges, by the static view (seed 20261204: 18 per decade, then 6), each with the
  sentence before and after as the view cuts them.

Round 3 drops the recalibration (it carries over) and adds 80 split cuts (each row the split rule
starts, with the row before it); its seeds are in `ROUNDS`. The first round's version of this
script is `099301b6`. The items quote corpus text, so they stay out of git; key.json maps each
blind id to its source. Run from the repository root on the run machine:
`python scripts/review/sentence_audit.py ROUND OUT [PROCESSES]`.
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

# Each round's seeds (`docs/sentence_view.md`): the fresh audit (2) and round 3, which drops the
# recalibration (it carries over) and adds the split cuts.
ROUNDS: dict[str, dict[str, int]] = {
    "2": {
        "recalibration": 20261201,
        "precision": 20261202,
        "joins": 20261203,
        "boundaries": 20261204,
        "shuffle": 20261205,
    },
    "3": {
        "precision": 20261212,
        "joins": 20261213,
        "boundaries": 20261214,
        "splits": 20261215,
        "shuffle": 20261216,
    },
}
RECALIBRATION = (3, 6)
PRECISION = (6, 2)
JOINS = 80
SPLITS = 80
BOUNDARIES = (18, 6)
KEEP_PER_PART = 40
CONTEXT = 600
# A candidate: draw key, decade, work id, sentence number, join index (-1 when not a join).
Candidate = tuple[str, int, str, int, int]


@dataclass(frozen=True)
class PartJob:
    """What a worker needs for one part (picklable)."""

    catalog: CatalogConfig
    targets: frozenset[str]
    seeds: dict[str, int]


def draw_key(seed: int, *parts: object) -> str:
    """A draw key: the sha256 of the seed and the parts."""
    return hashlib.sha256(":".join(str(part) for part in (seed, *parts)).encode()).hexdigest()


def find_candidates(part: str, job: PartJob) -> dict[str, list[Candidate]]:
    """The part's smallest-keyed candidates of each kind."""
    abstracts = load_abstracts_part(
        job.catalog, part, ["work_id", "decade", "removals", "dropped_reason"]
    ).filter(pl.col("dropped_reason").is_null())
    decades = dict(zip(abstracts["work_id"], abstracts["decade"], strict=True))
    removals = dict(zip(abstracts["work_id"], abstracts["removals"], strict=True))
    view = pl.read_parquet(part_path(job.catalog.sentence_view.path, part, ".parquet"))
    static = load_static_view(job.catalog, part, columns=["work_id", "tokens", "starts"])
    entries = {work_id: (tokens, starts) for work_id, tokens, starts in static.iter_rows()}
    found: dict[str, list[Candidate]] = {
        "recalibration": [],
        "precision": [],
        "joins": [],
        "splits": [],
        "targets": [],
    }
    seeds = job.seeds
    for (work_id,), rows in view.group_by("work_id", maintain_order=True):
        work = str(work_id)
        decade = decades[work]
        if rows.height > 1:
            for name in ("recalibration", "precision"):
                if name in seeds:
                    found[name].append((draw_key(seeds[name], work), decade, work, -1, -1))
        for number, rules, split in rows.select(
            "sentence_number", "merge_rules", "split"
        ).iter_rows():
            for index, rule in enumerate(rules):
                if rule != "no_letters":
                    key = draw_key(seeds["joins"], work, number, index)
                    found["joins"].append((key, decade, work, number, index))
            if split and "splits" in seeds:
                key = draw_key(seeds["splits"], work, number)
                found["splits"].append((key, decade, work, number, -1))
        found["targets"].extend(target_sentences(work, decade, rows, removals[work], entries, job))
    return {
        "recalibration": smallest_per_decade(found["recalibration"], KEEP_PER_PART),
        "precision": smallest_per_decade(found["precision"], KEEP_PER_PART),
        "joins": sorted(found["joins"])[:JOINS],
        "splits": sorted(found["splits"])[:SPLITS],
        "targets": smallest_per_decade(found["targets"], KEEP_PER_PART),
    }


def target_sentences(
    work: str,
    decade: int,
    rows: pl.DataFrame,
    removals: list[dict[str, Any]],
    entries: dict[str, tuple[list[str], list[int]]],
    job: PartJob,
) -> list[Candidate]:
    """The work's sentences with no exclusion that hold a target outside removed ranges."""
    merged = merge_ranges((removal["start"], removal["end"]) for removal in removals)
    lows = [low for low, _ in merged]
    spans = rows.select("start", "end", "sentence_number", "exclusion").rows()
    span_starts = [start for start, _, _, _ in spans]
    held: set[int] = set()
    tokens, starts = entries.get(work, ([], []))
    for entry, start in zip(tokens, starts, strict=True):
        if not job.targets.intersection(entry.split()):
            continue
        inside = bisect_right(lows, start) - 1
        if inside >= 0 and start < merged[inside][1]:
            continue
        index = bisect_right(span_starts, start) - 1
        if index >= 0 and start < spans[index][1] and spans[index][3] is None:
            held.add(spans[index][2])
    seed = job.seeds["boundaries"]
    return [(draw_key(seed, work, number), decade, work, number, -1) for number in held]


def smallest_per_decade(found: list[Candidate], count: int) -> list[Candidate]:
    """The ``count`` smallest keys of each decade."""
    kept: list[Candidate] = []
    for decade in sorted({row[1] for row in found}):
        kept.extend(sorted(row for row in found if row[1] == decade)[:count])
    return kept


def draw(found: list[Candidate], per_decade: int, extra: int) -> list[Candidate]:
    """``per_decade`` smallest keys of each decade, then the ``extra`` smallest among the rest."""
    chosen = smallest_per_decade(found, per_decade)
    rest = sorted(set(found) - set(chosen))
    return sorted(chosen + rest[:extra])


def load_sources(
    catalog: CatalogConfig, wanted: list[str]
) -> tuple[dict[str, str], dict[str, list[dict[str, Any]]], dict[str, list[tuple[Any, ...]]]]:
    """The wanted works' texts, spaCy sentences and view rows (start, end, merge rules)."""
    abstracts = (
        pl.scan_parquet(sorted(catalog.abstracts.path.glob("*.parquet")))
        .filter(pl.col("work_id").is_in(wanted))
        .select("work_id", "text", "sentences")
        .collect()
    )
    texts = dict(zip(abstracts["work_id"], abstracts["text"], strict=True))
    sentences = dict(zip(abstracts["work_id"], abstracts["sentences"].to_list(), strict=True))
    view = (
        pl.scan_parquet(sorted(catalog.sentence_view.path.glob("*.parquet")))
        .filter(pl.col("work_id").is_in(wanted))
        .sort("work_id", "sentence_number")
        .collect()
    )
    rows = {
        str(work): list(group.select("start", "end", "merge_rules").iter_rows())
        for (work,), group in view.group_by("work_id", maintain_order=True)
    }
    return texts, sentences, rows


def main() -> None:
    """Draw the samples and write the items and the key."""
    seeds, out = ROUNDS[sys.argv[1]], Path(sys.argv[2])
    processes = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    out.mkdir(parents=True, exist_ok=True)
    catalog = read_catalog()
    targets = frozenset(word.lemma for word in load_words(catalog) if not word.is_phrase)
    jobs = {part: PartJob(catalog, targets, seeds) for part in abstracts_parts(catalog)}
    with LiveProgress(Console()) as live:
        live.stage("finding candidates", total=len(jobs))
        results = map_parts(find_candidates, jobs, process_count=processes, progress=live)

    def pooled(name: str) -> list[Candidate]:
        return [row for result in results.values() for row in result[name]]

    recalibration = draw(pooled("recalibration"), *RECALIBRATION)
    taken = {row[2] for row in recalibration}
    precision = draw([row for row in pooled("precision") if row[2] not in taken], *PRECISION)
    joins = sorted(pooled("joins"))[:JOINS]
    splits = sorted(pooled("splits"))[:SPLITS]
    boundaries = draw(pooled("targets"), *BOUNDARIES)
    drawn = recalibration + precision + joins + splits + boundaries
    wanted = sorted({row[2] for row in drawn})
    texts, spacy_sentences, rows = load_sources(catalog, wanted)

    def shown(work: str) -> list[str]:
        return [texts[work][start:end] for start, end, _ in rows[work]]

    key: dict[str, Any] = {}
    whole = [(row, "recalibration") for row in recalibration]
    whole += [(row, "precision") for row in precision]
    random.Random(seeds["shuffle"]).shuffle(whole)
    with (out / "abstracts.jsonl").open("w", encoding="utf-8") as handle:
        for number, ((_, decade, work, _, _), name) in enumerate(whole):
            item_id = f"B{number:03d}"
            sentences = shown(work)
            key[item_id] = {
                "set": name,
                "work_id": work,
                "decade": decade,
                "cuts": len(sentences) - 1,
            }
            item: dict[str, Any] = {"id": item_id, "sentences": sentences}
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (out / "joins.jsonl").open("w", encoding="utf-8") as handle:
        for number, (_, decade, work, sentence_number, index) in enumerate(joins):
            item_id = f"J{number:03d}"
            start, end, rules = rows[work][sentence_number]
            parts = [s for s in spacy_sentences[work] if start <= s["start"] and s["end"] <= end]
            first, second = parts[index], parts[index + 1]
            low = max(start, (first["end"] + second["start"]) // 2 - CONTEXT // 2)
            item = {
                "id": item_id,
                "first": texts[work][first["start"] : first["end"]],
                "second": texts[work][second["start"] : second["end"]],
                "context": texts[work][low : min(end, low + CONTEXT)],
            }
            key[item_id] = {
                "set": "joins",
                "work_id": work,
                "decade": decade,
                "sentence_number": sentence_number,
                "join": index,
                "rule": rules[index],
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (out / "splits.jsonl").open("w", encoding="utf-8") as handle:
        for number, (_, decade, work, sentence_number, _) in enumerate(splits):
            item_id = f"S{number:03d}"
            (_, end_before, _), (start, end, _) = rows[work][
                sentence_number - 1 : sentence_number + 1
            ]
            low = max(0, (end_before + start) // 2 - CONTEXT // 2)
            item = {
                "id": item_id,
                "first": texts[work][rows[work][sentence_number - 1][0] : end_before],
                "second": texts[work][start:end],
                "context": texts[work][low : low + CONTEXT],
            }
            key[item_id] = {
                "set": "splits",
                "work_id": work,
                "decade": decade,
                "sentence_number": sentence_number,
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (out / "sentences.jsonl").open("w", encoding="utf-8") as handle:
        for number, (_, decade, work, sentence_number, _) in enumerate(boundaries):
            item_id = f"T{number:03d}"
            sentences = shown(work)
            key[item_id] = {
                "set": "boundaries",
                "work_id": work,
                "decade": decade,
                "sentence_number": sentence_number,
            }
            following = sentence_number + 1 < len(sentences)
            item = {
                "id": item_id,
                "before": sentences[sentence_number - 1] if sentence_number > 0 else None,
                "target": sentences[sentence_number],
                "after": sentences[sentence_number + 1] if following else None,
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (out / "key.json").write_text(json.dumps(key, indent=1) + "\n")
    cuts = sum(entry["cuts"] for entry in key.values() if entry["set"] == "precision")
    sys.stdout.write(
        f"{len(whole)} whole abstracts ({len(recalibration)} recalibration, {len(precision)} "
        f"precision with {cuts} cuts); {len(joins)} joins; {len(splits)} splits; "
        f"{len(boundaries)} target sentences\n"
    )


if __name__ == "__main__":
    main()
