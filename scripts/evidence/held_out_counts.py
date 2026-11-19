"""Corpus counts per decade for draft held-out words (agenda 6.2), kept out of the repository.

Agenda 6.2 drafts the held-out words with evidence that includes corpus counts per decade, never
change scores. This script reads the candidate lemmas from a JSON file outside the repository (a
list of objects with a ``lemma`` key), counts each one's static-view tokens and the kept works they
occur in, per decade, and writes the counts to another file outside the repository, so no held-out
word reaches git. Worker processes take whole parts. Run from the repository root:
`python scripts/evidence/held_out_counts.py CANDIDATES.json OUT.json [PROCESSES]`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from rich.console import Console

from interpretant.catalog.abstracts import load_abstracts_part
from interpretant.catalog.config import CatalogConfig, read_catalog
from interpretant.catalog.static_view import load_static_view, static_view_parts
from interpretant.cli.progress import LiveProgress
from interpretant.pipelines.workers import map_parts


@dataclass(frozen=True)
class CountJob:
    """What a worker needs (picklable)."""

    catalog: CatalogConfig
    lemmas: frozenset[str]


def count_part(part: str, job: CountJob) -> list[dict[str, object]]:
    """Each candidate's tokens and works in one part's kept works."""
    kept = (
        load_abstracts_part(job.catalog, part, ["work_id", "dropped_reason"])
        .filter(pl.col("dropped_reason").is_null())
        .select("work_id")
    )
    tokens = (
        load_static_view(job.catalog, part, columns=["work_id", "tokens"])
        .join(kept, on="work_id")
        .explode("tokens")
        .filter(pl.col("tokens").is_in(list(job.lemmas)))
    )
    counted = tokens.group_by("tokens").agg(
        pl.len().alias("tokens_count"), pl.col("work_id").n_unique().alias("works")
    )
    decade = int(part.rsplit("_", 1)[1])
    return [
        {"lemma": lemma, "decade": decade, "tokens": count, "works": works}
        for lemma, count, works in counted.iter_rows()
    ]


def main() -> None:
    """Count every candidate per decade and write the counts outside the repository."""
    candidates, out = Path(sys.argv[1]), Path(sys.argv[2])
    processes = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    repository = Path.cwd().resolve()
    if repository in out.resolve().parents:
        raise SystemExit("write the counts outside the repository: held-out words stay out of git")
    lemmas = frozenset(entry["lemma"] for entry in json.loads(candidates.read_text()))
    catalog = read_catalog()
    parts = static_view_parts(catalog)
    with LiveProgress(Console()) as live:
        live.stage("counting the candidates", total=len(parts))
        results = map_parts(
            count_part,
            dict.fromkeys(parts, CountJob(catalog, lemmas)),
            process_count=processes,
            progress=live,
        )
    rows = [row for part_rows in results.values() for row in part_rows]
    totals = (
        pl.DataFrame(
            rows,
            schema={"lemma": pl.String, "decade": pl.Int64, "tokens": pl.Int64, "works": pl.Int64},
        )
        .group_by("lemma", "decade")
        .agg(pl.col("tokens").sum(), pl.col("works").sum())
        .sort("lemma", "decade")
    )
    counts: dict[str, dict[str, dict[str, int]]] = {lemma: {} for lemma in sorted(lemmas)}
    for lemma, decade, tokens, works in totals.iter_rows():
        counts[lemma][str(decade)] = {"tokens": tokens, "works": works}
    out.write_text(json.dumps(counts, indent=1) + "\n")
    sys.stdout.write(f"{len(lemmas)} candidates counted over {len(parts)} parts\n")


if __name__ == "__main__":
    main()
