"""Drawing the junk gate's review items (`docs/junk_gate.md`): the sample, run 3 and confirmation.

Each set is drawn into its own folder. The sample holds the labelled sample L and the enrichment set
E together: worker processes describe, part by part, each record behind a drawn usage (its removed
ranges, whether a finite verb lies outside them, the repository-only flag and its kept length), and
the draws themselves are ``review.junk_gate``'s. Run 3's items are its judged records with the text
they had then (the catalog's ``run3_records``), under run 3's ids. The confirmation samples are
drawn from the cells the gate's screen flags, reusing every label already made. Writes the items
in batches of 50 (``batch_NN.jsonl``: they quote the corpus and stay out of git), ``key.json`` and
``counts.json``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from interpretant.catalog.abstracts import load_abstracts_part
from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.junk_gate import (
    load_junk_cells,
    load_junk_labels,
    load_run3_records,
    load_run3_verdicts,
)
from interpretant.catalog.static_view import load_static_view
from interpretant.catalog.words import load_words
from interpretant.corpus.removals import Removal, kept_text
from interpretant.corpus.work_metadata import REPOSITORY_ONLY
from interpretant.pipelines.cleaning_sensitivity import has_finite_verb_outside
from interpretant.pipelines.draw_usages_audit import (
    load_texts,
    load_usages_with_decades,
    save_batches,
)
from interpretant.pipelines.progress import SILENT, Progress
from interpretant.pipelines.workers import map_parts
from interpretant.review.junk_gate import (
    CONFIRMATION_PER_CELL,
    CONFIRMATION_SEED,
    ENRICHMENT_PER_STRATUM,
    ENRICHMENT_SEED,
    LABELLED_PER_DECADE,
    LABELLED_SEED,
    RECORDS_SCHEMA,
    USAGE_COLUMNS,
    DrawSet,
    build_confirmation_items,
    build_items,
    build_run3_items,
    build_strata,
    count_sample,
    count_strata,
    draw_confirmation,
    draw_enrichment,
    draw_labelled,
    parse_removals,
)
from interpretant.review.usages_audit import Sample

# The columns of a sample row: its usage, its sentence, where it sits, and its set and stratum
_SAMPLE_COLUMNS = [
    "work_id",
    "word",
    "content_key",
    "sentence_start",
    "sentence_end",
    "part",
    "decade",
    "set",
    "stratum",
]


@dataclass(frozen=True)
class JunkGateDraw:
    """What a draw wrote: its items by set and its batches."""

    items_by_set: dict[str, int]
    batch_count: int

    @property
    def item_count(self) -> int:
        """All the items drawn."""
        return sum(self.items_by_set.values())


@dataclass(frozen=True)
class RecordsJob:
    """What a worker describing one part's records needs (picklable)."""

    catalog: CatalogConfig
    work_ids: tuple[str, ...]


def draw_junk_gate(
    catalog: CatalogConfig,
    out: Path,
    *,
    draw_set: DrawSet,
    process_count: int,
    progress: Progress = SILENT,
) -> JunkGateDraw:
    """Draw one of the gate's sets and write its items, key and counts to ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    draws: dict[DrawSet, Callable[[], JunkGateDraw]] = {
        "sample": lambda: draw_sample(catalog, out, process_count=process_count, progress=progress),
        "run3": lambda: draw_run3(catalog, out),
        "confirmation": lambda: draw_confirmation_samples(
            catalog, out, process_count=process_count, progress=progress
        ),
    }
    return draws[draw_set]()


def draw_sample(
    catalog: CatalogConfig, out: Path, *, process_count: int, progress: Progress = SILENT
) -> JunkGateDraw:
    """Draw L and E from the drawn usages and write them as one set of items."""
    words = {word.lemma: word for word in load_words(catalog)}
    usages = load_usages_with_decades(catalog, USAGE_COLUMNS)
    drawn = usages.filter("drawn")
    labelled = draw_labelled(drawn, seed=LABELLED_SEED, per_decade=LABELLED_PER_DECADE)
    records = describe_records(catalog, drawn, process_count=process_count, progress=progress)
    strata = build_strata(drawn, records)
    enrichment = draw_enrichment(
        drawn,
        strata,
        set(labelled["work_id"]),
        seed=ENRICHMENT_SEED,
        per_stratum=ENRICHMENT_PER_STRATUM,
    )
    samples = pl.concat(
        [
            labelled.with_columns(
                pl.lit("labelled").alias("set"), pl.lit(None, dtype=pl.String).alias("stratum")
            ).select(_SAMPLE_COLUMNS),
            enrichment.with_columns(pl.lit("enrichment").alias("set")).select(_SAMPLE_COLUMNS),
        ]
    ).to_dicts()
    items, key = build_items(
        samples,
        load_texts(catalog, _as_samples(samples)),
        _removals_by_work(records),
        words,
        prefix="S",
    )
    counts = count_sample(usages, words, count_strata(strata), key)
    return _write_draw(out, items, key, counts, counts["items"])


def draw_run3(catalog: CatalogConfig, out: Path) -> JunkGateDraw:
    """Write run 3's judged records as items, with the text they had in that run."""
    records = load_run3_records(catalog)
    items, key = build_run3_items(
        load_run3_verdicts(catalog),
        dict(zip(records["work_id"], records["text"], strict=True)),
        _removals_by_work(records),
    )
    return _write_draw(out, items, key, {"items": len(items)}, {"run3": len(items)})


def draw_confirmation_samples(
    catalog: CatalogConfig, out: Path, *, process_count: int, progress: Progress = SILENT
) -> JunkGateDraw:
    """Draw each flagged cell's confirmation sample; a usage already labelled keeps its label.

    Raises:
        FileNotFoundError: if the cells table or the labels are missing.
    """
    words = {word.lemma: word for word in load_words(catalog)}
    labels = load_junk_labels(catalog)
    drawn = load_usages_with_decades(catalog, USAGE_COLUMNS).filter("drawn")
    chosen = draw_confirmation(
        drawn, load_junk_cells(catalog), seed=CONFIRMATION_SEED, per_cell=CONFIRMATION_PER_CELL
    )
    rows = chosen.with_columns(pl.lit("confirmation").alias("set")).to_dicts()
    records = describe_records(catalog, chosen, process_count=process_count, progress=progress)
    items, key = build_confirmation_items(
        chosen, labels, load_texts(catalog, _as_samples(rows)), _removals_by_work(records), words
    )
    counts = {
        "items": len(items),
        "flagged_cells": len(key["cells"]),
        "usages": chosen.height,
        "already_labelled": chosen.height - len(items),
    }
    return _write_draw(out, items, key, counts, {"confirmation": len(items)})


def describe_records(
    catalog: CatalogConfig, usages: pl.DataFrame, *, process_count: int, progress: Progress
) -> pl.DataFrame:
    """Each record behind the usages, described part by part (``RECORDS_SCHEMA``)."""
    work_ids_by_part = {
        part: tuple(sorted(work_ids))
        for part, work_ids in usages.group_by("part").agg(pl.col("work_id").unique()).iter_rows()
    }
    progress.stage("describing the records", total=len(work_ids_by_part))
    described = map_parts(
        describe_records_in_part,
        {
            part: RecordsJob(catalog, work_ids)
            for part, work_ids in sorted(work_ids_by_part.items())
        },
        process_count=process_count,
        progress=progress,
    )
    return pl.concat([pl.DataFrame(schema=RECORDS_SCHEMA), *described.values()])


def describe_records_in_part(part: str, job: RecordsJob) -> pl.DataFrame:
    """One part's records: removed ranges, finite verb outside them, repository-only, kept length.

    Whether a finite verb lies outside the removed ranges is read from the static view's tokens,
    as round S1 reads it (``pipelines.cleaning_sensitivity``).

    Raises:
        ValueError: if a record is missing from the part's abstracts or its static view.
    """
    wanted = pl.DataFrame({"work_id": list(job.work_ids)}, schema={"work_id": pl.String()})
    abstracts = load_abstracts_part(
        job.catalog, part, ["work_id", "text", "removals", "flags"]
    ).join(wanted, on="work_id", how="semi")
    view = load_static_view(job.catalog, part, ["work_id", "starts", "finite_verbs"])
    joined = abstracts.join(view, on="work_id", how="left")
    undescribed = len(job.work_ids) - joined.filter(pl.col("starts").is_not_null()).height
    if undescribed:
        raise ValueError(f"{undescribed} records of part {part} lack their text or static view")
    rows = [
        {
            "work_id": work_id,
            "removals": removals,
            "has_finite_verb": has_finite_verb_outside(
                starts, finite_verbs, [(removal["start"], removal["end"]) for removal in removals]
            ),
            "is_repository_only": REPOSITORY_ONLY in flags,
            "kept_characters": len(kept_text(text, parse_removals(removals))),
        }
        for work_id, text, removals, flags, starts, finite_verbs in joined.iter_rows()
    ]
    return pl.DataFrame(rows, schema=RECORDS_SCHEMA)


def _as_samples(rows: list[dict[str, Any]]) -> list[Sample]:
    """The rows as ``load_texts`` takes them, each with its set."""
    return [(row["set"], row) for row in rows]


def _removals_by_work(records: pl.DataFrame) -> dict[str, list[Removal]]:
    """Each record's removed ranges, by work id."""
    return {
        work_id: parse_removals(removals)
        for work_id, removals in records.select("work_id", "removals").iter_rows()
    }


def _write_draw(
    out: Path,
    items: list[dict[str, str]],
    key: Mapping[str, Any],
    counts: Mapping[str, Any],
    items_by_set: Mapping[str, int],
) -> JunkGateDraw:
    """Write the items in batches, the key and the counts; what was written."""
    batch_count = save_batches(items, out)
    (out / "key.json").write_text(json.dumps(key, indent=1) + "\n")
    (out / "counts.json").write_text(json.dumps(counts, indent=1) + "\n")
    return JunkGateDraw(items_by_set=dict(items_by_set), batch_count=batch_count)
