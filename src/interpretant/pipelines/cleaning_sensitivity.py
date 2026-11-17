"""Round S1 of ``docs/predictions_cleaning.md``: does the identifiable junk move static drift?

The base P is an earlier build's kept works; the treatment C leaves out of it the works a later
build drops, the works with no finite verb outside P's removals, and the ranges the later build
removes that P keeps. Placebos match C's removal in each cell with works and spans C leaves
alone; bootstrap samples draw each cell's works with replacement. Every version gets one PPMI-SVD
model per cell, and each within-field comparison of consecutive decades is measured against P's
(``drift/stability.py``). One worker takes one field, decade by decade, keeping only the previous
decade's models. No target is named in the movements; the targets enter only as a combined rank
change and as corpus counts.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import polars as pl
from gensim.models import KeyedVectors

from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.static_view import load_static_view, static_view_parts
from interpretant.corpus.cell_versions import (
    Cut,
    bootstrap_sample,
    cell_lines,
    kept_segments,
    placebo_cut,
    seeded,
    token_count,
)
from interpretant.drift.grid import add_shift_floor, compare_cells
from interpretant.drift.stability import movement
from interpretant.embedding.ppmi import train_lines
from interpretant.pipelines.progress import SILENT, Progress
from interpretant.pipelines.workers import map_parts

BASE_SEED = 20261116
# Bootstrap replicates are seeded apart from the placebos'.
_BOOTSTRAP_OFFSET = 1000
_THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
Range = tuple[int, int]
VersionKey = tuple[str, int]
BASE: VersionKey = ("base", 0)
TREATMENT: VersionKey = ("treatment", 0)

MOVEMENT_SCHEMA = pl.Schema(
    {
        "version": pl.String(),
        "replicate": pl.Int64(),
        "field": pl.String(),
        "decade_a": pl.Int64(),
        "decade_b": pl.Int64(),
        "base_measured": pl.Int64(),
        "flag_turnover": pl.Float64(),
        "rank_disagreement": pl.Float64(),
        "entering": pl.Int64(),
        "leaving": pl.Int64(),
        "target_rank_change": pl.Float64(),
        "targets_compared": pl.Int64(),
    }
)
DOSE_SCHEMA = pl.Schema(
    {
        "field": pl.String(),
        "decade": pl.Int64(),
        "works": pl.Int64(),
        "tokens": pl.Int64(),
        "dropped_works": pl.Int64(),
        "verbless_works": pl.Int64(),
        "removed_tokens": pl.Int64(),
    }
)


@dataclass(frozen=True)
class SensitivitySettings:
    """How many placebos and bootstrap samples, and which fields (all when empty)."""

    placebos: int = 19
    bootstraps: int = 9
    fields: tuple[str, ...] = ()
    # Round S1b: neighbour shift averaged over the bootstrap models of P, of C and of P again,
    # instead of single models and placebos.
    averaged: bool = False


@dataclass
class CellPlan:
    """A cell's works in P, their removals in P, and what the later build changes."""

    work_ids: list[str] = field(default_factory=list)
    base_removals: dict[str, list[Range]] = field(default_factory=dict)
    later_dropped: set[str] = field(default_factory=set)
    later_removals: dict[str, list[Range]] = field(default_factory=dict)
    # The static view's parts that hold the cell's works: a work in several fields is annotated
    # once, in one field's part of its decade.
    view_parts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _FieldJob:
    catalog: CatalogConfig
    cells: dict[int, CellPlan]
    targets: frozenset[str]
    settings: SensitivitySettings


@dataclass(frozen=True)
class _FieldResult:
    movements: list[tuple[object, ...]]
    doses: list[tuple[object, ...]]
    target_tokens: Counter[str] = field(default_factory=Counter)
    target_removed: Counter[str] = field(default_factory=Counter)
    models: int = 0


@dataclass(frozen=True)
class SensitivityResult:
    """What the round wrote: its tables, and how many models it trained."""

    movements: pl.DataFrame
    doses: pl.DataFrame
    target_doses: pl.DataFrame
    models: int


def measure_cleaning_sensitivity(
    catalog: CatalogConfig,
    base: pl.DataFrame,
    later: pl.DataFrame,
    *,
    targets: frozenset[str],
    settings: SensitivitySettings,
    process_count: int = 1,
    progress: Progress = SILENT,
) -> SensitivityResult:
    """Measure every version's movement from P for each within-field comparison.

    ``base`` is P's abstracts table and ``later`` the abstracts table of the build whose drops and
    removals C applies (``catalog.revisions.load_abstracts_at`` reads a recorded build).

    Raises:
        ValueError: if a work's text differs between the two builds, or a cell's work is missing
            from its part of the static view.
    """
    plans = plan_cells(base, later, settings.fields)
    _locate_works(catalog, plans)
    jobs = {
        name: _FieldJob(catalog=catalog, cells=cells, targets=targets, settings=settings)
        for name, cells in plans.items()
    }
    sizes = {
        name: sum(len(plan.work_ids) for plan in cells.values()) for name, cells in plans.items()
    }
    progress.stage("measuring the versions", total=sum(sizes.values()))
    with _one_blas_thread_per_worker(process_count):
        results = map_parts(
            _measure_field, jobs, process_count=process_count, progress=progress, sizes=sizes
        )
    target_tokens: Counter[str] = Counter()
    target_removed: Counter[str] = Counter()
    for result in results.values():
        target_tokens.update(result.target_tokens)
        target_removed.update(result.target_removed)
    return SensitivityResult(
        movements=pl.DataFrame(
            [row for result in results.values() for row in result.movements],
            schema=MOVEMENT_SCHEMA,
            orient="row",
        ),
        doses=pl.DataFrame(
            [row for result in results.values() for row in result.doses],
            schema=DOSE_SCHEMA,
            orient="row",
        ),
        target_doses=pl.DataFrame(
            {
                "target": sorted(targets),
                "tokens": [target_tokens[word] for word in sorted(targets)],
                "removed_tokens": [target_removed[word] for word in sorted(targets)],
            },
            schema={"target": pl.String(), "tokens": pl.Int64(), "removed_tokens": pl.Int64()},
        ),
        models=sum(result.models for result in results.values()),
    )


def plan_cells(
    base: pl.DataFrame, later: pl.DataFrame, fields: tuple[str, ...]
) -> dict[str, dict[int, CellPlan]]:
    """Each field's cells: P's kept regular works, their removals, and the later build's changes.

    Raises:
        ValueError: if a work's text differs between the two builds.
    """
    kept = base.filter(pl.col("dropped_reason").is_null() & (pl.col("source") == "regular"))
    joined = kept.join(
        later.select(
            "work_id",
            pl.col("text").alias("later_text"),
            pl.col("removals").alias("later_removals"),
            pl.col("dropped_reason").alias("later_dropped"),
        ),
        on="work_id",
        how="left",
    )
    changed = joined.filter(
        pl.col("later_text").is_not_null() & (pl.col("text") != pl.col("later_text"))
    )
    if changed.height:
        raise ValueError(f"{changed.height} works have another text in the later build")
    plans: dict[str, dict[int, CellPlan]] = {}
    for row in joined.iter_rows(named=True):
        base_ranges = [(removal["start"], removal["end"]) for removal in row["removals"]]
        dropped = row["later_text"] is None or row["later_dropped"] is not None
        added = [] if dropped else _added_ranges(base_ranges, row["later_removals"])
        for name in row["fields"]:
            if fields and name not in fields:
                continue
            cell = plans.setdefault(name, {}).setdefault(row["decade"], CellPlan())
            cell.work_ids.append(row["work_id"])
            cell.base_removals[row["work_id"]] = base_ranges
            if dropped:
                cell.later_dropped.add(row["work_id"])
            elif added:
                cell.later_removals[row["work_id"]] = base_ranges + added
    return {name: dict(sorted(cells.items())) for name, cells in sorted(plans.items())}


def _locate_works(catalog: CatalogConfig, plans: dict[str, dict[int, CellPlan]]) -> None:
    """Give each cell the static view's parts that hold its works.

    Raises:
        ValueError: if a cell's work is in no part of the static view.
    """
    part_of: dict[str, str] = {}
    for part in static_view_parts(catalog):
        for work_id in load_static_view(catalog, part, columns=["work_id"])["work_id"]:
            part_of.setdefault(work_id, part)
    for name, cells in plans.items():
        for decade, plan in cells.items():
            missing = [work_id for work_id in plan.work_ids if work_id not in part_of]
            if missing:
                raise ValueError(
                    f"{len(missing)} works of cell {name}_{decade} have no static view"
                )
            plan.view_parts = sorted({part_of[work_id] for work_id in plan.work_ids})


def has_finite_verb_outside(
    starts: list[int], finite_verbs: list[bool], removals: list[Range]
) -> bool:
    """Whether a finite verb lies outside every removed range."""
    return any(
        finite and not any(start <= offset < end for start, end in removals)
        for offset, finite in zip(starts, finite_verbs, strict=True)
    )


def _added_ranges(base_ranges: list[Range], later: list[dict[str, Any]]) -> list[Range]:
    """The later build's removed ranges that P does not remove."""
    known = set(base_ranges)
    ranges = [(int(removal["start"]), int(removal["end"])) for removal in later]
    return [span for span in ranges if span not in known]


def _measure_field(name: str, job: _FieldJob) -> _FieldResult:
    """Train every version of the field's cells, decade by decade, and measure each comparison."""
    movements: list[tuple[object, ...]] = []
    doses: list[tuple[object, ...]] = []
    target_tokens: Counter[str] = Counter()
    target_removed: Counter[str] = Counter()
    previous: dict[VersionKey, KeyedVectors] = {}
    previous_decade = 0
    models = 0
    for decade, plan in job.cells.items():
        cell = f"{name}_{decade}"
        base_segments, dropped, passages, dose = _treatment(job, cell, plan)
        doses.append((name, decade, *dose))
        _count_targets(base_segments, dropped, passages, job.targets, target_tokens, target_removed)
        current: dict[VersionKey, KeyedVectors] = {}
        versions = _averaged_versions if job.settings.averaged else _versions
        for key, lines in versions(base_segments, dropped, passages, cell, job.settings):
            current[key] = train_lines(lines)
            models += 1
        if previous:
            compare = _compare_averaged if job.settings.averaged else _compare
            movements.extend(compare(name, previous_decade, decade, previous, current, job.targets))
        previous, previous_decade = current, decade
    return _FieldResult(movements, doses, target_tokens, target_removed, models)


def _treatment(
    job: _FieldJob, cell: str, plan: CellPlan
) -> tuple[dict[str, list[list[str]]], frozenset[str], dict[str, list[list[str]]], tuple[int, ...]]:
    """P's segments of the cell, C's dropped works, C's cut works' segments, and the cell's dose."""
    wanted = set(plan.work_ids)
    view: dict[str, tuple[list[str], list[int], list[bool]]] = {}
    for part in plan.view_parts:
        rows = load_static_view(job.catalog, part).filter(pl.col("work_id").is_in(wanted))
        for work_id, tokens, starts, finite in rows.iter_rows():
            view.setdefault(work_id, (tokens, starts, finite))
    base_segments = {
        work_id: kept_segments(view[work_id][0], view[work_id][1], plan.base_removals[work_id])
        for work_id in plan.work_ids
    }
    verbless = frozenset(
        work_id
        for work_id in plan.work_ids
        if not has_finite_verb_outside(
            view[work_id][1], view[work_id][2], plan.base_removals[work_id]
        )
    )
    dropped = frozenset(plan.later_dropped) | verbless
    passages = {
        work_id: kept_segments(view[work_id][0], view[work_id][1], ranges)
        for work_id, ranges in plan.later_removals.items()
        if work_id not in dropped
    }
    removed = sum(token_count(base_segments[work_id]) for work_id in dropped) + sum(
        token_count(base_segments[work_id]) - token_count(segments)
        for work_id, segments in passages.items()
    )
    total = sum(token_count(segments) for segments in base_segments.values())
    return (
        base_segments,
        dropped,
        passages,
        (len(plan.work_ids), total, len(dropped), len(verbless), removed),
    )


def _versions(
    base_segments: dict[str, list[list[str]]],
    dropped: frozenset[str],
    passages: dict[str, list[list[str]]],
    cell: str,
    settings: SensitivitySettings,
) -> Iterator[tuple[VersionKey, list[str]]]:
    """Each version's lines, one at a time, so only one version's text is held."""
    yield BASE, cell_lines(base_segments)
    yield TREATMENT, cell_lines({**base_segments, **passages}, Cut(dropped=dropped))
    sizes = {
        work_id: token_count(base_segments[work_id]) - token_count(segments)
        for work_id, segments in passages.items()
        if token_count(base_segments[work_id]) > token_count(segments)
    }
    for replicate in range(settings.placebos):
        cut = placebo_cut(base_segments, dropped, sizes, seeded(BASE_SEED, replicate, cell))
        yield ("placebo", replicate), cell_lines(base_segments, cut)
    work_ids = list(base_segments)
    for replicate in range(settings.bootstraps):
        rng = seeded(BASE_SEED, _BOOTSTRAP_OFFSET + replicate, cell)
        yield (
            ("bootstrap", replicate),
            cell_lines(base_segments, sample=bootstrap_sample(work_ids, rng)),
        )


# Round S1b's three sets of bootstrap models, and the seed offset of each: averaged P reuses round
# S1's bootstrap samples of P.
AVERAGED_SETS = {"averaged_base": 1000, "averaged_noise": 2000, "averaged_treatment": 3000}


def _averaged_versions(
    base_segments: dict[str, list[list[str]]],
    dropped: frozenset[str],
    passages: dict[str, list[list[str]]],
    cell: str,
    settings: SensitivitySettings,
) -> Iterator[tuple[VersionKey, list[str]]]:
    """Each bootstrap sample of P, of P again and of C, one at a time."""
    treatment = {
        work_id: segments
        for work_id, segments in {**base_segments, **passages}.items()
        if work_id not in dropped
    }
    pools = {
        "averaged_base": base_segments,
        "averaged_noise": base_segments,
        "averaged_treatment": treatment,
    }
    for replicate in range(settings.bootstraps):
        for name, offset in AVERAGED_SETS.items():
            works = pools[name]
            sample = bootstrap_sample(list(works), seeded(BASE_SEED, offset + replicate, cell))
            yield (name, replicate), cell_lines(works, sample=sample)


def _compare_averaged(
    name: str,
    decade_a: int,
    decade_b: int,
    previous: dict[VersionKey, KeyedVectors],
    current: dict[VersionKey, KeyedVectors],
    targets: frozenset[str],
) -> Iterator[tuple[object, ...]]:
    """Averaged C's and averaged P′'s movement from averaged P, for one comparison."""
    replicates = sorted(
        {replicate for set_name, replicate in current if set_name == "averaged_base"}
    )
    averaged = {
        set_name: add_shift_floor(
            average_pairs(
                [
                    compare_cells(previous[(set_name, replicate)], current[(set_name, replicate)])
                    for replicate in replicates
                ]
            )
        )
        for set_name in AVERAGED_SETS
    }
    base = averaged["averaged_base"]
    for set_name in ("averaged_treatment", "averaged_noise"):
        version = averaged[set_name]
        moved = movement(base, version)
        change, compared = _target_rank_change(base, version, targets)
        yield (
            set_name,
            0,
            name,
            decade_a,
            decade_b,
            base.height,
            moved.flag_turnover,
            moved.rank_disagreement,
            moved.entering,
            moved.leaving,
            change,
            compared,
        )


def average_pairs(pairs: list[pl.DataFrame]) -> pl.DataFrame:
    """The words measured in every replicate's pair table, with counts and drift averaged."""
    if not pairs or any(pair.is_empty() for pair in pairs):
        return pl.DataFrame(schema={**pairs[0].schema} if pairs else None)
    stacked = pl.concat(pairs)
    return (
        stacked.group_by("word")
        .agg(
            pl.len().alias("replicates"),
            pl.col("count_a").mean(),
            pl.col("count_b").mean(),
            pl.col("cosine_drift").mean(),
            pl.col("neighbour_shift").mean(),
        )
        .filter(pl.col("replicates") == len(pairs))
        .drop("replicates")
        .sort("word")
    )


def _compare(
    name: str,
    decade_a: int,
    decade_b: int,
    previous: dict[VersionKey, KeyedVectors],
    current: dict[VersionKey, KeyedVectors],
    targets: frozenset[str],
) -> Iterator[tuple[object, ...]]:
    base = add_shift_floor(compare_cells(previous[BASE], current[BASE]))
    for key in current:
        if key == BASE:
            continue
        version = add_shift_floor(compare_cells(previous[key], current[key]))
        moved = movement(base, version)
        change, compared = _target_rank_change(base, version, targets)
        yield (
            key[0],
            key[1],
            name,
            decade_a,
            decade_b,
            base.height,
            moved.flag_turnover,
            moved.rank_disagreement,
            moved.entering,
            moved.leaving,
            change,
            compared,
        )


def _target_rank_change(
    base: pl.DataFrame, version: pl.DataFrame, targets: frozenset[str]
) -> tuple[float, int]:
    """The targets' summed absolute change in within-bin rank, and how many were compared."""
    joined = (
        base.filter(pl.col("word").is_in(sorted(targets)))
        .select("word", "shift_rank")
        .join(version.select("word", "shift_rank"), on="word", suffix="_version")
    )
    change = (joined["shift_rank"] - joined["shift_rank_version"]).abs().sum()
    return float(change or 0.0), joined.height


def _count_targets(
    base_segments: dict[str, list[list[str]]],
    dropped: frozenset[str],
    passages: dict[str, list[list[str]]],
    targets: frozenset[str],
    tokens: Counter[str],
    removed: Counter[str],
) -> None:
    """Add the cell's target tokens in P, and those C removes, to the running counts."""
    for work_id, segments in base_segments.items():
        before = Counter(word for segment in segments for word in segment if word in targets)
        tokens.update(before)
        if work_id in dropped:
            removed.update(before)
        elif work_id in passages:
            after = Counter(
                word for segment in passages[work_id] for word in segment if word in targets
            )
            removed.update(before - after)


@contextmanager
def _one_blas_thread_per_worker(process_count: int) -> Iterator[None]:
    """Start worker processes with BLAS limited to one thread; the main process is unchanged."""
    if process_count == 1:
        yield
        return
    previous = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
    os.environ.update(dict.fromkeys(_THREAD_VARIABLES, "1"))
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                del os.environ[name]
            else:
                os.environ[name] = value
