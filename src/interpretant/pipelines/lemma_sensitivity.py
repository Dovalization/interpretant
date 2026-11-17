"""Round L1 of ``docs/predictions_lemmas.md``: do the lemma exceptions move static drift?

Two static views of the same annotation cache: A without the lemma exceptions (the control) and B
with them (the treatment). An exception changes a token's string, never whether it is kept, so the
two views hold the same works and offsets, and each cell's works under the build of record's
removals give one text in each view. Per cell, round S1b's three sets of bootstrap models
(``cleaning_sensitivity.averaged_versions``): averaged A, averaged A again (the noise) and
averaged B; each within-field comparison of consecutive decades is measured against averaged A.
One worker takes one field, decade by decade. No target is named in the movements; the targets
enter only as a combined rank change and as corpus counts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import polars as pl
from gensim.models import KeyedVectors

from interpretant.catalog.config import CatalogConfig
from interpretant.corpus.cell_versions import kept_segments, token_count
from interpretant.embedding.ppmi import SvdBackend, train_lines
from interpretant.pipelines.cleaning_sensitivity import (
    AVERAGED_SETS,
    MOVEMENT_SCHEMA,
    CellPlan,
    VersionKey,
    WorkView,
    averaged_versions,
    compare_averaged,
    load_cell_view,
    locate_works,
    one_blas_thread_per_worker,
    plan_cells,
)
from interpretant.pipelines.progress import SILENT, Progress
from interpretant.pipelines.workers import Report, map_parts_reporting

DOSE_SCHEMA = pl.Schema(
    {
        "field": pl.String(),
        "decade": pl.Int64(),
        "works": pl.Int64(),
        "tokens": pl.Int64(),
        "changed_tokens": pl.Int64(),
    }
)
TARGET_COUNT_SCHEMA = pl.Schema(
    {"target": pl.String(), "tokens_a": pl.Int64(), "tokens_b": pl.Int64()}
)
# Each work's lines under its removals.
Segments = dict[str, list[list[str]]]


@dataclass(frozen=True)
class _FieldJob:
    control: CatalogConfig
    treatment: CatalogConfig
    cells: dict[int, CellPlan]
    targets: frozenset[str]
    bootstraps: int
    svd_backend: SvdBackend


@dataclass(frozen=True)
class _FieldResult:
    movements: list[tuple[object, ...]]
    doses: list[tuple[object, ...]]
    tokens_a: Counter[str] = field(default_factory=Counter)
    tokens_b: Counter[str] = field(default_factory=Counter)
    models: int = 0


@dataclass(frozen=True)
class LemmaSensitivityResult:
    """What the round wrote: its tables, and how many models it trained."""

    movements: pl.DataFrame
    doses: pl.DataFrame
    target_counts: pl.DataFrame
    models: int


def measure_lemma_sensitivity(
    control: CatalogConfig,
    treatment: CatalogConfig,
    abstracts: pl.DataFrame,
    *,
    targets: frozenset[str],
    bootstraps: int = 9,
    fields: tuple[str, ...] = (),
    svd_backend: SvdBackend = "arpack",
    process_count: int = 1,
    progress: Progress = SILENT,
) -> LemmaSensitivityResult:
    """Measure averaged B's and averaged A′'s movement from averaged A for each comparison.

    ``control`` and ``treatment`` are catalogs that differ only in their static view's path;
    ``abstracts`` is the build of record's abstracts table, whose kept regular works and removals
    make the cells.

    Raises:
        ValueError: if the two catalogs name the same static view, a cell's work is missing from a
            view, or a work's offsets differ between the views.
    """
    if control.static_view.path.resolve() == treatment.static_view.path.resolve():
        raise ValueError("the control and the treatment name the same static view")
    plans = plan_cells(abstracts, abstracts, fields)
    locate_works(treatment, plans)
    jobs = {
        name: _FieldJob(
            control=control,
            treatment=treatment,
            cells=cells,
            targets=targets,
            bootstraps=bootstraps,
            svd_backend=svd_backend,
        )
        for name, cells in plans.items()
    }
    sizes = {
        name: sum(len(plan.work_ids) for plan in cells.values()) for name, cells in plans.items()
    }
    cell_count = sum(len(field_cells) for field_cells in plans.values())
    progress.stage("training the cell models", total=cell_count * len(AVERAGED_SETS) * bootstraps)
    with one_blas_thread_per_worker(process_count):
        results = map_parts_reporting(
            _measure_field, jobs, process_count=process_count, progress=progress, sizes=sizes
        )
    tokens_a: Counter[str] = Counter()
    tokens_b: Counter[str] = Counter()
    for result in results.values():
        tokens_a.update(result.tokens_a)
        tokens_b.update(result.tokens_b)
    return LemmaSensitivityResult(
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
        target_counts=pl.DataFrame(
            {
                "target": sorted(targets),
                "tokens_a": [tokens_a[word] for word in sorted(targets)],
                "tokens_b": [tokens_b[word] for word in sorted(targets)],
            },
            schema=TARGET_COUNT_SCHEMA,
        ),
        models=sum(result.models for result in results.values()),
    )


def cell_segments(
    view_a: dict[str, WorkView], view_b: dict[str, WorkView], plan: CellPlan
) -> tuple[Segments, Segments, tuple[int, int, int]]:
    """Each view's lines of the cell's works under their removals, and the cell's dose.

    The dose is the works, the kept tokens, and the kept tokens whose string differs.

    Raises:
        ValueError: if a work is missing from a view, or its offsets differ between the views.
    """
    segments_a: Segments = {}
    segments_b: Segments = {}
    changed = 0
    for work_id in plan.work_ids:
        if work_id not in view_a or work_id not in view_b:
            raise ValueError(f"work {work_id} is missing from a static view")
        tokens_a, starts_a, _ = view_a[work_id]
        tokens_b, starts_b, _ = view_b[work_id]
        if starts_a != starts_b:
            raise ValueError(f"work {work_id} has other offsets in the two static views")
        removals = plan.base_removals[work_id]
        segments_a[work_id] = kept_segments(tokens_a, starts_a, removals)
        segments_b[work_id] = kept_segments(tokens_b, starts_b, removals)
        changed += sum(
            word_a != word_b
            for line_a, line_b in zip(segments_a[work_id], segments_b[work_id], strict=True)
            for word_a, word_b in zip(line_a, line_b, strict=True)
        )
    total = sum(token_count(segments) for segments in segments_a.values())
    return segments_a, segments_b, (len(plan.work_ids), total, changed)


def _measure_field(name: str, job: _FieldJob, report: Report) -> _FieldResult:
    """Train the field's cells' three bootstrap sets, decade by decade, and measure each comparison.

    Reports each model as it is trained, so the progress moves within a field.
    """
    movements: list[tuple[object, ...]] = []
    doses: list[tuple[object, ...]] = []
    tokens_a: Counter[str] = Counter()
    tokens_b: Counter[str] = Counter()
    previous: dict[VersionKey, KeyedVectors] = {}
    previous_decade = 0
    models = 0
    for decade, plan in job.cells.items():
        cell = f"{name}_{decade}"
        view_a = load_cell_view(job.control, plan)
        view_b = load_cell_view(job.treatment, plan)
        segments_a, segments_b, dose = cell_segments(view_a, view_b, plan)
        doses.append((name, decade, *dose))
        tokens_a.update(_target_tokens(segments_a, job.targets))
        tokens_b.update(_target_tokens(segments_b, job.targets))
        pools = {
            "averaged_base": segments_a,
            "averaged_noise": segments_a,
            "averaged_treatment": segments_b,
        }
        current: dict[VersionKey, KeyedVectors] = {}
        for key, lines in averaged_versions(pools, cell, job.bootstraps):
            current[key] = train_lines(lines, svd_backend=job.svd_backend)
            models += 1
            report(1)
        if previous:
            movements.extend(
                compare_averaged(name, previous_decade, decade, previous, current, job.targets)
            )
        previous, previous_decade = current, decade
    return _FieldResult(movements, doses, tokens_a, tokens_b, models)


def _target_tokens(segments: Segments, targets: frozenset[str]) -> Counter[str]:
    """The target words' tokens in a cell's lines."""
    return Counter(
        word for lines in segments.values() for line in lines for word in line if word in targets
    )
