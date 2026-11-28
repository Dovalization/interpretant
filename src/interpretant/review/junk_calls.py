"""The junk gate's call on each record and its quality (`docs/junk_gate.md`).

Since the deviation of 2026-11-28, a record is called junk when two decision models both call it
junk. A decision model is not trained, so its cut-off needs no inner cross-fitting. L's records
fall in folds by the hash of their source, so one journal never sits on both sides. Each fold's
cut-off is the lowest probability that still clears 97% of the other folds' abstracts, and the
final cut-off is set the same way on all of L. L's records are called by their own fold's
cut-offs, every other record by the final ones. The calls are read against the judges' junk
records on L: sensitivity and specificity, false alarms by decade and by field, and each decade's
correction.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.stats import norm

from interpretant.catalog.junk_gate import JunkLabels
from interpretant.review.sampling_plans import compute_wilson_interval

# The label sets a record's label comes from: L ("labelled") and E ("enrichment")
LABEL_SETS = ("labelled", "enrichment")
# The quality bar, on L's held-out calls, as point estimates
SENSITIVITY_BAR = 0.70
SPECIFICITY_BAR = 0.97
# The two-sided 95% interval of a correction term
_INTERVAL_CONFIDENCE = 0.95


@dataclass(frozen=True)
class RecordLabels:
    """Each labelled record's label and set, and how many records' items disagree on junk.

    ``table`` has one row per record, sorted by work id: ``work_id``, ``is_junk`` and
    ``in_labelled_sample`` (in L; otherwise only in E).
    """

    table: pl.DataFrame
    disagreeing_count: int


@dataclass(frozen=True)
class Rate:
    """A share of a count, with its Wilson interval."""

    count: int
    total: int
    share: float
    low: float
    high: float


@dataclass(frozen=True)
class CallQuality:
    """The held-out calls' sensitivity and specificity against the judges' junk records."""

    sensitivity: Rate
    specificity: Rate

    @property
    def meets_bar(self) -> bool:
        """Whether both clear the quality bar as point estimates (0.70 and 0.97)."""
        return (
            self.sensitivity.share >= SENSITIVITY_BAR and self.specificity.share >= SPECIFICITY_BAR
        )


@dataclass(frozen=True)
class MeanDifference:
    """A mean over items, with its 95% interval from the normal approximation."""

    mean: float
    low: float
    high: float
    items: int


@dataclass(frozen=True)
class CutOffs:
    """A decision model's cut-offs: one per fold of L, set on the other folds, and the final one."""

    by_fold: dict[int, float]
    final: float

    @property
    def lowest(self) -> float:
        """The lowest of them: a probability under it is never called junk."""
        return min([*self.by_fold.values(), self.final])


def build_record_labels(labels: JunkLabels) -> RecordLabels:
    """Each record's label from L's and E's items: junk when its verdict is ``junk``.

    A record with several items takes the label of its item with the smallest id; it counts as
    disagreeing when its items do not all agree on junk. A record with an item in L is in L.
    """
    rows = [
        (item_id, label.work_id, label.verdict == "junk", label.set == "labelled")
        for item_id, label in labels.items.items()
        if label.set in LABEL_SETS
    ]
    items = pl.DataFrame(
        rows,
        schema={
            "item_id": pl.String(),
            "work_id": pl.String(),
            "is_junk": pl.Boolean(),
            "in_labelled_sample": pl.Boolean(),
        },
        orient="row",
    )
    records = (
        items.sort("item_id")
        .group_by("work_id", maintain_order=True)
        .agg(
            pl.col("is_junk").first(),
            pl.col("in_labelled_sample").any(),
            (pl.col("is_junk").n_unique() > 1).alias("disagrees"),
        )
        .sort("work_id")
    )
    return RecordLabels(
        table=records.select("work_id", "is_junk", "in_labelled_sample"),
        disagreeing_count=int(records["disagrees"].sum()),
    )


def build_sources(records: pl.DataFrame) -> list[str]:
    """Each record's source: its ISSN-L, else its source name, else its own work id.

    ``records`` has ``work_id``, ``issn_l`` and ``source_name``; an empty name counts as none.
    """
    blank_as_null = [
        pl.when(pl.col(name).str.strip_chars() != "").then(pl.col(name))
        for name in ("issn_l", "source_name")
    ]
    sources: list[str] = records.select(pl.coalesce(*blank_as_null, "work_id"))[:, 0].to_list()
    return sources


def build_folds(sources: Sequence[str], *, folds: int, depth: int) -> NDArray[np.int64]:
    """Each source's fold: digit ``depth`` of its sha256 written in base ``folds``.

    Depth 0 is ``sha256(source) % folds``, the cross-fitting's folds of L. Depth 1 is the next
    digit, so an inner cross-fitting splits a training set (four outer folds) into ``folds`` folds
    again, each still holding whole sources.
    """
    divisor = folds**depth
    return np.array(
        [
            int(hashlib.sha256(source.encode()).hexdigest(), 16) // divisor % folds
            for source in sources
        ],
        dtype=np.int64,
    )


def compute_cut_off(abstract_scores: NDArray[np.float64], *, target: float) -> float:
    """The lowest score that at least ``target`` of the abstracts score below.

    With k the smallest count reaching ``target`` of the scores, it is the next float above the
    k-th smallest score: the most sensitive cut-off that still clears ``target`` of them.

    Raises:
        ValueError: if there are no scores, or the target is not above 0 and at most 1.
    """
    if not abstract_scores.size:
        raise ValueError("no abstract's score to set the cut-off on")
    if not 0 < target <= 1:
        raise ValueError(f"a specificity target of {target} is not above 0 and at most 1")
    # The target as a fraction, so 0.97 of 100 is 97 and not 97.00000000000001
    needed = math.ceil(Fraction(str(target)) * abstract_scores.size)
    return float(np.nextafter(np.sort(abstract_scores)[needed - 1], np.inf))


def set_cut_offs(
    probability: NDArray[np.float64],
    is_junk: NDArray[np.bool_],
    folds: NDArray[np.int64],
    *,
    fold_count: int,
    target: float,
) -> CutOffs:
    """An untrained model's cut-offs on L: each fold's set on the other folds, and the final one.

    ``probability``, ``is_junk`` and ``folds`` hold L's records, in one order.

    Raises:
        ValueError: if a fold leaves no abstract outside it, or L has none.
    """
    by_fold = {}
    for fold in range(fold_count):
        is_held_out = folds == fold
        if is_held_out.any():
            by_fold[fold] = compute_cut_off(probability[~is_held_out & ~is_junk], target=target)
    return CutOffs(by_fold=by_fold, final=compute_cut_off(probability[~is_junk], target=target))


def call_records(
    probability: NDArray[np.float64], folds: NDArray[np.int64], cut_offs: CutOffs
) -> NDArray[np.bool_]:
    """Each record's call by one model: L's at their fold's cut-off, the others at the final one.

    A record with a fold of 0 or more is one of L's; -1 marks any other. A missing probability
    (NaN) is never junk.

    Raises:
        ValueError: if one of L's records has a fold without a cut-off.
    """
    missing = set(np.unique(folds[folds >= 0]).tolist()) - set(cut_offs.by_fold)
    if missing:
        raise ValueError(f"the folds {sorted(missing)} have no cut-off")
    thresholds = np.full(probability.size, cut_offs.final)
    for fold, cut_off in cut_offs.by_fold.items():
        thresholds[folds == fold] = cut_off
    return np.asarray(probability >= thresholds, dtype=np.bool_)


def call_by_agreement(
    screen: NDArray[np.float64],
    confirm: NDArray[np.float64],
    folds: NDArray[np.int64],
    screen_cut_offs: CutOffs,
    confirm_cut_offs: CutOffs,
) -> NDArray[np.bool_]:
    """Each record's call: junk when both models' probabilities reach their cut-offs."""
    return call_records(screen, folds, screen_cut_offs) & call_records(
        confirm, folds, confirm_cut_offs
    )


def compute_rate(count: int, total: int) -> Rate:
    """The share ``count`` of ``total``, with its Wilson interval.

    Raises:
        ValueError: if ``total`` is zero.
    """
    low, high = compute_wilson_interval(count, total)
    return Rate(count=count, total=total, share=count / total, low=low, high=high)


def compute_quality(is_junk: NDArray[np.bool_], predicted_junk: NDArray[np.bool_]) -> CallQuality:
    """Sensitivity and specificity of the calls against the labels, record by record.

    Raises:
        ValueError: if the labels hold no junk record or no abstract.
    """
    if is_junk.all() or not is_junk.any():
        raise ValueError("sensitivity and specificity need both junk records and abstracts")
    return CallQuality(
        sensitivity=compute_rate(int((is_junk & predicted_junk).sum()), int(is_junk.sum())),
        specificity=compute_rate(int((~is_junk & ~predicted_junk).sum()), int((~is_junk).sum())),
    )


def compute_false_alarms(records: pl.DataFrame, by: str) -> dict[str, Rate]:
    """The share of the abstracts called junk, for each value of the column ``by``.

    ``records`` has ``is_junk``, ``predicted_junk`` and ``by``. A list column (a record's fields)
    counts each record under each of its values. The values come sorted.
    """
    abstracts = records.filter(~pl.col("is_junk"))
    if isinstance(abstracts.schema[by], pl.List):
        abstracts = abstracts.explode(by)
    counts = (
        abstracts.group_by(by)
        .agg(pl.col("predicted_junk").sum().alias("count"), pl.len().alias("total"))
        .sort(by)
    )
    return {
        str(value): compute_rate(int(count), int(total))
        for value, count, total in counts.iter_rows()
    }


def compute_corrections(items: pl.DataFrame) -> dict[int, MeanDifference]:
    """Each decade's mean of the judges' junk-usage label minus the call on L.

    ``items`` has one row per labelled usage: ``decade``, ``is_junk_usage`` (the judges' label)
    and ``predicted_junk`` (the held-out call on its record). The mean is the rectifier of
    prediction-powered inference (Angelopoulos et al. 2023, "Prediction-powered inference"), with
    its 95% interval from the normal approximation and the sample standard deviation.

    Raises:
        ValueError: if a decade has fewer than two items, where the interval has no width.
    """
    differences = items.select(
        "decade",
        (pl.col("is_junk_usage").cast(pl.Int64) - pl.col("predicted_junk").cast(pl.Int64)).alias(
            "difference"
        ),
    )
    critical = float(norm.ppf(0.5 + _INTERVAL_CONFIDENCE / 2))
    corrections: dict[int, MeanDifference] = {}
    for (decade,), group in differences.sort("decade").group_by("decade", maintain_order=True):
        values = group["difference"].to_numpy().astype(np.float64)
        if values.size < 2:
            raise ValueError(f"decade {decade} has {values.size} labelled usage, not two or more")
        mean = float(values.mean())
        margin = critical * float(values.std(ddof=1)) / math.sqrt(values.size)
        corrections[int(decade)] = MeanDifference(
            mean=mean, low=mean - margin, high=mean + margin, items=values.size
        )
    return corrections
