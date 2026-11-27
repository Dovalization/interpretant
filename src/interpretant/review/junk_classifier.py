"""The junk gate's classifier (`docs/junk_gate.md`, "The classifier"): features, folds, cut-off.

A record is scored by a logistic regression on its frozen text vector and 13 surface features of
its kept text, the features standardised on the training records. L's records are cross-fitted in
folds grouped by the hash of their source, so one journal never sits on both sides; E's records are
always trained on and never held out. Every model's cut-off is set inside its own training set, by
an inner cross-fitting of the same kind over that set's L records: the lowest score that at least
97% of those abstracts score below. The results are read on L's held-out calls, against the judges'
junk records, and the final model is trained on all of L and E with its cut-off set the same way.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.special import expit
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from interpretant.catalog.junk_gate import JunkLabels
from interpretant.corpus.sentence_view import merge_ranges
from interpretant.review.sampling_plans import compute_wilson_interval

# The 13 surface features in the spec's order (docs/junk_gate.md, "The surface features")
FEATURE_NAMES = (
    "log_characters",
    "log_words",
    "log_sentences",
    "capitalised_word_share",
    "capitals_word_share",
    "digit_share",
    "punctuation_share",
    "list_separators_per_100_words",
    "has_finite_verb",
    "unknown_word_share",
    "removed_share",
    "ends_with_stop",
    "words_per_sentence",
)
# The label sets the classifier learns from: L ("labelled") and E ("enrichment")
TRAINING_SETS = ("labelled", "enrichment")
# The quality bar, on L's held-out calls, as point estimates
SENSITIVITY_BAR = 0.70
SPECIFICITY_BAR = 0.97
# The separator of a run-on list of titles, as in a book's table of contents ("Methods.-Results")
LIST_SEPARATOR = ".-"
FINAL_STOPS = (".", "?", "!")
# A word is a run of letters: digits, underscores and punctuation split words
_WORD = re.compile(r"[^\W\d_]+")
# A sentence ends at a full stop, question mark or exclamation mark followed by a space
_SENTENCE_BREAK = re.compile(r"(?<=[.?!])\s+")
# Records scored at once: the float32 vectors are widened to float64 one chunk at a time
_SCORING_CHUNK = 65_536
# The two-sided 95% interval of a correction term
_INTERVAL_CONFIDENCE = 0.95

# Advances a caller's progress by the models fitted
Report = Callable[[int], None]


@dataclass(frozen=True)
class ClassifierSpec:
    """How the classifier is trained and its cut-off set (params.yaml's ``junk_gate.classifier``).

    Attributes:
        regularisation: scikit-learn's ``C``, the inverse of the L2 penalty's strength.
        class_weight: scikit-learn's ``class_weight``.
        folds: The folds of the cross-fitting and of each inner cross-fitting.
        specificity_target: The share of the inner held-out abstracts the cut-off clears.
        max_iterations: The solver's iteration budget; a fit that does not converge in it fails.
    """

    regularisation: float
    class_weight: str
    folds: int
    specificity_target: float
    max_iterations: int


@dataclass(frozen=True)
class RecordLabels:
    """Each labelled record's label and set, and how many records' items disagree on junk.

    ``table`` has one row per record, sorted by work id: ``work_id``, ``is_junk`` and
    ``in_labelled_sample`` (in L; otherwise only in E).
    """

    table: pl.DataFrame
    disagreeing_count: int


@dataclass(frozen=True)
class TrainingSet:
    """The labelled records, in one order: vector, raw surface features, label, set and source."""

    vectors: NDArray[np.float32]
    features: NDArray[np.float64]
    is_junk: NDArray[np.bool_]
    in_labelled_sample: NDArray[np.bool_]
    sources: tuple[str, ...]


@dataclass(frozen=True)
class LinearScorer:
    """A fitted logistic regression and the scaling of the surface features it was fitted on.

    ``weights`` holds the vector's dimensions first, then the standardised surface features.
    """

    weights: NDArray[np.float64]
    intercept: float
    feature_means: NDArray[np.float64]
    feature_scales: NDArray[np.float64]

    def compute_scores(
        self, vectors: NDArray[np.float32], features: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Each record's junk probability, from its vector and its raw surface features.

        Raises:
            ValueError: if the vectors and features do not have the weights' dimensions.
        """
        vector_size = self.weights.size - self.feature_means.size
        if vectors.shape[1] != vector_size or features.shape[1] != self.feature_means.size:
            raise ValueError(
                f"the scorer takes {vector_size} vector dimensions and {self.feature_means.size} "
                f"features, not {vectors.shape[1]} and {features.shape[1]}"
            )
        vector_weights, feature_weights = self.weights[:vector_size], self.weights[vector_size:]
        scores = np.empty(len(vectors), dtype=np.float64)
        for start in range(0, len(vectors), _SCORING_CHUNK):
            stop = start + _SCORING_CHUNK
            standardised = (features[start:stop] - self.feature_means) / self.feature_scales
            logits = (
                vectors[start:stop].astype(np.float64) @ vector_weights
                + standardised @ feature_weights
                + self.intercept
            )
            scores[start:stop] = expit(logits)
        return scores


@dataclass(frozen=True)
class JunkClassifier:
    """A scorer and its cut-off: a record scoring at or above the cut-off is called junk."""

    scorer: LinearScorer
    cut_off: float


@dataclass(frozen=True)
class CrossFitResult:
    """L's held-out scores, calls and folds, in the training set's order.

    E's rows, never held out, hold NaN, False and -1. ``cut_offs`` holds each outer fold's.
    """

    scores: NDArray[np.float64]
    predicted_junk: NDArray[np.bool_]
    folds: NDArray[np.int64]
    cut_offs: dict[int, float]


@dataclass(frozen=True)
class Rate:
    """A share of a count, with its Wilson interval."""

    count: int
    total: int
    share: float
    low: float
    high: float


@dataclass(frozen=True)
class ClassifierQuality:
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


class _ClassifierDocument(BaseModel):
    """The classifier as its JSON file holds it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_names: tuple[str, ...]
    vector_size: int
    weights: list[float]
    intercept: float
    feature_means: list[float]
    feature_scales: list[float]
    cut_off: float


def compute_surface_features(
    kept: str, *, has_finite_verb: bool, removed_share: float, unknown_word_share: float
) -> tuple[float, ...]:
    """The 13 surface features of a record's kept text, in ``FEATURE_NAMES``' order.

    A word is a run of letters, and a sentence ends at a full stop, question mark or exclamation
    mark followed by a space. A count under 1 counts as 1 in a logarithm and a ratio, and a share
    of nothing is 0. A word is in capitals when it has two letters or more, all capitals, so a
    sentence's opening "A" is not. The finite-verb flag, the removed share and the unknown-word
    share (the OCR score of the whole kept text) come from outside the text.
    """
    words = _WORD.findall(kept)
    word_count = len(words)
    sentence_count = max(1, sum(1 for piece in _SENTENCE_BREAK.split(kept) if piece.strip()))
    character_count = len(kept)
    return (
        math.log(max(1, character_count)),
        math.log(max(1, word_count)),
        math.log(sentence_count),
        _share(sum(word[0].isupper() for word in words), word_count),
        _share(sum(len(word) > 1 and word.isupper() for word in words), word_count),
        _share(sum(character.isdigit() for character in kept), character_count),
        _share(sum(_is_punctuation(character) for character in kept), character_count),
        100 * kept.count(LIST_SEPARATOR) / max(1, word_count),
        float(has_finite_verb),
        unknown_word_share,
        removed_share,
        float(kept.rstrip().endswith(FINAL_STOPS)),
        max(1, word_count) / sentence_count,
    )


def compute_removed_share(text_length: int, removed: Sequence[tuple[int, int]]) -> float:
    """The share of a text's characters inside its removed ranges, overlaps counted once."""
    if text_length == 0:
        return 0.0
    return sum(end - start for start, end in merge_ranges(removed)) / text_length


def build_record_labels(labels: JunkLabels) -> RecordLabels:
    """Each record's label from L's and E's items: junk when its verdict is ``junk``.

    A record with several items takes the label of its item with the smallest id; it counts as
    disagreeing when its items do not all agree on junk. A record with an item in L is in L.
    """
    rows = [
        (item_id, label.work_id, label.verdict == "junk", label.set == "labelled")
        for item_id, label in labels.items.items()
        if label.set in TRAINING_SETS
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


def train_classifier(
    training: TrainingSet, rows: NDArray[np.intp], spec: ClassifierSpec
) -> JunkClassifier:
    """A classifier fitted on ``rows``, with its cut-off set inside them.

    The cut-off comes from an inner cross-fitting over the rows' L records, in folds of depth 1
    (``build_folds``): each inner fold is scored by a model fitted on the other rows, E's always
    among them, and the cut-off clears ``spec.specificity_target`` of the abstracts' scores.

    Raises:
        ValueError: if the rows hold one class only, or no L abstract.
        sklearn.exceptions.ConvergenceWarning: as an error, if a fit does not converge.
    """
    scorer = _fit_scorer(training, rows, spec)
    inner = build_folds([training.sources[row] for row in rows], folds=spec.folds, depth=1)
    labelled = training.in_labelled_sample[rows]
    inner_scores = np.full(rows.size, np.nan)
    for fold in range(spec.folds):
        is_held_out = labelled & (inner == fold)
        if not is_held_out.any():
            continue
        inner_scorer = _fit_scorer(training, rows[~is_held_out], spec)
        held_out = rows[is_held_out]
        inner_scores[is_held_out] = inner_scorer.compute_scores(
            training.vectors[held_out], training.features[held_out]
        )
    is_abstract = labelled & ~training.is_junk[rows]
    cut_off = compute_cut_off(inner_scores[is_abstract], target=spec.specificity_target)
    return JunkClassifier(scorer=scorer, cut_off=cut_off)


def cross_fit(
    training: TrainingSet, spec: ClassifierSpec, *, report: Report | None = None
) -> CrossFitResult:
    """Score each fold of L by a classifier trained on the other folds and on all of E.

    The folds are of depth 0 (``build_folds``). Each fold's classifier has its own cut-off, set
    inside its training set (``train_classifier``), and calls its held-out records. ``report`` is
    told of each outer fold as it ends.
    """
    labelled = np.flatnonzero(training.in_labelled_sample)
    enrichment = np.flatnonzero(~training.in_labelled_sample)
    outer = build_folds(training.sources, folds=spec.folds, depth=0)
    scores = np.full(training.is_junk.size, np.nan)
    predicted = np.zeros(training.is_junk.size, dtype=np.bool_)
    folds = np.full(training.is_junk.size, -1, dtype=np.int64)
    cut_offs: dict[int, float] = {}
    for fold in range(spec.folds):
        held_out = labelled[outer[labelled] == fold]
        if held_out.size:
            rows = np.concatenate([labelled[outer[labelled] != fold], enrichment])
            classifier = train_classifier(training, rows, spec)
            scores[held_out] = classifier.scorer.compute_scores(
                training.vectors[held_out], training.features[held_out]
            )
            predicted[held_out] = scores[held_out] >= classifier.cut_off
            folds[held_out] = fold
            cut_offs[fold] = classifier.cut_off
        if report is not None:
            report(1)
    return CrossFitResult(scores=scores, predicted_junk=predicted, folds=folds, cut_offs=cut_offs)


def train_final_classifier(training: TrainingSet, spec: ClassifierSpec) -> JunkClassifier:
    """The classifier trained on all of L and E, its cut-off set by the same inner rule over L."""
    return train_classifier(training, np.arange(training.is_junk.size), spec)


def compute_rate(count: int, total: int) -> Rate:
    """The share ``count`` of ``total``, with its Wilson interval.

    Raises:
        ValueError: if ``total`` is zero.
    """
    low, high = compute_wilson_interval(count, total)
    return Rate(count=count, total=total, share=count / total, low=low, high=high)


def compute_quality(
    is_junk: NDArray[np.bool_], predicted_junk: NDArray[np.bool_]
) -> ClassifierQuality:
    """Sensitivity and specificity of the calls against the labels, record by record.

    Raises:
        ValueError: if the labels hold no junk record or no abstract.
    """
    if is_junk.all() or not is_junk.any():
        raise ValueError("sensitivity and specificity need both junk records and abstracts")
    return ClassifierQuality(
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
    """Each decade's mean of the judges' junk-usage label minus the classifier's call on L.

    ``items`` has one row per labelled usage: ``decade``, ``is_junk_usage`` (the judges' label)
    and ``predicted_junk`` (the classifier's held-out call on its record). The mean is the
    rectifier of prediction-powered inference (Angelopoulos et al. 2023, "Prediction-powered
    inference"), with its 95% interval from the normal approximation and the sample standard
    deviation.

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


def format_classifier(classifier: JunkClassifier) -> str:
    """The classifier as JSON: weights, intercept, feature scaling, cut-off and feature names.

    Floats are written in their shortest exact form, so the parsed classifier scores alike.
    """
    scorer = classifier.scorer
    document = _ClassifierDocument(
        feature_names=FEATURE_NAMES,
        vector_size=scorer.weights.size - scorer.feature_means.size,
        weights=scorer.weights.tolist(),
        intercept=scorer.intercept,
        feature_means=scorer.feature_means.tolist(),
        feature_scales=scorer.feature_scales.tolist(),
        cut_off=classifier.cut_off,
    )
    return json.dumps(document.model_dump(mode="json"), indent=1) + "\n"


def parse_classifier(text: str) -> JunkClassifier:
    """The classifier its JSON describes.

    Raises:
        pydantic.ValidationError: if the JSON is not a classifier's.
        ValueError: if its features are not this module's, or its sizes disagree.
    """
    document = _ClassifierDocument.model_validate_json(text)
    if document.feature_names != FEATURE_NAMES:
        raise ValueError(f"the classifier's features {document.feature_names} are not these")
    sizes = (len(document.weights), len(document.feature_means), len(document.feature_scales))
    if sizes != (document.vector_size + len(FEATURE_NAMES), *(2 * (len(FEATURE_NAMES),))):
        raise ValueError(f"the classifier's weights and scaling have the sizes {sizes}")
    return JunkClassifier(
        scorer=LinearScorer(
            weights=np.array(document.weights, dtype=np.float64),
            intercept=document.intercept,
            feature_means=np.array(document.feature_means, dtype=np.float64),
            feature_scales=np.array(document.feature_scales, dtype=np.float64),
        ),
        cut_off=document.cut_off,
    )


def _fit_scorer(
    training: TrainingSet, rows: NDArray[np.intp], spec: ClassifierSpec
) -> LinearScorer:
    """A logistic regression fitted on ``rows``: the vectors and the standardised features.

    A feature constant over the rows keeps a scale of 1, as scikit-learn's scaler does.
    """
    features = training.features[rows]
    means = features.mean(axis=0)
    deviations = features.std(axis=0)
    scales = np.where(deviations > 0, deviations, 1.0)
    design = np.hstack([training.vectors[rows].astype(np.float64), (features - means) / scales])
    model = LogisticRegression(
        C=spec.regularisation, class_weight=spec.class_weight, max_iter=spec.max_iterations
    )
    with warnings.catch_warnings():
        # An unconverged fit is not the model the spec describes: it stops the run
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(design, training.is_junk[rows])
    return LinearScorer(
        weights=model.coef_[0].astype(np.float64),
        intercept=float(model.intercept_[0]),
        feature_means=means,
        feature_scales=scales,
    )


def _share(count: int, total: int) -> float:
    return count / total if total else 0.0


def _is_punctuation(character: str) -> bool:
    """Whether the character is in one of Unicode's punctuation categories (P*)."""
    return unicodedata.category(character).startswith("P")
