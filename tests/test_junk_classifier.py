"""The junk gate's classifier on synthetic records: features, folds, cut-off, results, file."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from interpretant.review import junk_classifier
from interpretant.review.junk_calls import compute_quality
from interpretant.review.junk_classifier import (
    FEATURE_NAMES,
    ClassifierSpec,
    LinearScorer,
    TrainingSet,
    compute_removed_share,
    compute_surface_features,
    cross_fit,
    format_classifier,
    parse_classifier,
    train_classifier,
    train_final_classifier,
)

SPEC = ClassifierSpec(
    regularisation=1.0,
    class_weight="balanced",
    folds=5,
    specificity_target=0.97,
    max_iterations=1000,
)
VECTOR_SIZE = 8


def build_training(
    labelled_count: int = 400, enrichment_count: int = 60, separation: float = 10.0, seed: int = 7
) -> TrainingSet:
    """L's records, a quarter junk, and E's, mostly junk; junk records lean along one dimension.

    The records share 40 sources between them, so the folds hold several records of each.
    """
    generator = np.random.default_rng(seed)
    count = labelled_count + enrichment_count
    in_labelled_sample = np.arange(count) < labelled_count
    junk_share = np.where(in_labelled_sample, 0.25, 0.8)
    is_junk = generator.random(count) < junk_share
    vectors = generator.standard_normal((count, VECTOR_SIZE))
    vectors[:, 0] += separation * is_junk
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    features = generator.standard_normal((count, len(FEATURE_NAMES)))
    features[:, 1] += 4 * is_junk
    return TrainingSet(
        vectors=vectors.astype(np.float32),
        features=features,
        is_junk=is_junk,
        in_labelled_sample=in_labelled_sample,
        sources=tuple(f"source-{index % 40}" for index in range(count)),
    )


def test_surface_features_read_capitals_digits_lists_and_endings() -> None:
    kept = "CONTENTS. Methods.-Results.-Index of 1990"

    features = dict(
        zip(
            FEATURE_NAMES,
            compute_surface_features(
                kept, has_finite_verb=False, removed_share=0.25, unknown_word_share=0.5
            ),
            strict=True,
        )
    )

    assert features["log_characters"] == pytest.approx(math.log(len(kept)))
    assert features["log_words"] == pytest.approx(math.log(5))
    assert features["log_sentences"] == pytest.approx(math.log(2))
    assert features["capitalised_word_share"] == pytest.approx(4 / 5)
    assert features["capitals_word_share"] == pytest.approx(1 / 5)
    assert features["digit_share"] == pytest.approx(4 / len(kept))
    assert features["list_separators_per_100_words"] == pytest.approx(100 * 2 / 5)
    assert features["has_finite_verb"] == 0.0
    assert features["unknown_word_share"] == 0.5
    assert features["removed_share"] == 0.25
    assert features["ends_with_stop"] == 0.0
    assert features["words_per_sentence"] == pytest.approx(5 / 2)


def test_surface_features_of_prose_end_with_a_stop_and_count_no_single_capital() -> None:
    features = dict(
        zip(
            FEATURE_NAMES,
            compute_surface_features(
                "A model learns. It works!",
                has_finite_verb=True,
                removed_share=0.0,
                unknown_word_share=0.0,
            ),
            strict=True,
        )
    )

    assert features["capitals_word_share"] == 0.0
    assert features["ends_with_stop"] == 1.0
    assert features["has_finite_verb"] == 1.0
    assert features["punctuation_share"] == pytest.approx(2 / 25)


def test_the_removed_share_counts_overlapping_ranges_once() -> None:
    assert compute_removed_share(100, [(0, 10), (5, 20), (90, 100)]) == pytest.approx(0.3)
    assert compute_removed_share(0, []) == 0.0


def test_cross_fitted_folds_never_share_a_source() -> None:
    training = build_training()

    result = cross_fit(training, SPEC)

    labelled = np.flatnonzero(training.in_labelled_sample)
    folds_by_source: dict[str, set[int]] = {}
    for row in labelled:
        folds_by_source.setdefault(training.sources[row], set()).add(int(result.folds[row]))
    assert all(len(folds) == 1 for folds in folds_by_source.values())
    assert len({fold for folds in folds_by_source.values() for fold in folds}) > 1


def test_enrichment_records_are_never_held_out_and_always_trained_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = build_training()
    fitted_rows: list[NDArray[np.intp]] = []
    fit = junk_classifier._fit_scorer

    def record_rows(
        training: TrainingSet, rows: NDArray[np.intp], spec: ClassifierSpec
    ) -> LinearScorer:
        fitted_rows.append(rows)
        return fit(training, rows, spec)

    monkeypatch.setattr(junk_classifier, "_fit_scorer", record_rows)

    result = cross_fit(training, SPEC)

    enrichment = set(np.flatnonzero(~training.in_labelled_sample))
    assert (result.folds[~training.in_labelled_sample] == -1).all()
    assert np.isnan(result.scores[~training.in_labelled_sample]).all()
    assert not np.isnan(result.scores[training.in_labelled_sample]).any()
    assert fitted_rows
    assert all(enrichment <= set(rows) for rows in fitted_rows)


def test_each_cut_off_clears_97_percent_of_the_abstracts_it_was_set_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = build_training()
    seen: list[tuple[NDArray[np.float64], float]] = []
    set_cut_off = junk_classifier.compute_cut_off

    def record_cut_off(abstract_scores: NDArray[np.float64], *, target: float) -> float:
        cut_off = set_cut_off(abstract_scores, target=target)
        seen.append((abstract_scores, cut_off))
        return cut_off

    monkeypatch.setattr(junk_classifier, "compute_cut_off", record_cut_off)

    result = cross_fit(training, SPEC)

    assert len(seen) == len(result.cut_offs) == SPEC.folds
    for abstract_scores, cut_off in seen:
        assert not np.isnan(abstract_scores).any()
        assert (abstract_scores < cut_off).mean() >= 0.97


def test_the_cross_fitting_and_the_final_model_are_identical_on_every_run() -> None:
    first, second = cross_fit(build_training(), SPEC), cross_fit(build_training(), SPEC)

    np.testing.assert_array_equal(first.scores, second.scores)
    np.testing.assert_array_equal(first.predicted_junk, second.predicted_junk)
    assert first.cut_offs == second.cut_offs
    assert format_classifier(train_final_classifier(build_training(), SPEC)) == format_classifier(
        train_final_classifier(build_training(), SPEC)
    )


def test_a_separable_set_is_found_with_high_sensitivity_and_specificity() -> None:
    training = build_training()

    result = cross_fit(training, SPEC)

    labelled = training.in_labelled_sample
    quality = compute_quality(training.is_junk[labelled], result.predicted_junk[labelled])
    assert quality.sensitivity.share >= 0.9
    assert quality.specificity.share >= 0.95
    assert quality.meets_bar
    assert quality.sensitivity.low < quality.sensitivity.share <= quality.sensitivity.high


def test_the_parsed_classifier_scores_exactly_as_the_trained_one() -> None:
    training = build_training()
    classifier = train_final_classifier(training, SPEC)

    parsed = parse_classifier(format_classifier(classifier))

    np.testing.assert_array_equal(
        parsed.scorer.compute_scores(training.vectors, training.features),
        classifier.scorer.compute_scores(training.vectors, training.features),
    )
    assert parsed.cut_off == classifier.cut_off


def test_a_records_score_does_not_depend_on_the_batch_it_is_scored_in() -> None:
    training = build_training(labelled_count=2000)
    scorer = train_final_classifier(training, SPEC).scorer

    everything = scorer.compute_scores(training.vectors, training.features)

    for start, stop in [(0, 1), (3, 20), (7, 330), (100, 2060)]:
        np.testing.assert_array_equal(
            scorer.compute_scores(training.vectors[start:stop], training.features[start:stop]),
            everything[start:stop],
        )


def test_a_classifier_file_with_other_features_is_refused() -> None:
    text = format_classifier(train_final_classifier(build_training(), SPEC))

    with pytest.raises(ValueError, match="features"):
        parse_classifier(text.replace('"removed_share"', '"removed_fraction"'))


def test_a_trained_classifier_calls_the_rows_it_was_not_trained_on() -> None:
    training = build_training()
    rows = np.flatnonzero(np.arange(training.is_junk.size) % 2 == 0)

    classifier = train_classifier(training, rows, SPEC)

    others = np.flatnonzero(np.arange(training.is_junk.size) % 2 == 1)
    called = (
        classifier.scorer.compute_scores(training.vectors[others], training.features[others])
        >= classifier.cut_off
    )
    assert (called == training.is_junk[others]).mean() > 0.9
