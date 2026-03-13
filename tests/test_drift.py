"""Tests for drift metric functions."""

from __future__ import annotations

import numpy as np
import pytest

from interpretant.drift.metrics import (
    average_pairwise_distance,
    cosine_distance,
    frequency_corrected_drift,
    neighborhood_shift,
)


class TestCosineDistance:
    def test_identical_vectors_give_zero(self) -> None:
        vector = np.array([1.0, 0.0, 0.0])
        assert cosine_distance(vector, vector) == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_vectors_give_one(self) -> None:
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert cosine_distance(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors_give_two(self) -> None:
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert cosine_distance(a, b) == pytest.approx(2.0, abs=1e-6)

    def test_result_in_range(self, synthetic_model: dict) -> None:
        decades = sorted(synthetic_model.keys())
        for word in list(synthetic_model[decades[0]].keys())[:3]:
            v1 = synthetic_model[decades[0]][word]
            v2 = synthetic_model[decades[-1]][word]
            dist = cosine_distance(v1, v2)
            assert 0.0 <= dist <= 2.0

    def test_zero_vector_returns_one(self) -> None:
        a = np.zeros(10)
        b = np.ones(10)
        assert cosine_distance(a, b) == pytest.approx(1.0)

    def test_synthetic_drift_is_positive(self, synthetic_model: dict) -> None:
        decades = sorted(synthetic_model.keys())
        t0 = decades[0]
        t1 = decades[-1]
        word = "network"
        dist = cosine_distance(synthetic_model[t0][word], synthetic_model[t1][word])
        assert dist > 0.0


class TestNeighborhoodShift:
    def test_result_in_range(self, synthetic_model: dict) -> None:
        decades = sorted(synthetic_model.keys())
        vectors_t1 = synthetic_model[decades[0]]
        vectors_t2 = synthetic_model[decades[-1]]
        shift = neighborhood_shift("network", vectors_t1, vectors_t2, k=3)
        assert 0.0 <= shift <= 1.0

    def test_identical_spaces_give_zero(self, synthetic_model: dict) -> None:
        decades = sorted(synthetic_model.keys())
        vectors = synthetic_model[decades[0]]
        shift = neighborhood_shift("network", vectors, vectors, k=3)
        assert shift == pytest.approx(0.0)

    def test_missing_word_raises(self, synthetic_model: dict) -> None:
        decades = sorted(synthetic_model.keys())
        vectors = synthetic_model[decades[0]]
        with pytest.raises(KeyError):
            neighborhood_shift("nonexistent_word_xyz", vectors, vectors, k=3)

    def test_all_words(self, synthetic_model: dict, synthetic_vocab: list[str]) -> None:
        decades = sorted(synthetic_model.keys())
        vectors_t1 = synthetic_model[decades[0]]
        vectors_t2 = synthetic_model[decades[-1]]
        for word in synthetic_vocab:
            shift = neighborhood_shift(word, vectors_t1, vectors_t2, k=3)
            assert 0.0 <= shift <= 1.0


class TestFrequencyCorrectedDrift:
    def test_high_frequency_preserves_drift(self) -> None:
        corrected = frequency_corrected_drift(0.5, freq_t1=0.01, freq_t2=0.01)
        assert corrected > 0.0

    def test_zero_frequency_gives_near_zero(self) -> None:
        corrected = frequency_corrected_drift(0.5, freq_t1=0.0, freq_t2=0.01)
        assert corrected < 0.5

    def test_corrected_never_exceeds_raw(self) -> None:
        raw = 0.8
        corrected = frequency_corrected_drift(raw, freq_t1=0.001, freq_t2=0.002)
        assert corrected <= raw


class TestAveragePairwiseDistance:
    def test_single_vector_returns_zero(self) -> None:
        vectors = [np.array([1.0, 0.0, 0.0])]
        assert average_pairwise_distance(vectors) == pytest.approx(0.0)

    def test_identical_vectors_return_zero(self) -> None:
        vector = np.array([1.0, 0.0, 0.0])
        assert average_pairwise_distance([vector, vector]) == pytest.approx(0.0)

    def test_result_in_range(self, synthetic_model: dict) -> None:
        decades = sorted(synthetic_model.keys())
        word = "information"
        all_vectors = [synthetic_model[decade][word] for decade in decades]
        result = average_pairwise_distance(all_vectors)
        assert 0.0 <= result <= 2.0
