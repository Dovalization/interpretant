"""Tests for drift metric functions."""

from __future__ import annotations

import numpy as np
import pytest

from interpretant.drift.metrics import (
    cosine_distance,
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
