"""Shared pytest fixtures for interpretant tests."""

from __future__ import annotations

import numpy as np
import pytest

VOCAB = ["network", "atom", "information", "model", "evolution", "structure"]
DECADES = [1960, 1970, 1980, 1990, 2000, 2010]
VECTOR_DIM = 300
SEED = 42


@pytest.fixture(scope="session")
def synthetic_vocab() -> list[str]:
    """Return the canonical 6-word demo vocabulary."""
    return list(VOCAB)


@pytest.fixture(scope="session")
def synthetic_model() -> dict[int, dict[str, np.ndarray]]:
    """Return seeded synthetic embedding model: {decade: {word: vector}}.

    Vectors are 300-dimensional, unit-normalised, and jittered per decade
    to simulate plausible semantic drift.
    """
    rng = np.random.default_rng(SEED)
    raw = rng.standard_normal((len(VOCAB), VECTOR_DIM))
    base_vectors = {word: v / np.linalg.norm(v) for word, v in zip(VOCAB, raw, strict=True)}

    model: dict[int, dict[str, np.ndarray]] = {}
    for decade_idx, decade in enumerate(DECADES):
        noise_scale = 0.05 * (decade_idx + 1)
        decade_vectors: dict[str, np.ndarray] = {}
        for word in VOCAB:
            noise = rng.standard_normal(VECTOR_DIM) * noise_scale
            vector = base_vectors[word] + noise
            decade_vectors[word] = vector / np.linalg.norm(vector)
        model[decade] = decade_vectors
    return model
