"""Shared pytest fixtures for interpretant tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from spacy.language import Language

from interpretant.catalog import (
    CatalogConfig,
    CleaningPatterns,
    load_cleaning_patterns,
    read_catalog,
)
from interpretant.corpus.annotation import build_pipeline
from tests.corpus_catalog import write_corpus_catalog

VOCAB = ["network", "atom", "information", "model", "evolution", "structure"]
DECADES = [1960, 1970, 1980, 1990, 2000, 2010]
VECTOR_DIM = 300
SEED = 42


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


@pytest.fixture
def catalog_text() -> str:
    """A catalog file's content with every dataset, its paths relative to the file."""
    return (
        "raw_abstracts:\n  path: data/raw\n  format: jsonl.gz\n"
        "work_metadata:\n  path: data/metadata\n  format: jsonl.gz\n"
        "annotated_abstracts:\n  path: data/annotated\n  format: parquet\n"
        "abstracts:\n  path: data/abstracts\n  format: parquet\n"
        "cleaning_rules:\n  path: curated/rules.yaml\n  format: yaml\n"
        "cleaning_patterns:\n  path: curated/patterns.jsonl\n  format: jsonl\n"
        "annotations:\n  path: data/annotations\n  format: docbin\n"
    )


@pytest.fixture
def catalog(tmp_path: Path, catalog_text: str) -> CatalogConfig:
    """A catalog whose datasets live under the test's temporary folder."""
    catalog_file = tmp_path / "catalog.yaml"
    catalog_file.write_text(catalog_text)
    return read_catalog(catalog_file)


@pytest.fixture
def corpus_catalog_file(tmp_path: Path) -> Path:
    """The small corpus of ``tests/corpus_catalog.py`` in a temporary catalog."""
    return write_corpus_catalog(tmp_path)


@pytest.fixture(scope="session")
def patterns() -> CleaningPatterns:
    """The repository's cleaning rules and patterns."""
    return load_cleaning_patterns(read_catalog())


@pytest.fixture(scope="session")
def nlp(patterns: CleaningPatterns) -> Language:
    """The spaCy pipeline of the pass (en_core_web_sm, a pinned dependency), loaded once.

    Its language component reads English, German, French and Spanish.
    """
    return build_pipeline(patterns, languages=["en", "de", "fr", "es"], english_threshold=0.9)
