"""Shared pytest fixtures for interpretant tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from lingua import LanguageDetector
from spacy.language import Language

from interpretant.catalog.config import CatalogConfig, read_catalog
from interpretant.catalog.curated import CleaningPatterns, load_cleaning_patterns
from interpretant.corpus.annotation import build_pipeline
from interpretant.corpus.language import build_language_detector
from interpretant.corpus.removals import build_cleaning_pipeline
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
        "crossref_records:\n  path: data/crossref\n  format: jsonl.gz\n"
        "pubmed_records:\n  path: data/pubmed\n  format: jsonl.gz\n"
        "openalex_works:\n  path: data/openalex_works.parquet\n  format: parquet\n"
        "crossref_works:\n  path: data/crossref_works.parquet\n  format: parquet\n"
        "pubmed_works:\n  path: data/pubmed_works.parquet\n  format: parquet\n"
        "annotated_abstracts:\n  path: data/annotated\n  format: parquet\n"
        "abstracts:\n  path: data/abstracts\n  format: parquet\n"
        "cleaning_rules:\n  path: curated/rules.yaml\n  format: yaml\n"
        "cleaning_patterns:\n  path: curated/patterns.jsonl\n  format: jsonl\n"
        "doi_corrections:\n  path: curated/doi_corrections.yaml\n  format: yaml\n"
        "record_types:\n  path: curated/record_types.yaml\n  format: yaml\n"
        "kept_joined_words:\n  path: curated/kept_joined_words.yaml\n  format: yaml\n"
        "kept_apart_words:\n  path: curated/kept_apart_words.yaml\n  format: yaml\n"
        "glued_words:\n  path: data/glued_words.parquet\n  format: parquet\n"
        "split_words:\n  path: data/split_words.parquet\n  format: parquet\n"
        "lower_glued_words:\n  path: data/lower_glued_words.parquet\n  format: parquet\n"
        "repaired_texts:\n  path: data/repaired_texts\n  format: parquet\n"
        "annotations:\n  path: data/annotations\n  format: docbin\n"
        "sentences:\n  path: data/sentences\n  format: parquet\n"
        "static_view:\n  path: data/static_view\n  format: parquet\n"
        "sentence_view:\n  path: data/sentence_view\n  format: parquet\n"
        "usages:\n  path: data/usages\n  format: parquet\n"
        "annotation_cache:\n  path: data/annotation_cache\n  format: parquet\n"
        "lemma_exceptions:\n  path: curated/lemma_exceptions.tsv\n  format: tsv\n"
        "words:\n  path: curated/words.tsv\n  format: tsv\n"
        "spellings:\n  path: curated/spellings.tsv\n  format: tsv\n"
        "compounds:\n  path: curated/compounds.tsv\n  format: tsv\n"
        "sentence_languages:\n  path: data/sentence_languages\n  format: parquet\n"
        "corpus_report:\n  path: data/corpus_report.md\n  format: markdown\n"
        "corpus_comparison:\n  path: data/corpus_comparison.md\n  format: markdown\n"
        "corpus_metrics:\n  path: metrics/corpus.json\n  format: json\n"
        "run_log:\n  path: metrics/runs.jsonl\n  format: jsonl\n"
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
def nlp() -> Language:
    """The spaCy pipeline of the pass (en_core_web_sm, a pinned dependency), loaded once."""
    return build_pipeline()


@pytest.fixture(scope="session")
def detector() -> LanguageDetector:
    """A lingua detector for English, German, French and Spanish, built once."""
    return build_language_detector(["en", "de", "fr", "es"])


@pytest.fixture(scope="session")
def cleaning(patterns: CleaningPatterns) -> Language:
    """The blank pipeline that matches the repository's cleaning patterns."""
    return build_cleaning_pipeline(patterns)
