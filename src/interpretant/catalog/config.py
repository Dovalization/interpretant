"""The catalog file: where each dataset lives and in what format, validated when it is read."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

CATALOG_FILE = Path(__file__).resolve().parents[3] / "catalog.yaml"


class DatasetEntry(BaseModel):
    """One dataset's location and storage format."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    format: Literal["parquet", "yaml", "json", "jsonl", "jsonl.gz", "docbin", "markdown", "tsv"]


class CatalogConfig(BaseModel):
    """Every dataset the pipelines read or write, one typed field per dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_abstracts: DatasetEntry
    work_metadata: DatasetEntry
    crossref_records: DatasetEntry
    pubmed_records: DatasetEntry
    openalex_works: DatasetEntry
    crossref_works: DatasetEntry
    pubmed_works: DatasetEntry
    annotated_abstracts: DatasetEntry
    abstracts: DatasetEntry
    corpus_report: DatasetEntry
    corpus_comparison: DatasetEntry
    corpus_metrics: DatasetEntry
    run_log: DatasetEntry
    cleaning_rules: DatasetEntry
    cleaning_patterns: DatasetEntry
    doi_corrections: DatasetEntry
    record_types: DatasetEntry
    kept_joined_words: DatasetEntry
    kept_apart_words: DatasetEntry
    glued_words: DatasetEntry
    split_words: DatasetEntry
    lower_glued_words: DatasetEntry
    repaired_texts: DatasetEntry
    annotations: DatasetEntry
    sentences: DatasetEntry
    static_view: DatasetEntry
    lemma_exceptions: DatasetEntry
    words: DatasetEntry
    spellings: DatasetEntry
    sentence_languages: DatasetEntry


def read_catalog(catalog_file: Path = CATALOG_FILE) -> CatalogConfig:
    """Read the catalog file, with each dataset's path resolved against the file's folder.

    Raises:
        FileNotFoundError: if the catalog file does not exist.
        pydantic.ValidationError: if a dataset is missing, misspelt or has an unknown format.
    """
    entries = CatalogConfig.model_validate(yaml.safe_load(catalog_file.read_text(encoding="utf-8")))
    root = catalog_file.parent
    resolved = {
        name: entry.model_copy(update={"path": root / entry.path}) for name, entry in entries
    }
    return CatalogConfig(**resolved)
