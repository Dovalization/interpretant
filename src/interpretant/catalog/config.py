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
    format: Literal["parquet", "yaml", "jsonl", "jsonl.gz", "docbin"]


class CatalogConfig(BaseModel):
    """Every dataset the pipelines read or write, one typed field per dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_abstracts: DatasetEntry
    work_metadata: DatasetEntry
    annotated_abstracts: DatasetEntry
    abstracts: DatasetEntry
    cleaning_rules: DatasetEntry
    cleaning_patterns: DatasetEntry
    annotations: DatasetEntry


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
