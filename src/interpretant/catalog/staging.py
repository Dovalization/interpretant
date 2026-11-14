"""The staging tables: one checked table per source, keyed by work id (decision agenda 4.8).

Each is rebuilt whole from the raw files by the ``build_linked_tables`` stage and written in one
step, so a reader never sees half a table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandera.polars as pa
import polars as pl

from interpretant.catalog.config import CatalogConfig, DatasetEntry
from interpretant.catalog.schemas import (
    CROSSREF_WORKS_SCHEMA,
    OPENALEX_WORKS_SCHEMA,
    PUBMED_WORKS_SCHEMA,
)

StagingSource = Literal["openalex", "crossref", "pubmed"]


def save_staging_table(catalog: CatalogConfig, source: StagingSource, table: pl.DataFrame) -> Path:
    """Check a source's staging table against its schema and write it, replacing the old one.

    Raises:
        pandera.errors.SchemaError: if the table breaks the schema; nothing is written.
    """
    entry, schema = _dataset(catalog, source)
    schema.validate(table)
    entry.path.parent.mkdir(parents=True, exist_ok=True)
    partial = entry.path.with_suffix(".parquet.partial")
    table.write_parquet(partial)
    partial.replace(entry.path)
    return entry.path


def load_staging_table(catalog: CatalogConfig, source: StagingSource) -> pl.LazyFrame:
    """A source's staging table, its columns and types checked; rows were checked when saved.

    Raises:
        FileNotFoundError: if the table has not been built.
    """
    entry, schema = _dataset(catalog, source)
    if not entry.path.exists():
        raise FileNotFoundError(
            f"no {source} staging table: run `interpretant build-linked-tables`"
        )
    table: pl.LazyFrame = schema.validate(pl.scan_parquet(entry.path))
    return table


def _dataset(
    catalog: CatalogConfig, source: StagingSource
) -> tuple[DatasetEntry, pa.DataFrameSchema]:
    if source == "openalex":
        return catalog.openalex_works, OPENALEX_WORKS_SCHEMA
    if source == "crossref":
        return catalog.crossref_works, CROSSREF_WORKS_SCHEMA
    return catalog.pubmed_works, PUBMED_WORKS_SCHEMA
