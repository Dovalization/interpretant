"""The abstracts table, saved in parts and read lazily as one table."""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.schemas import ABSTRACTS_SCHEMA

_PART_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def save_abstracts(catalog: CatalogConfig, abstracts: pl.DataFrame, part: str) -> Path:
    """Check one part of the abstracts table against its schema and write it.

    A part holds the abstracts of one input cell or download. Saving a part again replaces it;
    the file is written whole or not at all.

    Args:
        catalog: the catalog, which says where the table lives.
        abstracts: the rows of this part.
        part: the part's name, such as ``ai_2010``: lowercase letters, digits, ``_``, ``.``, ``-``.

    Returns:
        The file written.

    Raises:
        ValueError: if the part name is not safe as a file name.
        pandera.errors.SchemaError: if the rows break the schema; nothing is written.
    """
    if not _PART_NAME.match(part):
        raise ValueError(f"part name {part!r} must match {_PART_NAME.pattern}")
    validated: pl.DataFrame = ABSTRACTS_SCHEMA.validate(abstracts)
    folder = catalog.abstracts.path
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{part}.parquet"
    staging = target.with_suffix(".parquet.partial")
    validated.write_parquet(staging)
    staging.replace(target)
    return target


def load_abstracts(catalog: CatalogConfig) -> pl.LazyFrame:
    """Every saved part as one lazy table, so callers stream only what they select.

    The columns and types are checked now; row-level checks ran when each part was saved.

    Raises:
        FileNotFoundError: if no part has been saved yet.
    """
    parts = sorted(catalog.abstracts.path.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no part of the abstracts table under {catalog.abstracts.path}")
    table: pl.LazyFrame = ABSTRACTS_SCHEMA.validate(pl.scan_parquet(parts))
    return table
