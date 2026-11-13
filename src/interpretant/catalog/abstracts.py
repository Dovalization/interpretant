"""The abstracts table, saved in parts: as the spaCy pass leaves it, and after the corpus rules.

Both stages write the same schema. The annotated table (stage 2 of decision agenda 9.1) holds the
pass's removals and drops; the abstracts table (stage 3) adds the corpus-level rules and flags.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from interpretant.catalog.config import CatalogConfig, DatasetEntry
from interpretant.catalog.parts import part_path
from interpretant.catalog.schemas import ABSTRACTS_SCHEMA

_SUFFIX = ".parquet"


def save_abstracts(catalog: CatalogConfig, abstracts: pl.DataFrame, part: str) -> Path:
    """Check one part of the abstracts table against its schema and write it.

    A part holds the abstracts whose first field is one field in one decade. Saving a part again
    replaces it; the file is written whole or not at all.

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
    return _save_part(catalog.abstracts, abstracts, part)


def save_annotated_abstracts(catalog: CatalogConfig, abstracts: pl.DataFrame, part: str) -> Path:
    """Check and write one part of the table as the spaCy pass leaves it (see ``save_abstracts``).

    Raises:
        ValueError: if the part name is not safe as a file name.
        pandera.errors.SchemaError: if the rows break the schema; nothing is written.
    """
    return _save_part(catalog.annotated_abstracts, abstracts, part)


def load_abstracts(catalog: CatalogConfig) -> pl.LazyFrame:
    """Every saved part as one lazy table, so callers stream only what they select.

    The columns and types are checked now; row-level checks ran when each part was saved.

    Raises:
        FileNotFoundError: if no part has been saved yet.
    """
    parts = sorted(catalog.abstracts.path.glob(f"*{_SUFFIX}"))
    if not parts:
        raise FileNotFoundError(f"no part of the abstracts table under {catalog.abstracts.path}")
    table: pl.LazyFrame = ABSTRACTS_SCHEMA.validate(pl.scan_parquet(parts))
    return table


def load_annotated_abstracts(
    catalog: CatalogConfig, parts: Sequence[str]
) -> dict[str, pl.DataFrame]:
    """The named parts of the table as the spaCy pass left it, each checked against the schema.

    Raises:
        FileNotFoundError: naming every part that has not been annotated.
    """
    paths = {part: part_path(catalog.annotated_abstracts.path, part, _SUFFIX) for part in parts}
    missing = [part for part, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"parts not annotated yet: {', '.join(missing)}")
    tables: dict[str, pl.DataFrame] = {
        part: ABSTRACTS_SCHEMA.validate(pl.read_parquet(path)) for part, path in paths.items()
    }
    return tables


def _save_part(dataset: DatasetEntry, abstracts: pl.DataFrame, part: str) -> Path:
    target = part_path(dataset.path, part, _SUFFIX)
    validated: pl.DataFrame = ABSTRACTS_SCHEMA.validate(abstracts)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(f"{_SUFFIX}.partial")
    validated.write_parquet(staging)
    staging.replace(target)
    return target
