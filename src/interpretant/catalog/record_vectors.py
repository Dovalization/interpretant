"""The record vectors (`docs/junk_gate.md`, the classifier), saved in parts.

One row per record behind a drawn usage: its work id and the frozen text encoder's vector of its
kept text, 768 float32 numbers normalised to length 1. One part per part of the usages table; a
work is written in one part only.
"""

from __future__ import annotations

from pathlib import Path

import pandera.polars as pa
import polars as pl
from pandera.polars import PolarsData

from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.parts import part_path

VECTOR_SIZE = 768
RECORD_VECTORS_COLUMNS = pl.Schema(
    {"work_id": pl.String(), "vector": pl.Array(pl.Float32(), VECTOR_SIZE)}
)
# float32 rounding leaves a normalised vector's length a little off 1
_LENGTH_TOLERANCE = 1e-3
_SUFFIX = ".parquet"


def _vectors_have_length_one(data: PolarsData) -> pl.LazyFrame:
    length = pl.col("vector").arr.eval(pl.element().pow(2)).arr.sum().sqrt()
    checked: pl.LazyFrame = data.lazyframe.select((length - 1).abs() < _LENGTH_TOLERANCE)
    return checked


RECORD_VECTORS_SCHEMA = pa.DataFrameSchema(
    {
        "work_id": pa.Column(
            RECORD_VECTORS_COLUMNS["work_id"],
            checks=pa.Check.str_startswith("https://openalex.org/W"),
        ),
        "vector": pa.Column(RECORD_VECTORS_COLUMNS["vector"]),
    },
    checks=[pa.Check(_vectors_have_length_one, name="every vector has length 1")],
    unique=["work_id"],
    strict=True,
    name="record_vectors",
    description="One row per record behind a drawn usage: the frozen encoder's vector of its text.",
)


def save_record_vectors(catalog: CatalogConfig, vectors: pl.DataFrame, part: str) -> Path:
    """Check one part's vectors against the schema and write them, replacing the part.

    Raises:
        ValueError: if the part name is not safe as a file name.
        pandera.errors.SchemaError: if the rows break the schema; nothing is written.
    """
    validated: pl.DataFrame = RECORD_VECTORS_SCHEMA.validate(vectors)
    target = part_path(catalog.record_vectors.path, part, _SUFFIX)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(f"{_SUFFIX}.partial")
    validated.write_parquet(staging)
    staging.replace(target)
    return target


def load_record_vectors(catalog: CatalogConfig) -> pl.LazyFrame:
    """Every saved part as one lazy table.

    Raises:
        FileNotFoundError: if no part has been saved yet.
    """
    folder = catalog.record_vectors.path
    parts = sorted(folder.glob(f"*{_SUFFIX}"))
    if not parts:
        raise FileNotFoundError(f"no part of the record vectors under {folder}")
    return pl.scan_parquet(parts, schema=RECORD_VECTORS_COLUMNS)
