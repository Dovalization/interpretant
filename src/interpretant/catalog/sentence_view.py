"""The contextual sentence view, saved in parts (decision agenda 2.2; `docs/sentence_view.md`).

One row per sentence of a kept work, in the abstracts table's order: its work, its number within
the work (from 0), its character span in the repaired text, its language and confidence as the
language stage read them, its word count, and its exclusion: ``removed`` when every non-space
character lies in removed ranges, ``cut_off`` for the unfinished last sentence of a truncated work,
null otherwise. A part is written whole or not at all.
"""

from __future__ import annotations

from pathlib import Path

import pandera.polars as pa
import polars as pl
from pandera.polars import PolarsData

from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.parts import part_path

REMOVED = "removed"
CUT_OFF = "cut_off"
SENTENCE_VIEW_COLUMNS = pl.Schema(
    {
        "work_id": pl.String(),
        "sentence_number": pl.Int64(),
        "start": pl.Int64(),
        "end": pl.Int64(),
        "language": pl.String(),
        "confidence": pl.Float64(),
        "word_count": pl.Int64(),
        "exclusion": pl.String(),
    }
)
_SUFFIX = ".parquet"


def _spans_are_valid(data: PolarsData) -> pl.LazyFrame:
    checked: pl.LazyFrame = data.lazyframe.select(
        (pl.col("start") >= 0) & (pl.col("start") < pl.col("end"))
    )
    return checked


SENTENCE_VIEW_SCHEMA = pa.DataFrameSchema(
    {
        "work_id": pa.Column(
            SENTENCE_VIEW_COLUMNS["work_id"],
            checks=pa.Check.str_startswith("https://openalex.org/W"),
        ),
        "sentence_number": pa.Column(
            SENTENCE_VIEW_COLUMNS["sentence_number"], checks=pa.Check.ge(0)
        ),
        "start": pa.Column(SENTENCE_VIEW_COLUMNS["start"]),
        "end": pa.Column(SENTENCE_VIEW_COLUMNS["end"]),
        "language": pa.Column(SENTENCE_VIEW_COLUMNS["language"]),
        "confidence": pa.Column(SENTENCE_VIEW_COLUMNS["confidence"]),
        "word_count": pa.Column(SENTENCE_VIEW_COLUMNS["word_count"], checks=pa.Check.ge(0)),
        "exclusion": pa.Column(
            SENTENCE_VIEW_COLUMNS["exclusion"],
            nullable=True,
            checks=pa.Check.isin([REMOVED, CUT_OFF]),
        ),
    },
    checks=[pa.Check(_spans_are_valid, name="each span starts at or after 0, before its end")],
    unique=["work_id", "sentence_number"],
    strict=True,
    name="sentence_view",
    description=(
        "One row per sentence of a kept work, with the sentence's span, language, word count and "
        "exclusion from the usage draw (docs/sentence_view.md)."
    ),
)


def save_sentence_view(catalog: CatalogConfig, view: pl.DataFrame, part: str) -> Path:
    """Check one part's sentence view against its schema and write it, replacing the part.

    Raises:
        ValueError: if the part name is not safe as a file name.
        pandera.errors.SchemaError: if the rows break the schema, such as a repeated work and
            sentence number; nothing is written.
    """
    validated: pl.DataFrame = SENTENCE_VIEW_SCHEMA.validate(view)
    target = part_path(catalog.sentence_view.path, part, _SUFFIX)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(f"{_SUFFIX}.partial")
    validated.write_parquet(staging)
    staging.replace(target)
    return target


def load_sentence_view(catalog: CatalogConfig) -> pl.LazyFrame:
    """Every saved part as one lazy table, so callers stream only what they select.

    Raises:
        FileNotFoundError: if no part has been saved yet.
    """
    parts = sorted(catalog.sentence_view.path.glob(f"*{_SUFFIX}"))
    if not parts:
        raise FileNotFoundError(f"no part of the sentence view under {catalog.sentence_view.path}")
    return pl.scan_parquet(parts, schema=SENTENCE_VIEW_COLUMNS)
