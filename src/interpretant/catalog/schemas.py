"""The schemas the catalog checks tables against when it writes and reads them."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.polars import PolarsData

SENTENCE = pl.Struct(
    {"start": pl.Int64(), "end": pl.Int64(), "language": pl.String(), "confidence": pl.Float64()}
)
REMOVAL = pl.Struct({"start": pl.Int64(), "end": pl.Int64(), "reason": pl.String()})

ABSTRACTS_COLUMNS = pl.Schema(
    {
        "work_id": pl.String(),
        "year": pl.Int64(),
        "decade": pl.Int64(),
        "fields": pl.List(pl.String()),
        "source": pl.String(),
        "document_type": pl.String(),
        "author_ids": pl.List(pl.String()),
        "text": pl.String(),
        "sentences": pl.List(SENTENCE),
        "removals": pl.List(REMOVAL),
        "dropped_reason": pl.String(),
    }
)


def _spans_are_valid(column: str) -> list[pl.Expr]:
    """Every span starts at or after 0, before its end, and ends inside the text."""
    span = pl.element().struct
    starts_before_ends = (
        pl.col(column)
        .list.eval((span.field("start") >= 0) & (span.field("start") < span.field("end")))
        .list.all()
        .alias(f"{column}_start_before_end")
    )
    last_end = pl.col(column).list.eval(span.field("end")).list.max().fill_null(0)
    ends_inside_text = (last_end <= pl.col("text").str.len_chars()).alias(f"{column}_inside_text")
    return [starts_before_ends, ends_inside_text]


def _decade_is_the_year_rounded_down(data: PolarsData) -> pl.LazyFrame:
    checked: pl.LazyFrame = data.lazyframe.select(
        pl.col("decade") == pl.col("year") - pl.col("year") % 10
    )
    return checked


def _every_work_has_a_field(data: PolarsData) -> pl.LazyFrame:
    checked: pl.LazyFrame = data.lazyframe.select(pl.col("fields").list.len() > 0)
    return checked


def _spans_lie_inside_the_text(data: PolarsData) -> pl.LazyFrame:
    checked: pl.LazyFrame = data.lazyframe.select(
        *_spans_are_valid("sentences"), *_spans_are_valid("removals")
    )
    return checked


ABSTRACTS_SCHEMA = pa.DataFrameSchema(
    {
        "work_id": pa.Column(
            ABSTRACTS_COLUMNS["work_id"],
            unique=True,
            checks=pa.Check.str_startswith("https://openalex.org/W"),
        ),
        "year": pa.Column(ABSTRACTS_COLUMNS["year"], checks=pa.Check.in_range(1900, 2100)),
        "decade": pa.Column(ABSTRACTS_COLUMNS["decade"]),
        "fields": pa.Column(ABSTRACTS_COLUMNS["fields"]),
        "source": pa.Column(
            ABSTRACTS_COLUMNS["source"],
            checks=pa.Check.str_matches(r"^(regular|download:[a-z0-9_-]+)$"),
        ),
        "document_type": pa.Column(ABSTRACTS_COLUMNS["document_type"]),
        "author_ids": pa.Column(ABSTRACTS_COLUMNS["author_ids"]),
        "text": pa.Column(ABSTRACTS_COLUMNS["text"]),
        "sentences": pa.Column(ABSTRACTS_COLUMNS["sentences"]),
        "removals": pa.Column(ABSTRACTS_COLUMNS["removals"]),
        "dropped_reason": pa.Column(ABSTRACTS_COLUMNS["dropped_reason"], nullable=True),
    },
    checks=[
        pa.Check(_decade_is_the_year_rounded_down, name="decade is the year rounded down"),
        pa.Check(_every_work_has_a_field, name="every work has a field"),
        pa.Check(_spans_lie_inside_the_text, name="spans lie inside the text"),
    ],
    strict=True,
    name="abstracts",
    description=(
        "One row per work, its repaired text never edited after repair. Removals are labelled "
        "character ranges into the text (stand-off annotation, decision agenda 2.2); "
        "dropped_reason is set when a keeping rule drops the whole abstract, so dropped "
        "abstracts stay available for the corpus review (4b)."
    ),
)
