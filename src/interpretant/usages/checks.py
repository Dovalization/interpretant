"""The usages table checked against the repaired text (decision agenda 4.5).

A usage points into the abstracts table instead of copying text, so agenda 4.5 asks for two checks
the table's own schema cannot make: the content key is recomputable from the sentence and the
target's offsets in it, and the offsets point at the target. Here the second means the span is a
whole word or hyphen part (letters, hyphens, apostrophes and spaces only, no letter or digit just
outside it) whose first letter is the word's. The stage validates every part against this schema,
joined with its works' texts, before writing it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandera.polars as pa
import polars as pl
from pandera.polars import PolarsData

from interpretant.corpus.static_view import fold
from interpretant.usages.identifiers import build_content_key

_SPAN_CHARACTERS = frozenset("-' ")
_OFFSET_COLUMNS = ("text", "sentence_start", "sentence_end", "target_start", "target_end")


def _key_is_recomputable(row: dict[str, Any]) -> bool:
    sentence = row["text"][row["sentence_start"] : row["sentence_end"]]
    start = row["target_start"] - row["sentence_start"]
    end = row["target_end"] - row["sentence_start"]
    return bool(build_content_key(sentence, start, end) == row["content_key"])


def _span_is_the_target(row: dict[str, Any]) -> bool:
    text, start, end = row["text"], row["target_start"], row["target_end"]
    span = text[start:end]
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    return (
        bool(span.strip())
        and all(character.isalpha() or character in _SPAN_CHARACTERS for character in span)
        and not before.isalnum()
        and not after.isalnum()
        and fold(span)[:1] == row["word"][:1]
    )


def _keys_are_recomputable(data: PolarsData) -> pl.LazyFrame:
    checked: pl.LazyFrame = data.lazyframe.select(
        pl.struct(*_OFFSET_COLUMNS, "content_key").map_elements(
            _key_is_recomputable, return_dtype=pl.Boolean
        )
    )
    return checked


def _offsets_point_at_the_target(data: PolarsData) -> pl.LazyFrame:
    checked: pl.LazyFrame = data.lazyframe.select(
        pl.struct(*_OFFSET_COLUMNS, "word").map_elements(
            _span_is_the_target, return_dtype=pl.Boolean
        )
    )
    return checked


USAGES_TEXT_SCHEMA = pa.DataFrameSchema(
    {
        "work_id": pa.Column(pl.String),
        "word": pa.Column(pl.String),
        "content_key": pa.Column(pl.String),
        "text": pa.Column(pl.String),
    },
    checks=[
        pa.Check(_keys_are_recomputable, name="the content key is recomputable from the text"),
        pa.Check(_offsets_point_at_the_target, name="the offsets point at the target"),
    ],
    name="usages against the text",
    description="The usages table joined with its works' texts (decision agenda 4.5).",
)


def validate_usages_against_text(usages: pl.DataFrame, texts: Mapping[str, str]) -> None:
    """Check every usage's content key and target span against its work's text.

    Raises:
        pandera.errors.SchemaError: if a usage's work has no text, its key does not recompute, or
            its offsets do not point at the target.
    """
    frame = pl.DataFrame(
        {"work_id": list(texts), "text": list(texts.values())},
        schema={"work_id": pl.String, "text": pl.String},
    )
    USAGES_TEXT_SCHEMA.validate(usages.join(frame, on="work_id", how="left"))
