"""Rows of the abstracts table for tests, built from their sentences."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl

from interpretant.catalog import ABSTRACTS_COLUMNS


def abstract_row(work: int, sentences: Sequence[str], /, **changes: Any) -> dict[str, Any]:
    """One English abstract whose text is its sentences joined by spaces, with their offsets."""
    starts = [
        sum(len(sentence) + 1 for sentence in sentences[:index]) for index in range(len(sentences))
    ]
    year = changes.get("year", 2017)
    row: dict[str, Any] = {
        "work_id": f"https://openalex.org/W{work}",
        "year": year,
        "decade": year - year % 10,
        "fields": ["ai"],
        "source": "regular",
        "document_type": "article",
        "author_ids": ["https://openalex.org/A1"],
        "text": " ".join(sentences),
        "sentences": [
            {"start": start, "end": start + len(sentence), "language": "en", "confidence": 0.99}
            for start, sentence in zip(starts, sentences, strict=True)
        ],
        "removals": [],
        "flags": [],
        "dropped_reason": None,
        "duplicate_of": None,
    }
    row.update(changes)
    return row


def abstracts_table(rows: Sequence[dict[str, Any]]) -> pl.DataFrame:
    """The rows as a table with the abstracts table's columns and types."""
    return pl.DataFrame(rows, schema=ABSTRACTS_COLUMNS)


def removals_of(table: pl.DataFrame, work: int) -> list[tuple[str, str]]:
    """One work's removals as (removed text, reason) pairs, in stored order."""
    row = table.filter(pl.col("work_id") == f"https://openalex.org/W{work}").row(0, named=True)
    return [
        (row["text"][removal["start"] : removal["end"]], removal["reason"])
        for removal in row["removals"]
    ]


def dropped_reasons(table: pl.DataFrame) -> dict[int, str | None]:
    """Each work's dropped reason, by the work's number."""
    return {
        int(work_id.removeprefix("https://openalex.org/W")): reason
        for work_id, reason in zip(table["work_id"], table["dropped_reason"], strict=True)
    }
