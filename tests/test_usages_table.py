"""The usages table: its schema refuses bad rows, and its rows are checked against the text."""

from __future__ import annotations

from typing import Any

import pandera.errors
import polars as pl
import pytest

from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.usages import USAGES_COLUMNS, load_usages, save_usages
from interpretant.usages.checks import validate_usages_against_text
from interpretant.usages.identifiers import build_content_key, build_draw_key

TEXT = "Intro text. Attention is limited. More text."
WORK = "https://openalex.org/W1"


def usage_row(**changes: Any) -> dict[str, Any]:
    """One valid usage of *attention* in TEXT's second sentence."""
    sentence_start, sentence_end, start = 12, 33, 12
    sentence = TEXT[sentence_start:sentence_end]
    key = build_content_key(sentence, start - sentence_start, start - sentence_start + 9)
    row: dict[str, Any] = {
        "work_id": WORK,
        "word": "attention",
        "content_key": key,
        "draw_key": build_draw_key(1, WORK, "attention", key),
        "sentence_number": 1,
        "sentence_start": sentence_start,
        "sentence_end": sentence_end,
        "target_start": start,
        "target_end": start + 9,
        "compound": None,
        "position": None,
        "pos": "NOUN",
        "tag": "NN",
        "morph": "Number=Sing",
        "lemma": "attention",
        "entity": None,
        "static_token": "attention",
        "sentence_word_count": 3,
        "sentence_language": "en",
        "sentence_confidence": 0.99,
        "sentence_exclusion": None,
        "in_scope": True,
        "drawn": True,
        "model": "en_core_web_sm",
        "windowed": False,
    }
    row.update(changes)
    return row


def table(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """The rows with the usages table's columns."""
    return pl.DataFrame(rows, schema=USAGES_COLUMNS)


def test_save_usages_writes_a_valid_part(catalog: CatalogConfig) -> None:
    save_usages(catalog, table([usage_row()]), part="ai_2010")
    assert load_usages(catalog).collect().height == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"content_key": "xyz"},
        {"target_end": 40},
        {"drawn": True, "in_scope": False},
        {"drawn": True, "sentence_exclusion": "cut_off"},
        {"position": "middle"},
        {"model": "en_core_web_lg"},
    ],
    ids=["key", "span outside", "drawn out of scope", "drawn excluded", "position", "model"],
)
def test_save_usages_refuses_a_bad_row(catalog: CatalogConfig, changes: dict[str, Any]) -> None:
    with pytest.raises(pandera.errors.SchemaError):
        save_usages(catalog, table([usage_row(**changes)]), part="ai_2010")


def test_save_usages_refuses_a_repeated_usage_id(catalog: CatalogConfig) -> None:
    with pytest.raises(pandera.errors.SchemaError):
        save_usages(catalog, table([usage_row(), usage_row(drawn=False)]), part="ai_2010")


def test_validate_usages_against_text_accepts_a_usage_that_points_at_its_target() -> None:
    validate_usages_against_text(table([usage_row()]), {WORK: TEXT})


@pytest.mark.parametrize(
    "changes",
    [
        {"content_key": "0" * 20},
        {"target_start": 13, "target_end": 21},
        {"target_start": 12, "target_end": 20},
        {"target_start": 25, "target_end": 32},
    ],
    ids=["key", "cut at the start", "cut at the end", "another word"],
)
def test_validate_usages_against_text_refuses_a_bad_key_or_span(changes: dict[str, Any]) -> None:
    row = usage_row(**changes)
    if "content_key" not in changes:
        sentence = TEXT[row["sentence_start"] : row["sentence_end"]]
        start = row["target_start"] - row["sentence_start"]
        end = row["target_end"] - row["sentence_start"]
        row["content_key"] = build_content_key(sentence, start, end)
    with pytest.raises(pandera.errors.SchemaError):
        validate_usages_against_text(table([row]), {WORK: TEXT})


def test_validate_usages_against_text_refuses_a_usage_without_text() -> None:
    with pytest.raises(pandera.errors.SchemaError):
        validate_usages_against_text(table([usage_row()]), {})


def test_validate_usages_against_text_accepts_combining_marks_inside_the_target() -> None:
    marked = "Intro text. \u0314Attention\u0315 is limited. More text."
    sentence = marked[12:35]
    row = usage_row(sentence_end=35, target_start=12, target_end=23)
    row["content_key"] = build_content_key(sentence, 0, 11)
    validate_usages_against_text(table([row]), {WORK: marked})
