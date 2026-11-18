"""The contextual sentence view: which sentences are removed or cut off, and the stage."""

from __future__ import annotations

import pandera.errors
import polars as pl
import pytest

from interpretant.catalog.abstracts import save_abstracts
from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.sentence_view import (
    SENTENCE_VIEW_COLUMNS,
    load_sentence_view,
    save_sentence_view,
)
from interpretant.corpus.sentence_view import (
    build_sentence_view,
    is_inside_removal,
    merge_ranges,
    word_count,
)
from interpretant.pipelines.build_sentence_view import build_sentence_view as build_stage
from tests.abstracts_tables import abstract_row, abstracts_table

SENTENCES = ["First sentence here.", "Second one follows.", "Third and last"]


def removal(start: int, end: int, reason: str = "citation_line") -> dict[str, object]:
    """A removed range of the abstracts table."""
    return {"start": start, "end": end, "reason": reason}


def test_merge_ranges_joins_overlapping_and_touching_ranges() -> None:
    assert merge_ranges([(10, 20), (0, 5), (5, 8), (15, 25), (30, 31)]) == [
        (0, 8),
        (10, 25),
        (30, 31),
    ]


@pytest.mark.parametrize(
    ("ranges", "expected"),
    [
        ([(0, 20)], True),
        ([(0, 9), (10, 20)], True),
        ([(0, 8), (10, 20)], False),
        ([(0, 15)], False),
        ([(5, 20)], False),
        ([], False),
    ],
    ids=[
        "one range",
        "two ranges and a space",
        "a letter between",
        "end left",
        "start left",
        "none",
    ],
)
def test_is_inside_removal_counts_only_non_space_characters(
    ranges: list[tuple[int, int]], *, expected: bool
) -> None:
    text = "aaaaaaaaa bbbbbbbbbb"
    assert is_inside_removal(text, 0, 20, merge_ranges(ranges)) is expected


def test_is_inside_removal_needs_a_character_to_remove() -> None:
    assert not is_inside_removal("a    b", 1, 5, [(0, 6)])


def test_word_count_counts_pieces_with_a_letter() -> None:
    assert word_count("The 3 models (2019) reach 95% on GLUE-2.") == 5


def test_build_sentence_view_marks_removed_and_cut_off_sentences() -> None:
    second = len(SENTENCES[0]) + 1
    table = abstracts_table(
        [
            abstract_row(
                1,
                SENTENCES,
                removals=[removal(second, second + 6), removal(second + 7, second + 19)],
                flags=["truncated"],
            ),
            abstract_row(2, SENTENCES, removals=[removal(0, 5)]),
            abstract_row(3, SENTENCES, dropped_reason="template"),
        ]
    )
    view = build_sentence_view(table)
    assert view.schema == SENTENCE_VIEW_COLUMNS
    assert view.select("work_id", "sentence_number", "exclusion").rows() == [
        ("https://openalex.org/W1", 0, None),
        ("https://openalex.org/W1", 1, "removed"),
        ("https://openalex.org/W1", 2, "cut_off"),
        ("https://openalex.org/W2", 0, None),
        ("https://openalex.org/W2", 1, None),
        ("https://openalex.org/W2", 2, None),
    ]
    assert view["word_count"].to_list()[:3] == [3, 3, 3]


def test_build_sentence_view_keeps_a_removed_last_sentence_removed() -> None:
    last = len(SENTENCES[0]) + len(SENTENCES[1]) + 2
    table = abstracts_table(
        [abstract_row(1, SENTENCES, removals=[removal(last, last + 14)], flags=["truncated"])]
    )
    assert build_sentence_view(table)["exclusion"].to_list() == [None, None, "removed"]


def test_build_sentence_view_keeps_other_languages_and_records_them() -> None:
    row = abstract_row(1, SENTENCES[:2])
    row["sentences"][1] |= {"language": "und", "confidence": 0.0}
    view = build_sentence_view(abstracts_table([row]))
    assert view.select("language", "confidence", "exclusion").rows() == [
        ("en", 0.99, None),
        ("und", 0.0, None),
    ]


def test_save_sentence_view_refuses_a_repeated_sentence(catalog: CatalogConfig) -> None:
    view = build_sentence_view(abstracts_table([abstract_row(1, SENTENCES)]))
    repeated = pl.concat([view, view.head(1)])
    with pytest.raises(pandera.errors.SchemaError):
        save_sentence_view(catalog, repeated, part="ai_2010")
    assert not catalog.sentence_view.path.exists()


def test_save_sentence_view_refuses_an_unknown_exclusion(catalog: CatalogConfig) -> None:
    view = build_sentence_view(abstracts_table([abstract_row(1, SENTENCES)]))
    with pytest.raises(pandera.errors.SchemaError):
        save_sentence_view(catalog, view.with_columns(exclusion=pl.lit("junk")), part="ai_2010")


def test_the_stage_writes_every_part_and_counts_its_sentences(catalog: CatalogConfig) -> None:
    second = len(SENTENCES[0]) + 1
    save_abstracts(
        catalog,
        abstracts_table(
            [
                abstract_row(1, SENTENCES, removals=[removal(second, second + 19)]),
                abstract_row(2, SENTENCES, dropped_reason="template"),
            ]
        ),
        part="ai_2010",
    )
    save_abstracts(
        catalog,
        abstracts_table([abstract_row(3, SENTENCES[:1], year=1975, flags=["truncated"])]),
        part="ai_1970",
    )
    result = build_stage(catalog)
    assert (result.work_count, result.sentence_count) == (2, 4)
    assert (result.removed_count, result.cut_off_count) == (1, 1)
    assert result.parts == ["ai_1970", "ai_2010"]
    written = load_sentence_view(catalog).sort("work_id", "sentence_number").collect()
    assert written["exclusion"].to_list() == [None, "removed", None, "cut_off"]
