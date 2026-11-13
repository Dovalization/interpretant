"""One row per work, and near-duplicates by MinHash, on synthetic abstracts."""

from __future__ import annotations

from interpretant.catalog import RawWork
from interpretant.corpus.annotation import Removal
from interpretant.corpus.deduplication import (
    NEAR_DUPLICATE,
    build_shingles,
    mark_near_duplicates,
    merge_cells,
)
from tests.abstracts_tables import abstract_row, abstracts_table, dropped_reasons

# A long synthetic abstract, so one changed word leaves most 5-shingles shared.
WORDS = [f"term{index}" for index in range(100)]
ORIGINAL = " ".join(WORDS) + "."
REVISED = " ".join(["revised", *WORDS[1:]]) + "."
UNRELATED = " ".join(f"other{index}" for index in range(100)) + "."


def raw_work(work: int, text: str = "Models learn senses from context.") -> RawWork:
    return RawWork.model_validate(
        {
            "id": f"https://openalex.org/W{work}",
            "year": 2017,
            "type": "article",
            "authors": ["https://openalex.org/A1"],
            "abstract": text,
        }
    )


def test_merge_cells_lists_every_field_of_a_work_once() -> None:
    abstracts = merge_cells(
        [
            ("compsci", [raw_work(1), raw_work(2)]),
            ("ai", [raw_work(1, text="A later copy."), raw_work(3)]),
        ],
        source="regular",
    )
    assert [(abstract.work_id[-2:], abstract.fields) for abstract in abstracts] == [
        ("W1", ("ai", "compsci")),
        ("W2", ("compsci",)),
        ("W3", ("ai",)),
    ]
    assert abstracts[0].text == "Models learn senses from context."


def test_build_shingles_skips_removed_ranges() -> None:
    text = "One two three four five https://example.org six"
    removal = Removal(text.index("https"), text.index(" six"), "url")
    assert build_shingles(text, [removal]) == {
        b"one two three four five",
        b"two three four five six",
    }


def test_build_shingles_makes_one_shingle_of_a_short_text_and_none_of_an_empty_one() -> None:
    assert build_shingles("Two words", []) == {b"two words"}
    assert build_shingles("", []) == set()


def test_mark_near_duplicates_keeps_the_earliest_and_records_it() -> None:
    table = abstracts_table(
        [
            abstract_row(1, [REVISED], year=2019),
            abstract_row(2, [ORIGINAL], year=2015),
            abstract_row(3, [UNRELATED], year=2016),
        ]
    )
    marked = mark_near_duplicates(table, threshold=0.8)
    assert dropped_reasons(marked) == {1: NEAR_DUPLICATE, 2: None, 3: None}
    assert marked["duplicate_of"].to_list() == [table["work_id"][1], None, None]


def test_mark_near_duplicates_ignores_abstracts_already_dropped() -> None:
    table = abstracts_table(
        [
            abstract_row(1, [ORIGINAL], year=2015, dropped_reason="web_scrape"),
            abstract_row(2, [REVISED], year=2019),
        ]
    )
    marked = mark_near_duplicates(table, threshold=0.8)
    assert dropped_reasons(marked) == {1: "web_scrape", 2: None}


def test_mark_near_duplicates_compares_the_kept_text_only() -> None:
    footer = "Copyright notice repeated on every page of this publisher's site today."
    first = abstract_row(1, [ORIGINAL, footer], year=2015)
    second = abstract_row(2, [UNRELATED, footer], year=2016)
    for row in (first, second):
        footer_range = row["sentences"][1]
        row["removals"] = [
            {"start": footer_range["start"], "end": footer_range["end"], "reason": "footer"}
        ]
    marked = mark_near_duplicates(abstracts_table([first, second]), threshold=0.5)
    assert dropped_reasons(marked) == {1: None, 2: None}
