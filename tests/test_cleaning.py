"""The corpus-level rules on small synthetic abstracts tables."""

from __future__ import annotations

import hashlib

import polars as pl
import pytest

from interpretant.corpus.cleaning import (
    AUTHOR_CAP,
    REPEATED_SENTENCE,
    TEMPLATE,
    TEMPLATE_SENTENCE,
    mark_author_cap,
    mark_repeated_sentences,
    mark_template_sentences,
)
from tests.abstracts_tables import abstract_row, abstracts_table, dropped_reasons, removals_of

BODY = "We measure how attention shifts across fields over several decades of research."


def distinct_body(work: int) -> str:
    """A sentence no other work shares, even with digits masked."""
    return f"We measure how attention shifts {'again ' * work}across fields."


def template_table(sharing: int, *, dropped: int = 0) -> pl.DataFrame:
    """Abstracts each ending in the same notice with a different year, some already dropped."""
    rows = [
        abstract_row(
            work,
            [distinct_body(work), f"Published under licence in {1990 + work} by the authors."],
            dropped_reason="web_scrape" if work <= dropped else None,
        )
        for work in range(1, sharing + 1)
    ]
    return abstracts_table(rows)


def test_mark_template_sentences_removes_a_sentence_shared_with_digits_masked() -> None:
    marked = mark_template_sentences(template_table(20), min_abstracts=20)
    assert removals_of(marked, 7) == [
        ("Published under licence in 1997 by the authors.", TEMPLATE_SENTENCE)
    ]


def test_mark_template_sentences_ignores_sentences_below_the_count() -> None:
    marked = mark_template_sentences(template_table(19), min_abstracts=20)
    assert removals_of(marked, 7) == []


def test_mark_template_sentences_does_not_count_dropped_abstracts() -> None:
    marked = mark_template_sentences(template_table(20, dropped=1), min_abstracts=20)
    assert removals_of(marked, 7) == []


def test_mark_template_sentences_drops_an_abstract_made_mostly_of_templates() -> None:
    rows = [
        abstract_row(
            work, [f"Body {'x' * work}.", "All rights to this text are held by the press."]
        )
        for work in range(1, 21)
    ]
    marked = mark_template_sentences(abstracts_table(rows), min_abstracts=20)
    assert set(dropped_reasons(marked).values()) == {TEMPLATE}
    assert template_table(20)["dropped_reason"].null_count() == 20


def test_mark_repeated_sentences_keeps_the_first_copy_only() -> None:
    table = abstracts_table([abstract_row(1, [BODY, "Café results hold.", BODY])])
    marked = mark_repeated_sentences(table)
    assert removals_of(marked, 1) == [(BODY, REPEATED_SENTENCE)]
    removal = marked["removals"][0][0]
    assert removal["start"] == len(BODY) + len(" Café results hold. ")


def test_mark_repeated_sentences_inserts_in_order_and_skips_ranges_already_removed() -> None:
    row = abstract_row(1, ["See the data now.", BODY, BODY, BODY])
    third_copy = row["sentences"][3]
    row["removals"] = [
        {"start": 4, "end": 8, "reason": "url"},
        {"start": third_copy["start"], "end": third_copy["end"], "reason": "footer"},
    ]
    marked = mark_repeated_sentences(abstracts_table([row]))
    assert [reason for _, reason in removals_of(marked, 1)] == [
        "url",
        REPEATED_SENTENCE,
        "footer",
    ]


def stable_order(works: list[int]) -> list[int]:
    return sorted(
        works,
        key=lambda work: hashlib.sha256(f"https://openalex.org/W{work}".encode()).hexdigest(),
    )


def test_mark_author_cap_keeps_the_floor_for_an_author_in_a_small_cell() -> None:
    prolific = "https://openalex.org/A9"
    rows = [
        abstract_row(
            work, [BODY], author_ids=[prolific] if work <= 8 else ["https://openalex.org/A1"]
        )
        for work in range(1, 11)
    ]
    marked = mark_author_cap(abstracts_table(rows), max_share=0.01, floor=5)
    capped = [work for work, reason in dropped_reasons(marked).items() if reason == AUTHOR_CAP]
    assert sorted(capped) == sorted(stable_order(list(range(1, 9)))[5:])


def test_mark_author_cap_uses_the_share_in_a_large_cell() -> None:
    prolific = "https://openalex.org/A9"
    rows = [
        abstract_row(
            work, [BODY], author_ids=[prolific if work <= 30 else f"https://openalex.org/A{work}"]
        )
        for work in range(1, 2001)
    ]
    marked = mark_author_cap(abstracts_table(rows), max_share=0.01, floor=5)
    assert list(dropped_reasons(marked).values()).count(AUTHOR_CAP) == 30 - 20


def test_mark_author_cap_counts_each_cell_of_a_work_in_two_fields() -> None:
    prolific = "https://openalex.org/A9"
    rows = [
        abstract_row(work, [BODY], author_ids=[prolific], fields=["ai", "compsci"])
        for work in range(1, 7)
    ]
    marked = mark_author_cap(abstracts_table(rows), max_share=0.01, floor=5)
    assert list(dropped_reasons(marked).values()).count(AUTHOR_CAP) == 1


def test_mark_author_cap_caps_each_cell_on_all_its_works() -> None:
    """No cell's cap depends on another cell's drops, so the order of the cells cannot matter."""
    prolific = "https://openalex.org/A9"
    fields_by_work = {
        **dict.fromkeys(range(1, 6), ("ai", "compsci")),
        **dict.fromkeys(range(6, 10), ("ai",)),
        **dict.fromkeys(range(10, 13), ("compsci",)),
    }
    rows = [
        abstract_row(work, [BODY], author_ids=[prolific], fields=list(fields))
        for work, fields in fields_by_work.items()
    ]
    marked = mark_author_cap(abstracts_table(rows), max_share=0.01, floor=5)
    capped = {work for work, reason in dropped_reasons(marked).items() if reason == AUTHOR_CAP}
    ai_cell = set(stable_order(list(range(1, 10)))[5:])
    compsci_cell = set(stable_order([*range(1, 6), *range(10, 13)])[5:])
    # Capped one after the other, compsci would count only the works ai kept and miss one.
    assert capped == ai_cell | compsci_cell


@pytest.mark.parametrize(
    "rule",
    [
        lambda table: mark_template_sentences(table, min_abstracts=20),
        mark_repeated_sentences,
        lambda table: mark_author_cap(table, max_share=0.01, floor=5),
    ],
    ids=["templates", "repeated sentences", "author cap"],
)
def test_corpus_rules_keep_the_rows_in_order(rule: object) -> None:
    table = template_table(25)
    assert callable(rule)
    assert rule(table)["work_id"].to_list() == table["work_id"].to_list()
