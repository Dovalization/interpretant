"""The contextual sentence view: which cuts are joined, which sentences are removed or cut off."""

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
    join_rule,
    merge_ranges,
    missed_breaks,
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
    assert (result.work_count, result.sentence_count, result.joined_count) == (2, 4, 0)
    assert result.split_count == 0
    assert (result.removed_count, result.cut_off_count) == (1, 1)
    assert result.parts == ["ai_1970", "ai_2010"]
    written = load_sentence_view(catalog).sort("work_id", "sentence_number").collect()
    assert written["exclusion"].to_list() == [None, "removed", None, "cut_off"]


@pytest.mark.parametrize(
    ("before", "after", "rule"),
    [
        (
            "Notes [Landshut 1809][Schelling, p. 8",
            ': "The absolute can be given',
            "closing_punctuation",
        ),
        ("2", ". Subjects using booklets did better.", "no_letters"),
        (
            "They differ in how they are learned 2",
            ". Subjects using booklets did better.",
            "period",
        ),
        ("Sociological theory . .", "is for us that aspect of the theory", "spaced_dots"),
        (
            "The hypotheses to be tested were: 1.",
            "Subjects using booklets did better.",
            "list_number",
        ),
        (
            "Moreover, n-2 is the best possible;(2)",
            "If G is connected, then it holds.",
            "list_number",
        ),
        (
            "The Still Life as a Personal Object-A",
            "Note on Heidegger and van Gogh.",
            "function_word",
        ),
        ("(Cambridge: Cambridge Univ.", "Press, 1970).", "abbreviation"),
        (
            "tongue movements (Kier WM and Smith KK.",
            "Zool J Linn Soc 83: 207-324, 1985).",
            "inside_brackets",
        ),
        ("Genealogy of Morality and Other Writings, ed.", "Keith Ansell-Pearson.", "abbreviation"),
        ("These were studied in many ways, e.g.", "In the first two examples.", "abbreviation"),
        ("This is discussed in ch.", "2 (Phonology) at length.", "reference_number"),
        ("PROCESSING IN ORGANIZATIONS, VOL.", "I. Edited by Lee S. Sproull.", "reference_number"),
        ("University School of Medicine-St.", "Louis, Missouri is proud.", "saint"),
        ("The section ends here.", "The next one starts.", None),
        ("Teaching methods, materials, etc.", "Teaching students well matters.", None),
        ("As described by Smith et al.", "We then tested the model.", None),
        ("Menlo Park, CA.", "Address reprint requests to the author.", None),
        ("It ideologies, (1941:84)", "Mannheim envisaged all ideologies.", None),
        ("The edges cross in p.", "In the CROSSING number problem, it is given.", None),
        ("Industries have lost their huge leads. .", "A debilitating acceptance followed.", None),
        ("A list of results follows.", "(b) What types of feedback were given?", None),
        ("Rushby [1993] . .", ".Desire for correctness proofs spawned research.", None),
        ("(as in the study of resistance to intimacy.", "We need to be careful.)", None),
        ("Mastication in patients with ST.", "Likewise, no data are available.", None),
        ("(the transfer is shown, eg.", "Ketterson and Nolan 1992)", "abbreviation"),
    ],
)
def test_join_rule_finds_the_cuts_where_no_sentence_ends(
    before: str, after: str, rule: str | None
) -> None:
    assert join_rule(before, after) == rule


def test_build_sentence_view_joins_wrong_cuts_and_records_their_rules() -> None:
    sentences = ["The hypotheses were: 1.", "Booklets help.", "See ch.", "2 for the details.", "5"]
    view = build_sentence_view(abstracts_table([abstract_row(1, sentences)]))
    text = " ".join(sentences)
    assert [text[start:end] for start, end in view.select("start", "end").rows()] == [
        "The hypotheses were: 1. Booklets help.",
        "See ch. 2 for the details. 5",
    ]
    assert view["merge_rules"].to_list() == [["list_number"], ["reference_number", "no_letters"]]
    assert view["sentence_number"].to_list() == [0, 1]


def test_build_sentence_view_never_joins_a_removed_sentence() -> None:
    sentences = ["The hypotheses were: 1.", "Booklets help a lot.", "And more."]
    second = len(sentences[0]) + 1
    row = abstract_row(1, sentences, removals=[removal(second, second + len(sentences[1]))])
    view = build_sentence_view(abstracts_table([row]))
    assert view.select("exclusion", "merge_rules").rows() == [
        (None, []),
        ("removed", []),
        (None, []),
    ]


def test_build_sentence_view_gives_a_joined_sentence_its_longest_parts_language() -> None:
    row = abstract_row(
        1, ["See p.", "12 for the full and very long details of it."], flags=["truncated"]
    )
    row["sentences"][0] |= {"language": "und", "confidence": 0.1}
    view = build_sentence_view(abstracts_table([row]))
    assert view.select("language", "confidence", "exclusion").rows() == [("en", 0.99, "cut_off")]


def test_build_sentence_view_keeps_a_removed_list_number_with_its_item() -> None:
    sentences = ["Two hypotheses were: 1.", "Booklets help.", "2", ". Drills help.", "Then more."]
    third = len(sentences[0]) + len(sentences[1]) + 2
    row = abstract_row(1, sentences, removals=[removal(third, third + 1, "template_sentence")])
    view = build_sentence_view(abstracts_table([row]))
    text = " ".join(sentences)
    assert [text[start:end] for start, end in view.select("start", "end").rows()] == [
        "Two hypotheses were: 1. Booklets help.",
        "2 . Drills help.",
        "Then more.",
    ]
    assert view.select("exclusion", "merge_rules").rows() == [
        (None, ["list_number"]),
        (None, ["no_letters"]),
        (None, []),
    ]


def test_build_sentence_view_joins_a_removed_stray_period_back_to_its_ellipsis() -> None:
    sentences = ["It may be a looking glasse. .", ".", ".*-The", "Mutable estate of France."]
    second = len(sentences[0]) + 1
    row = abstract_row(1, sentences, removals=[removal(second, second + 1, "template_sentence")])
    view = build_sentence_view(abstracts_table([row]))
    assert view.height == 1
    assert view["merge_rules"].to_list() == [["no_letters", "period", "function_word"]]


@pytest.mark.parametrize(
    ("sentence", "pieces"),
    [
        (
            "Streams run over a VP. Also, we propose two schemes.",
            ["Streams run over a VP.", "Also, we propose two schemes."],
        ),
        (
            "It was sensed in 3D. In addition, a circuit converts it.",
            ["It was sensed in 3D.", "In addition, a circuit converts it."],
        ),
        (
            "A portrait by artist George P. A. Healy hangs here.",
            ["A portrait by artist George P. A. Healy hangs here."],
        ),
        (
            "For a review, see E. A. Goerner on natural law.",
            ["For a review, see E. A. Goerner on natural law."],
        ),
        ("I. The Man and his background.", ["I. The Man and his background."]),
        (
            "Studies by Werner, H. A. Witkin and others.",
            ["Studies by Werner, H. A. Witkin and others."],
        ),
    ],
)
def test_missed_breaks_cut_after_a_capital_before_an_opener(
    sentence: str, pieces: list[str]
) -> None:
    cuts = missed_breaks(sentence)
    starts = [0] + [after for _, after in cuts]
    ends = [before for before, _ in cuts] + [len(sentence)]
    assert [sentence[low:high] for low, high in zip(starts, ends, strict=True)] == pieces


def test_build_sentence_view_splits_and_keeps_joins_with_their_piece() -> None:
    sentences = ["See ch.", "2 for streams over a VP. Also, we propose more", "e.g."]
    row = abstract_row(1, sentences[:2], flags=["truncated"])
    view = build_sentence_view(abstracts_table([row]))
    text = " ".join(sentences[:2])
    assert [text[low:high] for low, high in view.select("start", "end").rows()] == [
        "See ch. 2 for streams over a VP.",
        "Also, we propose more",
    ]
    assert view.select("merge_rules", "split", "exclusion").rows() == [
        (["reference_number"], False, None),
        ([], True, "cut_off"),
    ]
