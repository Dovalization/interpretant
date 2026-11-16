"""A text cut at its sentences' edges: the pieces the repairs change, where sentences end up."""

from __future__ import annotations

import pytest

from interpretant.corpus.sentence_pieces import (
    cut_into_pieces,
    pieced_texts,
    repair_pieces,
    sentence_ranges,
)

ENGLISH_ONE = "The study looks at ofthe models."
GERMAN = "Die Studie zeigt ofthe Modelle."
ENGLISH_TWO = "Its infor- mation is new."
TEXT = f"{ENGLISH_ONE} {GERMAN}  {ENGLISH_TWO}"


def sentences(language_two: str = "de") -> list[tuple[int, int, str, float]]:
    starts = [TEXT.index(ENGLISH_ONE), TEXT.index(GERMAN), TEXT.index(ENGLISH_TWO)]
    ends = [
        start + len(sentence)
        for start, sentence in zip(starts, (ENGLISH_ONE, GERMAN, ENGLISH_TWO), strict=True)
    ]
    return [
        (starts[0], ends[0], "en", 0.99),
        (starts[1], ends[1], language_two, 0.9),
        (starts[2], ends[2], "en", 0.98),
    ]


def test_cut_into_pieces_marks_english_sentences_repairable_and_keeps_the_text() -> None:
    pieced = cut_into_pieces("W1", TEXT, sentences())
    assert pieced.text == TEXT
    assert pieced.repairable_pieces() == [ENGLISH_ONE, ENGLISH_TWO]


@pytest.mark.parametrize(
    ("language", "expected"),
    [("und", [ENGLISH_ONE, GERMAN, ENGLISH_TWO]), ("de", [ENGLISH_ONE, ENGLISH_TWO])],
)
def test_cut_into_pieces_repairs_undetermined_sentences_only_in_english_abstracts(
    language: str, expected: list[str]
) -> None:
    assert cut_into_pieces("W1", TEXT, sentences(language)).repairable_pieces() == expected


def test_cut_into_pieces_leaves_a_mostly_foreign_abstract_s_undetermined_sentences() -> None:
    text = "Eine lange deutsche Einleitung zeigt die Studie hier sehr genau. ofthe."
    split = text.index("ofthe")
    pieced = cut_into_pieces(
        "W1", text, [(0, split - 1, "de", 0.9), (split, len(text), "und", 0.5)]
    )
    assert pieced.repairable_pieces() == []


def test_sentence_ranges_follow_the_repairs_and_keep_untouched_sentences_whole() -> None:
    pieced = cut_into_pieces("W1", TEXT, sentences())

    def repair(text: str) -> str:
        return text.replace("ofthe", "of the").replace("infor- mation", "information")

    repaired = repair_pieces(pieced, repair)
    ranges = sentence_ranges(repaired)
    text = repaired.text
    assert [text[start:end] for start, end, _, _ in ranges] == [
        "The study looks at of the models.",
        GERMAN,
        "Its information is new.",
    ]
    assert [language for _, _, language, _ in ranges] == ["en", "de", "en"]


def test_pieced_texts_gives_a_work_without_sentences_one_piece_left_alone() -> None:
    works = pieced_texts(
        [("W1", TEXT), ("W2", "Untouched text.")],
        [
            ("W1", start, end, language, confidence)
            for start, end, language, confidence in sentences()
        ],
    )
    assert [work.work_id for work in works] == ["W1", "W2"]
    assert works[1].pieces == ("Untouched text.",)
    assert works[1].repairable_pieces() == []
