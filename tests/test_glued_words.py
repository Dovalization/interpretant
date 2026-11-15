"""Glued words: counted over the texts, decided from the counts, and split text by text."""

from __future__ import annotations

import polars as pl
import pytest

from interpretant.corpus.glued_words import (
    APART,
    KEPT_JOINED,
    SENTENCE_OPENER,
    count_glued_words,
    decide_glued_word_splits,
    glued_word_splits,
    split_glued_words,
)

TEXTS = [
    "AbstractThe study looks at PubMed records.",
    "Abstract The study. The model works. The end.",
    "Abstract: The results hold. We search PubMed.",
    "To the EditorWe read the paper. See www.ClinicalTrials.gov for the data.",
    "Our mRNA and pH data. The PubMed search. In short, LinkedIn helps.",
    "Ties were linked in time. This holds. EditorsThis issue.",
    "Abstract the abstract. To the editor. We agree.",
    "theMacArthur grant. AbstractThe end.",
]


def texts_table(texts: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"work_id": [f"W{work}" for work in range(len(texts))], "text": texts})


def decide(counts: pl.DataFrame, apart_margin: float = 2) -> pl.DataFrame:
    return decide_glued_word_splits(
        counts,
        apart_margin=apart_margin,
        opener_count=5,
        opener_min_left_letters=4,
        kept_joined=["linkedin"],
    )


def test_count_glued_words_counts_each_word_joined_apart_and_alone() -> None:
    counts = count_glued_words(texts_table(TEXTS))
    row = counts.filter(pl.col("word") == "AbstractThe").row(0, named=True)
    assert (row["left"], row["right"], row["works"], row["occurrences"]) == (
        "Abstract",
        "The",
        2,
        2,
    )
    # Apart: "Abstract The", "Abstract: The" and "Abstract the", whatever the punctuation or case.
    assert (row["joined"], row["apart"], row["left_alone"]) == (2, 3, 4)
    assert row["opener_rank"] == 1
    pubmed = counts.filter(pl.col("word") == "PubMed").row(0, named=True)
    assert (pubmed["joined"], pubmed["apart"], pubmed["opener_rank"]) == (3, 0, None)


def test_count_glued_words_leaves_out_one_letter_prefixes_and_several_boundaries() -> None:
    words = set(count_glued_words(texts_table(TEXTS))["word"])
    assert {"mRNA", "pH", "theMacArthur"}.isdisjoint(words)


def test_count_glued_words_gives_an_empty_table_for_texts_without_glued_words() -> None:
    counts = count_glued_words(texts_table(["Plain text with no joins at all."]))
    assert counts.is_empty()
    assert decide(counts).filter(pl.col("split")).is_empty()


def test_decide_glued_word_splits_gives_each_word_its_reason() -> None:
    reasons = dict(decide(count_glued_words(texts_table(TEXTS))).select("word", "reason").rows())
    assert reasons["EditorWe"] == SENTENCE_OPENER
    assert reasons["EditorsThis"] == SENTENCE_OPENER
    assert reasons["LinkedIn"] == KEPT_JOINED
    assert reasons["PubMed"] is None
    assert reasons["ClinicalTrials"] is None


def test_decide_glued_word_splits_splits_a_word_written_apart_by_the_margin() -> None:
    counts = count_glued_words(texts_table(TEXTS))
    assert decide(counts, apart_margin=1.5).filter(pl.col("word") == "AbstractThe")[
        "reason"
    ].to_list() == [APART]
    assert decide(counts, apart_margin=2).filter(pl.col("word") == "AbstractThe")[
        "reason"
    ].to_list() == [SENTENCE_OPENER]


def test_decide_glued_word_splits_needs_a_long_first_part_for_an_opener() -> None:
    counts = count_glued_words(texts_table(["The paper. ofThe paper. The end."]))
    assert decide(counts, apart_margin=100)["split"].to_list() == [False]


def test_glued_word_splits_maps_each_split_word_to_its_parts() -> None:
    splits = glued_word_splits(decide(count_glued_words(texts_table(TEXTS))))
    assert splits["EditorWe"] == "Editor We"
    assert "LinkedIn" not in splits
    assert "PubMed" not in splits


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("To the EditorWe read it.", "To the Editor We read it."),
        ("See www.EditorWe.org today.", "See www.EditorWe.org today."),
        ("The EditorWeekly is a journal.", "The EditorWeekly is a journal."),
        ("Write to someone@EditorWe.org now.", "Write to someone@EditorWe.org now."),
        ("We searched PubMed.", "We searched PubMed."),
    ],
    ids=["a listed word", "inside a web address", "a longer word", "inside an e-mail", "unlisted"],
)
def test_split_glued_words_replaces_whole_listed_words_outside_addresses(
    text: str, expected: str
) -> None:
    assert split_glued_words(text, {"EditorWe": "Editor We"}) == expected


def test_split_glued_words_leaves_the_text_alone_without_splits() -> None:
    assert split_glued_words("To the EditorWe read it.", {}) == "To the EditorWe read it."
