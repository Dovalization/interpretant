"""Lower-case glued words (run 8): the candidates counted, the decisions and the split."""

from __future__ import annotations

import polars as pl
import pytest

from interpretant.corpus.lower_glued_words import (
    A_AND_WORD,
    CAPITALISED,
    HYPHENATED_CHUNK,
    THREE_OR_MORE_WORDS,
    TWO_WORDS,
    LowerGluedSplits,
    count_lower_glued_words,
    decide_hyphenated_chunks,
    decide_lower_glued_word_splits,
    lower_glued_word_splits,
    split_lower_glued_words,
)

# Sentences written cleanly many times, so the parts are counted words and their pairs apart.
CLEAN = [
    "The results of the study show the state of the art and real-time models in practice.",
    "We report a substantial effect of the proposed model on the state-of-the-art system.",
    "The real-time results of the study hold, and we report them abroad.",
] * 30
GLUED = [
    "Ofthe results, theresults ofthe asubstantial oftheproposed model show realtime gains.",
    "Thestudy used stateof-the-art work and \\ofthe macro in 1ofthe tests at www.ofthe.org today.",
    "A single ofthe here, and Thestudy stays whole.",
]


def decisions_for(texts: list[str], kept_joined: tuple[str, ...] = ()) -> pl.DataFrame:
    frame = pl.DataFrame({"text": texts})
    counts = count_lower_glued_words(frame, min_part_count=1, min_first_part_letters=2)
    decided = decide_lower_glued_word_splits(counts, apart_margin=10, kept_joined=kept_joined)
    return pl.concat([decided, decide_hyphenated_chunks(frame, decided)])


def splits_for(texts: list[str]) -> LowerGluedSplits:
    return lower_glued_word_splits(decisions_for(texts))


def test_count_lower_glued_words_finds_each_pattern_and_leaves_known_words_out() -> None:
    table = decisions_for(CLEAN + GLUED)
    patterns = dict(table.select("word", "pattern").iter_rows())
    assert patterns["theresults"] == TWO_WORDS
    assert patterns["asubstantial"] == A_AND_WORD
    assert patterns["oftheproposed"] == THREE_OR_MORE_WORDS
    assert patterns["stateof-the-art"] == HYPHENATED_CHUNK
    assert "abroad" not in patterns
    uses = {
        row["word"]: (row["uses"], row["capitalised_uses"]) for row in table.iter_rows(named=True)
    }
    assert uses["thestudy"] == (0, 2)


def test_count_lower_glued_words_takes_the_separator_the_corpus_uses_more() -> None:
    separators = dict(decisions_for(CLEAN + GLUED).select("word", "separators").iter_rows())
    assert separators["realtime"] == ["-"]
    assert separators["theresults"] == [" "]


def test_decide_lower_glued_word_splits_keeps_listed_words_joined() -> None:
    table = decisions_for(CLEAN + GLUED, kept_joined=("TheResults",))
    row = table.filter(pl.col("word") == "theresults").row(0, named=True)
    assert (row["kept_joined"], row["split"]) == (True, False)


def test_decide_lower_glued_word_splits_needs_the_margin() -> None:
    rare = ["The results of the study hold."] * 5 + ["theresults"] * 2
    table = decisions_for(rare)
    assert not table.filter(pl.col("word") == "theresults")["split"].item()


def test_split_lower_glued_words_splits_every_pattern_in_a_damaged_text() -> None:
    text, counts = split_lower_glued_words(GLUED[0], splits_for(CLEAN + GLUED), min_dense_splits=2)
    expected = (
        "Of the results, the results of the a substantial of the proposed model show real-time"
    )
    assert text == expected + " gains."
    assert counts == {TWO_WORDS: 3, CAPITALISED: 1, A_AND_WORD: 1, THREE_OR_MORE_WORDS: 1}


def test_split_lower_glued_words_spares_guarded_patterns_outside_damaged_texts() -> None:
    text, counts = split_lower_glued_words(GLUED[2], splits_for(CLEAN + GLUED), min_dense_splits=2)
    assert text == "A single of the here, and Thestudy stays whole."
    assert counts == {TWO_WORDS: 1}


@pytest.mark.parametrize(
    ("glued", "kept"),
    [("\\ofthe", "\\ofthe"), ("1ofthe", "1ofthe"), ("www.ofthe.org", "www.ofthe.org")],
    ids=["after a backslash", "touching a digit", "inside a web address"],
)
def test_split_lower_glued_words_leaves_some_places_alone(glued: str, kept: str) -> None:
    text, _ = split_lower_glued_words(
        f"See {glued} now.", splits_for(CLEAN + GLUED), min_dense_splits=2
    )
    assert text == f"See {kept} now."


def test_split_lower_glued_words_takes_a_hyphenated_chunk_s_separator() -> None:
    text, _ = split_lower_glued_words(GLUED[1], splits_for(CLEAN + GLUED), min_dense_splits=2)
    assert "state-of-the-art work" in text
