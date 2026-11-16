"""Split words: counted over the texts, decided from the counts, and joined text by text."""

from __future__ import annotations

import polars as pl
import pytest

from interpretant.corpus.dictionary import is_dictionary_word
from interpretant.corpus.split_words import (
    SPLIT_WORD_COUNTS_SCHEMA,
    SplitWordJoins,
    count_split_words,
    decide_split_word_joins,
    join_split_words,
    split_word_joins,
)

# "information" whole twenty times and broken once each way; "avoid" whole and "a void" apart
# (both pieces words); "multimodal" whole and often "multi-modal" (a real hyphenated word).
TEXTS = (
    ["The information flows."] * 20
    + ["The infor- mation flows.", "The infor-mation flows.", "The infor mation flows."]
    + ["We avoid it."] * 5
    + ["A void opens."]
    + ["A multimodal model."] * 8
    + ["A multi-modal model."] * 4
)


def texts_table(texts: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"work_id": [f"W{work}" for work in range(len(texts))], "text": texts})


def decide(counts: pl.DataFrame, kept_apart: list[tuple[str, str]] | None = None) -> pl.DataFrame:
    return decide_split_word_joins(
        counts, join_margin=10, hyphen_margin=100, kept_apart=kept_apart or []
    )


def counts_row(
    word: str,
    left: str,
    right: str,
    *,
    joined: int,
    space: int = 0,
    hyphen_space: int = 0,
    hyphen: int = 0,
    right_is_word: bool = False,
) -> dict[str, object]:
    return {
        "word": word,
        "left": left,
        "right": right,
        "left_is_word": False,
        "right_is_word": right_is_word,
        "joined": joined,
        "space": space,
        "hyphen_space": hyphen_space,
        "hyphen": hyphen,
    }


def test_spacy_knows_stop_words_lemmas_and_inflected_forms_but_not_pieces() -> None:
    assert is_dictionary_word("the")
    assert is_dictionary_word("avoid")
    assert is_dictionary_word("Children")
    assert not is_dictionary_word("vestigation")
    assert not is_dictionary_word("mation")


def test_count_split_words_counts_each_break_by_what_separates_the_pieces() -> None:
    counts = count_split_words(texts_table(TEXTS), min_word_count=2)
    row = counts.filter(pl.col("word") == "information").row(0, named=True)
    assert (row["left"], row["right"]) == ("infor", "mation")
    assert (row["left_is_word"], row["right_is_word"]) == (False, False)
    assert (row["joined"], row["space"], row["hyphen_space"], row["hyphen"]) == (20, 1, 1, 1)
    hyphenated = counts.filter(pl.col("word") == "multimodal").row(0, named=True)
    assert (hyphenated["joined"], hyphenated["hyphen"]) == (8, 4)


def test_count_split_words_leaves_out_two_words_spacy_knows() -> None:
    counts = count_split_words(texts_table(TEXTS), min_word_count=2)
    assert "avoid" not in counts["word"].to_list()


def test_count_split_words_leaves_out_words_written_too_rarely() -> None:
    counts = count_split_words(texts_table(TEXTS), min_word_count=21)
    assert counts.is_empty()
    assert counts.schema == SPLIT_WORD_COUNTS_SCHEMA


def test_decide_joins_a_break_written_whole_far_more_often() -> None:
    counts = pl.DataFrame(
        [
            counts_row("information", "infor", "mation", joined=1000, space=1, hyphen=1),
            counts_row(
                "multimodal",
                "multi",
                "modal",
                joined=3971,
                space=17,
                hyphen=667,
                right_is_word=True,
            ),
        ],
        schema=SPLIT_WORD_COUNTS_SCHEMA,
    )
    decisions = decide(counts)
    joins = dict(decisions.select("word", "join").rows())
    assert joins == {"information": True, "multimodal": False}


def test_decide_joins_a_hyphen_alone_only_above_its_margin_with_a_second_piece_spacy_lacks() -> (
    None
):
    counts = pl.DataFrame(
        [
            counts_row("information", "infor", "mation", joined=1000, hyphen=2),
            counts_row("preschoolers", "pre", "schoolers", joined=609, hyphen=27),
            counts_row("selfesteem", "self", "esteem", joined=5000, hyphen=1, right_is_word=True),
            counts_row("xray", "x", "ray", joined=500, hyphen=1),
        ],
        schema=SPLIT_WORD_COUNTS_SCHEMA,
    )
    decisions = decide(counts)
    hyphenated = dict(decisions.select("word", "join_hyphenated").rows())
    assert hyphenated == {
        "information": True,
        "preschoolers": False,
        "selfesteem": False,
        "xray": False,
    }


def test_decide_keeps_a_listed_phrase_apart_whatever_the_counts() -> None:
    counts = pl.DataFrame(
        [counts_row("aphasic", "a", "phasic", joined=555, space=33)],
        schema=SPLIT_WORD_COUNTS_SCHEMA,
    )
    row = decide(counts, kept_apart=[("A", "Phasic")]).row(0, named=True)
    assert row["kept_apart"]
    assert not (row["join"] or row["join_hyphenated"])


def test_split_word_joins_index_the_pieces_by_their_second_piece() -> None:
    counts = pl.DataFrame(
        [
            counts_row("information", "infor", "mation", joined=1000, space=1),
            counts_row("formation", "for", "mation", joined=900, space=1),
        ],
        schema=SPLIT_WORD_COUNTS_SCHEMA,
    )
    joins = split_word_joins(decide(counts))
    assert joins.spaced == {"mation": frozenset({"infor", "for"})}
    assert joins.hyphenated == {"mation": frozenset({"infor", "for"})}


SPACED = SplitWordJoins(spaced={"mation": frozenset({"infor"}), "lowing": frozenset({"fol"})})


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The infor mation flows.", "The information flows."),
        ("The infor- mation flows.", "The information flows."),
        ("See the fol-\nlowing table.", "See the following table."),
        ("Infor mation flows.", "Information flows."),
        ("The infor-mation flows.", "The infor-mation flows."),
        ("The INFOR MATION flows.", "The INFOR MATION flows."),
        ("The infor, mation flows.", "The infor, mation flows."),
        ("Visit www.infor mation.org today.", "Visit www.infor mation.org today."),
    ],
)
def test_join_split_words_joins_after_a_space_or_a_line_end_hyphen(
    text: str, expected: str
) -> None:
    assert join_split_words(text, SPACED) == expected


def test_join_split_words_joins_after_a_hyphen_alone_only_the_hyphenated_pairs() -> None:
    hyphenated = SplitWordJoins(hyphenated={"mation": frozenset({"infor"})})
    assert join_split_words("The infor-mation flows.", hyphenated) == "The information flows."
    assert join_split_words("The infor mation flows.", hyphenated) == "The infor mation flows."


def test_join_split_words_keeps_initials_and_capitals_apart() -> None:
    joins = SplitWordJoins(spaced={"s": frozenset({"william"}), "ummary": frozenset({"s"})})
    assert join_split_words("By William S. Wang.", joins) == "By William S. Wang."
    assert join_split_words("S ummary of the work.", joins) == "Summary of the work."


def test_join_split_words_leaves_pieces_touching_a_digit_apart() -> None:
    joins = SplitWordJoins(spaced={"and": frozenset({"st"})})
    assert join_split_words("the 1st and 3rd questions", joins) == "the 1st and 3rd questions"


def test_join_split_words_leaves_a_name_before_et_al_apart() -> None:
    joins = SplitWordJoins(spaced={"et": frozenset({"bull"})})
    assert join_split_words("as Bull et al. found", joins) == "as Bull et al. found"
    assert join_split_words("the bull et the cow", joins) == "the bullet the cow"


def test_join_split_words_does_not_join_a_joined_word_again() -> None:
    joins = SplitWordJoins(spaced={"and": frozenset({"st"}), "ard": frozenset({"stand"})})
    assert join_split_words("the st and ard rule", joins) == "the stand ard rule"


def test_join_split_words_leaves_a_text_alone_without_joins() -> None:
    assert join_split_words("The infor mation flows.", SplitWordJoins()) == (
        "The infor mation flows."
    )
