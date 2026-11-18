"""Compound spelling variants: the noun-compound gate, the map and the written-away check."""

from __future__ import annotations

import pytest
import spacy
from spacy.tokens import Doc

from interpretant.corpus.compound_spelling import (
    NO_COMPOUNDS,
    build_compound_map,
    compound_key,
    compounds_written_away,
    is_noun_compound,
)


def tagged(
    words: list[str],
    pos: list[str],
    spaces: list[bool] | None = None,
    sent_starts: list[bool | int | None] | None = None,
) -> Doc:
    """A document with fixed tags, so the gate is tested apart from the tagger."""
    return Doc(
        spacy.blank("en").vocab, words=words, spaces=spaces, pos=pos, sent_starts=sent_starts
    )


def test_two_adjacent_nouns_can_be_a_compound() -> None:
    doc = tagged(["the", "data", "base", "grew"], ["DET", "NOUN", "NOUN", "VERB"])
    assert is_noun_compound(doc[1], doc[2])


def test_a_proper_noun_counts_as_a_noun() -> None:
    doc = tagged(["Web", "site"], ["PROPN", "NOUN"])
    assert is_noun_compound(doc[0], doc[1])


@pytest.mark.parametrize(
    ("words", "pos", "spaces", "pair"),
    [
        (["this", "is", "a", "test"], ["DET", "AUX", "DET", "NOUN"], None, (1, 2)),
        (["the", "long", "term", "effect"], ["DET", "ADJ", "NOUN", "NOUN"], None, (1, 2)),
        (["data", ",", "base"], ["NOUN", "PUNCT", "NOUN"], [False, True, False], (0, 2)),
        (["data", "12", "base"], ["NOUN", "NUM", "NOUN"], None, (0, 2)),
        (["data", "base"], ["NOUN", "NOUN"], [False, False], (0, 1)),
    ],
    ids=["auxiliary", "adjective first", "comma between", "number between", "no space"],
)
def test_other_pairs_are_not_compounds(
    words: list[str], pos: list[str], spaces: list[bool] | None, pair: tuple[int, int]
) -> None:
    doc = tagged(words, pos, spaces)
    assert not is_noun_compound(doc[pair[0]], doc[pair[1]])


def test_a_pair_across_two_sentences_is_not_a_compound() -> None:
    doc = tagged(
        ["we", "used", "data", "base", "rates", "rose"],
        ["PRON", "VERB", "NOUN", "NOUN", "NOUN", "VERB"],
        sent_starts=[True, False, False, True, False, False],
    )
    assert not is_noun_compound(doc[2], doc[3])


def test_build_compound_map_splits_spaced_pairs_from_one_token_forms() -> None:
    compounds = build_compound_map(
        [("data base", "database"), ("re-search", "research"), ("user-interface", "user interface")]
    )
    assert compounds.spaced == {("data", "base"): "database"}
    assert compounds.one_token == {"re-search": "research", "user-interface": "user interface"}


def test_compound_key_is_the_letters_of_any_form() -> None:
    assert compound_key("data base") == compound_key("data-base") == compound_key("database")


def test_compounds_written_away_names_forms_and_the_words_of_spaced_forms() -> None:
    compounds = build_compound_map([("web site", "website"), ("on-line", "online")])
    words = ["web", "site", "website", "web site", "on-line", "online", "network"]
    assert compounds_written_away(compounds, words) == ["on-line", "site", "web", "web site"]


def test_no_compounds_writes_nothing_away() -> None:
    assert compounds_written_away(NO_COMPOUNDS, ["data", "base"]) == []
