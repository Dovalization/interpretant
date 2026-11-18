"""Spelling variants: each hyphen segment mapped, one token for one."""

from __future__ import annotations

from interpretant.corpus.spelling import has_accent, mapped_away, spell

SPELLINGS = {"behaviour": "behavior", "colouring": "coloring", "organise": "organize"}


def test_spell_maps_a_whole_word() -> None:
    assert spell("behaviour", SPELLINGS) == "behavior"


def test_spell_leaves_an_unlisted_word() -> None:
    assert spell("model", SPELLINGS) == "model"


def test_spell_maps_each_hyphen_segment() -> None:
    assert spell("edge-colouring", SPELLINGS) == "edge-coloring"
    assert spell("self-organise", SPELLINGS) == "self-organize"
    assert spell("self-attention", SPELLINGS) == "self-attention"


def test_mapped_away_names_the_words_the_map_would_change() -> None:
    assert mapped_away(SPELLINGS, ["behavior", "behaviour", "model"]) == ["behaviour"]


def test_has_accent_finds_combining_marks_only() -> None:
    assert has_accent("utilisé")
    assert has_accent("naïve")
    assert not has_accent("utilise")
    assert not has_accent("edge-colouring")
