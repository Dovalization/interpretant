"""Tests for the type-level lemma map."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from interpretant.corpus.lemmas import (
    LemmaMap,
    american,
    learn_lemma_map,
    normal_form,
    read_overrides,
    singular,
)


@pytest.mark.parametrize(
    ("token", "lemma"),
    [
        ("structures", "structure"),
        ("embeddings", "embedding"),
        ("theories", "theory"),
        ("analyses", "analysis"),
        ("boxes", "box"),
        ("affordances", "affordance"),
    ],
)
def test_regular_plurals_map_to_their_singular(token: str, lemma: str) -> None:
    assert singular(token) == lemma


@pytest.mark.parametrize(
    "token",
    ["learning", "meaning", "embedding", "being", "embodied", "learned", "better", "was", "is"],
)
def test_verbs_participles_comparatives_and_ing_nouns_stay(token: str) -> None:
    assert normal_form(token, set(), {}) == (token, "")


@pytest.mark.parametrize(
    ("token", "lemma"),
    [
        ("behaviour", "behavior"),
        ("behavioural", "behavioral"),
        ("colour", "color"),
        ("conceptualisation", "conceptualization"),
        ("organised", "organized"),
        ("analyse", "analyze"),
        ("modelling", "modeling"),
        ("labelled", "labeled"),
        ("centre", "center"),
        ("centred", "centered"),
        ("dialogue", "dialog"),
        ("programme", "program"),
        ("judgement", "judgment"),
    ],
)
def test_british_spellings_map_to_american(token: str, lemma: str) -> None:
    assert american(token) == lemma


@pytest.mark.parametrize(
    "token",
    ["promise", "exercise", "surprise", "enterprise", "hour", "contour", "genre", "falling"],
)
def test_spelling_rules_leave_lookalikes_alone(token: str) -> None:
    assert american(token) is None


def test_plural_then_spelling_compose_in_one_pass() -> None:
    assert normal_form("behaviours", set(), {}) == ("behavior", "plural")
    assert normal_form("colours", set(), {}) == ("color", "plural")


def test_targets_hyphens_and_phrases_are_never_mapped() -> None:
    assert normal_form("networks", {"networks"}, {}) == ("networks", "")
    assert normal_form("self-locations", set(), {}) == ("self-locations", "")
    assert normal_form("neural_networks", set(), {}) == ("neural_networks", "")


def test_overrides_win_and_an_empty_lemma_keeps_the_token(tmp_path: Path) -> None:
    path = tmp_path / "overrides.tsv"
    path.write_text("# comment\nphenomena\tphenomenon\nmeans\t\n", encoding="utf-8")

    overrides = read_overrides(path)

    assert overrides == {"phenomena": "phenomenon", "means": ""}
    assert normal_form("phenomena", set(), overrides) == ("phenomenon", "override")
    assert normal_form("means", set(), overrides) == ("means", "")


def test_learned_map_applies_to_a_line_and_round_trips(tmp_path: Path) -> None:
    counts = Counter({"structures": 5, "learning": 9, "behaviours": 2, "network": 3})

    lemma_map = learn_lemma_map(counts, protected={"network"}, overrides={})
    lemma_map.save(tmp_path / "lemmas.tsv", counts)
    loaded = LemmaMap.load(tmp_path / "lemmas.tsv")

    assert loaded.mapping == {"structures": "structure", "behaviours": "behavior"}
    assert loaded.apply("the structures of learning behaviours") == (
        "the structure of learning behavior"
    )
