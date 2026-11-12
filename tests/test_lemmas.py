"""Tests for the type-level lemma map."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from interpretant.corpus.lemmas import (
    LemmaMap,
    american,
    learn_lemma_map,
    lemma_votes,
    normal_form,
    read_overrides,
    voted_lemma,
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
def test_regular_plurals_read_as_plural_map_to_their_singular(token: str, lemma: str) -> None:
    assert voted_lemma(token, {token: [("NNS", lemma, 10)]}) == (lemma, "plural")


def test_third_person_verbs_map_to_their_base_form() -> None:
    votes = {"uses": [("VBZ", "use", 30), ("NNS", "use", 20)]}

    assert voted_lemma("uses", votes) == ("use", "verb")


@pytest.mark.parametrize(
    ("token", "readings"),
    [
        ("loss", [("NN", "loss", 40)]),  # read as a singular noun
        ("discuss", [("VB", "discuss", 30), ("VBP", "discuss", 10)]),  # read as a verb
        ("aesthetics", [("NN", "aesthetic", 25), ("NNS", "aesthetic", 5)]),  # mostly singular
        ("structures", [("NNS", "structure", 4)]),  # too few readings
        ("data", [("NNS", "datum", 50)]),  # not a regular -s form of its lemma
    ],
)
def test_words_spacy_does_not_mostly_read_as_regular_inflections_stay(
    token: str, readings: list[tuple[str, str, int]]
) -> None:
    assert voted_lemma(token, {token: readings}) is None


def test_proper_noun_readings_do_not_count_against_a_merge() -> None:
    votes = {"states": [("NNS", "state", 60), ("NNP", "states", 40)]}

    assert voted_lemma("states", votes) == ("state", "plural")


def test_english_spelling_limits_the_singular() -> None:
    votes = {"planes": [("NNS", "plan", 10)], "plan": [("NN", "plan", 90)]}
    counts = Counter({"planes": 10, "plane": 5, "plan": 90})

    assert voted_lemma("planes", votes, counts) == ("plane", "plural")


def test_the_singular_read_as_the_matching_part_of_speech_wins() -> None:
    votes = {
        "senses": [("NNS", "sens", 50)],
        "sense": [("NN", "sense", 80)],
        "emphasises": [("VBZ", "emphasise", 20)],
        "emphasise": [("VB", "emphasise", 15)],
        "emphasis": [("NN", "emphasis", 300)],
    }
    counts = Counter({"senses": 50, "sense": 80, "sens": 3, "emphasis": 300, "emphasise": 15})

    assert voted_lemma("senses", votes, counts) == ("sense", "plural")
    assert voted_lemma("emphasises", votes, counts) == ("emphasise", "verb")


def test_a_singular_rare_in_the_corpus_is_not_merged_into() -> None:
    votes = {"polis": [("NNS", "poli", 20)]}

    assert voted_lemma("polis", votes, Counter({"polis": 100, "poli": 2})) is None


def test_lemma_votes_count_spacy_readings_by_processed_type() -> None:
    votes = lemma_votes(["The models learn structures.", "She uses Models and theories."])

    assert votes[("structures", "NNS", "structure")] == 1
    assert votes[("theories", "NNS", "theory")] == 1
    assert votes[("uses", "VBZ", "use")] == 1
    # Target variants are keyed on the type preprocessing gives them, as in the processed cells.
    assert votes[("model", "NNS", "model")] == 2


def test_lemma_votes_in_worker_processes_match_one_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("interpretant.corpus.lemmas.VOTE_CHUNK", 2)
    texts = ["The models learn structures.", "She uses theories.", "Networks grow.", "It works."]

    assert lemma_votes(texts, workers=2) == lemma_votes(texts, workers=1)


@pytest.mark.parametrize(
    "token",
    ["learning", "meaning", "embedding", "being", "embodied", "learned", "better", "was", "is"],
)
def test_verbs_participles_comparatives_and_ing_nouns_stay(token: str) -> None:
    assert normal_form(token, set(), {}, {token: [("VBG", "learn", 9)]}) == (token, "")


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
    [
        "promise",
        "exercise",
        "surprise",
        "surprising",
        "arising",
        "otherwise",
        "enterprise",
        "hour",
        "contour",
        "genre",
        "falling",
        "filled",
        "controlled",
        "willing",
    ],
)
def test_spelling_rules_leave_lookalikes_alone(token: str) -> None:
    assert american(token) is None


def test_plural_then_spelling_compose_in_one_pass() -> None:
    votes = {"behaviours": [("NNS", "behaviour", 9)], "colours": [("NNS", "colour", 7)]}

    assert normal_form("behaviours", set(), {}, votes) == ("behavior", "plural")
    assert normal_form("colours", set(), {}, votes) == ("color", "plural")


def test_targets_hyphens_and_phrases_are_never_mapped() -> None:
    votes = {"networks": [("NNS", "network", 50)]}

    assert normal_form("networks", {"networks"}, {}, votes) == ("networks", "")
    assert normal_form("self-locations", set(), {}, votes) == ("self-locations", "")
    assert normal_form("neural_networks", set(), {}, votes) == ("neural_networks", "")


def test_overrides_win_and_an_empty_lemma_keeps_the_token(tmp_path: Path) -> None:
    path = tmp_path / "overrides.tsv"
    path.write_text("# comment\nphenomena\tphenomenon\nmeans\t\n", encoding="utf-8")

    overrides = read_overrides(path)

    assert overrides == {"phenomena": "phenomenon", "means": ""}
    assert normal_form("phenomena", set(), overrides, {}) == ("phenomenon", "override")
    assert normal_form("means", set(), overrides, {}) == ("means", "")


def test_learned_map_applies_to_a_line_and_round_trips(tmp_path: Path) -> None:
    counts = Counter(
        {
            "structures": 5,
            "structure": 7,
            "learning": 9,
            "behaviours": 2,
            "behavior": 4,
            "network": 3,
        }
    )

    votes = {
        "structures": [("NNS", "structure", 6)],
        "behaviours": [("NNS", "behaviour", 5)],
        "behaviour": [("NN", "behaviour", 6)],
        "learning": [("VBG", "learn", 9)],
    }

    lemma_map = learn_lemma_map(counts, votes, protected={"network"}, overrides={})
    lemma_map.save(tmp_path / "lemmas.tsv", counts)
    loaded = LemmaMap.load(tmp_path / "lemmas.tsv")

    assert loaded.mapping == {"structures": "structure", "behaviours": "behavior"}
    assert loaded.apply("the structures of learning behaviours") == (
        "the structure of learning behavior"
    )


def test_spelling_merges_need_the_american_form_in_the_corpus() -> None:
    counts = Counter({"organisation": 50, "organization": 500, "colonisation": 50})

    assert american("organisation", counts) == "organization"
    assert american("colonisation", counts) is None
