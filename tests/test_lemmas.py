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
        ("bacteria", [("NNS", "bacterium", 50)]),  # neither regular nor in spaCy's irregulars
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


@pytest.mark.parametrize(
    ("token", "lemma"),
    [
        ("mice", "mouse"),
        ("wives", "wife"),
        ("appendices", "appendix"),
        ("shelves", "shelf"),  # not the verb *shelve* the spelling rule offers
        ("testes", "testis"),
    ],
)
def test_irregular_plurals_map_to_spacys_singular(token: str, lemma: str) -> None:
    votes = {token: [("NNS", lemma, 20)], lemma: [("NN", lemma, 30)]}

    assert voted_lemma(token, votes, Counter({token: 20, lemma: 30})) == (lemma, "plural")


def test_irregular_singulars_apply_only_to_plural_nouns() -> None:
    votes = {"leaves": [("VBZ", "leave", 50), ("NNS", "leaf", 10)], "leave": [("VB", "leave", 40)]}

    assert voted_lemma("leaves", votes) == ("leave", "verb")


def test_men_plurals_need_a_singular_read_as_a_noun() -> None:
    votes = {
        "freshmen": [("NNS", "freshman", 30)],
        "freshman": [("NN", "freshman", 25)],
        "specimen": [("NNS", "speciman", 30)],
    }
    counts = Counter({"freshmen": 30, "freshman": 25, "specimen": 30})

    assert voted_lemma("freshmen", votes, counts) == ("freshman", "plural")
    assert voted_lemma("specimen", votes, counts) is None
    assert voted_lemma("specimen", votes) is None


def test_men_plurals_are_not_merged_into_a_name() -> None:
    votes = {"hymen": [("NNS", "hyman", 15)], "hyman": [("NN", "hyman", 18), ("NNP", "hyman", 375)]}

    assert voted_lemma("hymen", votes, Counter({"hymen": 19, "hyman": 372})) is None


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
        ("characterise", "characterize"),
        ("characterising", "characterizing"),
        ("summarise", "summarize"),
        ("theorise", "theorize"),
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
        "rising",
        "sunrise",
        "uprising",
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


def test_american_forms_missing_from_spacys_word_list_count_when_the_corpus_has_them() -> None:
    counts = Counter({"contextualise": 270, "contextualize": 810, "personalisation": 500})

    assert american("contextualise") is None  # no corpus, so the word list decides
    assert american("contextualise", counts) == "contextualize"
    assert american("personalisation", counts) is None


def test_chains_are_followed_to_their_end() -> None:
    counts = Counter({"particularises": 6, "particularizes": 40, "particularize": 90})
    votes = {
        "particularizes": [("VBZ", "particularize", 30)],
        "particularize": [("VB", "particularize", 60)],
    }

    lemma_map = learn_lemma_map(counts, votes, protected=set(), overrides={})

    assert lemma_map.mapping["particularises"] == "particularize"
    assert lemma_map.rules["particularises"] == "spelling"
