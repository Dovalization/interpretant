"""Target matching: by lemma or form, spelled, in hyphen parts, and phrase targets."""

from __future__ import annotations

import spacy
from spacy.tokens import Doc

from interpretant.catalog.words import Word
from interpretant.usages.target_matching import (
    TargetMention,
    build_target_matcher,
    match_targets,
)

WORDS = [
    Word(lemma="attention", pos=("NOUN", "PROPN"), match="lemma", held_out=False),
    Word(lemma="behavior", pos=("NOUN", "PROPN"), match="lemma", held_out=False),
    Word(lemma="having", pos=("VERB", "AUX"), match="form", held_out=False),
    Word(lemma="learning", pos=("NOUN", "PROPN", "VERB"), match="lemma", held_out=False),
    Word(lemma="medium", pos=("NOUN", "PROPN"), match="lemma", held_out=False),
    Word(lemma="network", pos=("NOUN", "PROPN"), match="lemma", held_out=False),
    Word(lemma="neural network", pos=("NOUN", "PROPN"), match="lemma", held_out=False),
    Word(lemma="user interface", pos=("NOUN", "PROPN"), match="lemma", held_out=False),
]
MATCHER = build_target_matcher(WORDS, {"behaviour": "behavior"})


def make_doc(words: list[str], lemmas: list[str], tags: list[str] | None = None) -> Doc:
    """A document of the given tokens, separated by spaces, with their lemmas and tags."""
    vocab = spacy.blank("en").vocab
    return Doc(vocab, words=words, lemmas=lemmas, tags=tags or ["NN"] * len(words))


def found(doc: Doc) -> list[tuple[str, str, str | None, str | None]]:
    """Each mention as (word, the text it covers, compound, position)."""
    return [
        (m.word, doc.text[m.start_char : m.end_char], m.compound, m.position)
        for m in match_targets(doc, MATCHER)
    ]


def test_match_targets_finds_a_word_by_lemma_and_by_form() -> None:
    doc = make_doc(["Networks", "help", "learning"], ["network", "help", "learn"])
    assert found(doc) == [("network", "Networks", None, None), ("learning", "learning", None, None)]


def test_match_targets_keeps_media_apart_from_medium() -> None:
    doc = make_doc(["media", "and", "a", "medium"], ["media", "and", "a", "medium"])
    assert found(doc) == [("medium", "medium", None, None)]


def test_match_targets_finds_a_british_spelling_as_the_word() -> None:
    doc = make_doc(["human", "behaviour"], ["human", "behaviour"])
    assert found(doc) == [("behavior", "behaviour", None, None)]


def test_match_targets_matches_a_form_only_word_by_its_form() -> None:
    doc = make_doc(["having", "has"], ["have", "have"])
    assert found(doc) == [("having", "having", None, None)]


def test_match_targets_finds_a_capitalised_plural_read_as_a_name() -> None:
    doc = make_doc(["Neural", "Networks"], ["Neural", "Networks"], ["NNP", "NNPS"])
    assert found(doc) == [
        ("network", "Networks", None, None),
        ("neural network", "Neural Networks", None, None),
    ]


def test_match_targets_finds_a_plural_read_as_a_singular_proper_noun() -> None:
    doc = make_doc(["networks", "grow"], ["networks", "grow"], ["NNP", "VBP"])
    assert found(doc) == [("network", "networks", None, None)]
    doc = make_doc(["User", "Interfaces"], ["User", "Interfaces"], ["NNP", "NNP"])
    assert found(doc) == [("user interface", "User Interfaces", None, None)]


def test_match_targets_finds_a_phrase_reaching_into_a_hyphenated_token() -> None:
    doc = make_doc(["multi-user", "interfaces"], ["multi-user", "interface"], ["JJ", "NNS"])
    assert found(doc) == [("user interface", "user interfaces", "multi-user", "head")]
    doc = make_doc(["neural", "network-based"], ["neural", "network-based"], ["JJ", "JJ"])
    assert found(doc) == [
        ("network", "network", "network-based", "modifier"),
        ("neural network", "neural network", "network-based", "modifier"),
    ]
    doc = make_doc(["neural", "networks-based"], ["neural", "networks-based"], ["JJ", "JJ"])
    assert found(doc) == [("neural network", "neural networks", "networks-based", "modifier")]


def test_match_targets_finds_targets_as_parts_of_hyphenated_words() -> None:
    doc = make_doc(
        ["self-attention", "and", "attention-deficit", "e-learning"],
        ["self-attention", "and", "attention-deficit", "e-learning"],
    )
    assert found(doc) == [
        ("attention", "attention", "self-attention", "head"),
        ("attention", "attention", "attention-deficit", "modifier"),
        ("learning", "learning", "e-learning", "head"),
    ]


def test_match_targets_finds_a_plural_last_part() -> None:
    doc = make_doc(["actor-networks"], ["actor-networks"])
    assert found(doc) == [("network", "networks", "actor-networks", "head")]


def test_match_targets_finds_a_phrase_with_an_inflected_head() -> None:
    doc = make_doc(["deep", "neural", "networks"], ["deep", "neural", "network"])
    assert found(doc) == [
        ("network", "networks", None, None),
        ("neural network", "neural networks", None, None),
    ]


def test_match_targets_finds_a_hyphenated_phrase() -> None:
    doc = make_doc(["neural-networks"], ["neural-networks"])
    assert found(doc) == [
        ("network", "networks", "neural-networks", "head"),
        ("neural network", "neural-networks", "neural-networks", None),
    ]


def test_match_targets_records_the_head_token() -> None:
    doc = make_doc(["neural", "network"], ["neural", "network"])
    phrase = [m for m in match_targets(doc, MATCHER) if m.word == "neural network"]
    assert phrase == [TargetMention("neural network", 1, 0, 14, None, None)]
