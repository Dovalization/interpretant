"""The lookup attribute ruler: the same matches and annotations as spaCy's attribute ruler."""

from __future__ import annotations

from typing import Any

import spacy
from spacy.pipeline import AttributeRuler
from spacy.tokens import Doc

from interpretant.corpus.attribute_rules import LookupAttributeRuler

RULES: list[dict[str, Any]] = [
    {"patterns": [[{"TAG": "NN"}]], "attrs": {"POS": "NOUN"}},
    {"patterns": [[{"TAG": "NNS", "LOWER": {"IN": ["data", "criteria"]}}]], "attrs": {"POS": "X"}},
    {"patterns": [[{"LOWER": "not"}]], "attrs": {"LEMMA": "not"}},
    {"patterns": [[{"LOWER": "y'"}, {"LOWER": "all"}]], "attrs": {"LEMMA": "you"}, "index": 0},
    {"patterns": [[{"LOWER": {"REGEX": "^n[ou]thin'$"}}]], "attrs": {"LEMMA": "nothing"}},
    {"patterns": [[{"TAG": "NN"}]], "attrs": {"POS": "PROPN"}},
]


def rulers() -> tuple[AttributeRuler, LookupAttributeRuler, Doc]:
    nlp = spacy.blank("en")
    stock, lookup = AttributeRuler(nlp.vocab), LookupAttributeRuler(nlp.vocab)
    stock.add_patterns(RULES)
    lookup.add_patterns(RULES)
    words = ["The", "data", "are", "not", "nothin'", "y'", "all", "model", "criteria"]
    tags = ["DT", "NNS", "VBP", "RB", "NN", "PRP", "DT", "NN", "NNS"]
    return stock, lookup, Doc(nlp.vocab, words=words, tags=tags)


def test_lookup_attribute_ruler_finds_the_matches_spacys_ruler_finds() -> None:
    stock, lookup, doc = rulers()
    assert lookup.match(doc) == stock.match(doc)
    assert lookup.patterns == stock.patterns


def test_lookup_attribute_ruler_lets_later_rules_win_as_spacys_ruler_does() -> None:
    stock, lookup, doc = rulers()
    expected = [(token.pos_, token.lemma_) for token in stock(doc.copy())]
    assert [(token.pos_, token.lemma_) for token in lookup(doc)] == expected
    assert doc[7].pos_ == "PROPN"


def test_lookup_attribute_ruler_forgets_its_rules_when_cleared() -> None:
    _, lookup, doc = rulers()
    lookup.clear()
    assert lookup.match(doc) == []
    lookup.add_patterns(RULES[:1])
    assert [start for _, _, start, _ in lookup.match(doc)] == [4, 7]
