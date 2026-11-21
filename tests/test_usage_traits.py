"""A usage's role and flags for the drift checks: compound parts, modifiers, capitals, mentions."""

from __future__ import annotations

import spacy
from spacy.tokens import Doc

from interpretant.catalog.usages import ROLES
from interpretant.usages.traits import UsageTraits, is_list_like, usage_traits

VOCAB = spacy.blank("en").vocab


def traits(
    words: list[str], pos: list[str], target: str, *, hyphen_part: bool = False
) -> UsageTraits:
    """The traits of the first token that is the target, in a sentence of these tokens."""
    doc = Doc(VOCAB, words=words, pos=pos)
    head = words.index(target)
    start = doc[head].idx
    return usage_traits(doc, start, start + len(target), head, hyphen_part=hyphen_part)


def test_the_role_is_a_compound_part_a_noun_modifier_or_a_head() -> None:
    noun_noun = traits(["a", "network", "analysis"], ["DET", "NOUN", "NOUN"], "network")
    tagged_adjective = traits(["the", "network", "model"], ["DET", "ADJ", "NOUN"], "network")
    head = traits(["the", "network", "grew"], ["DET", "NOUN", "VERB"], "network")
    last = traits(["a", "network"], ["DET", "NOUN"], "network")
    assert noun_noun.role == tagged_adjective.role == "modifier"
    assert head.role == last.role == "head"
    part = traits(["self-attention", "works"], ["NOUN", "VERB"], "self-attention", hyphen_part=True)
    assert part.role == "compound"
    assert {noun_noun.role, head.role, part.role} <= set(ROLES)


def test_a_capitalised_target_inside_a_sentence_is_flagged_but_not_one_opening_it() -> None:
    inside = traits(["the", "Network", "grew"], ["DET", "PROPN", "VERB"], "Network")
    opening = traits(["Networks", "grew"], ["NOUN", "VERB"], "Networks")
    after_quote = traits(['"', "Networks", "grew"], ["PUNCT", "NOUN", "VERB"], "Networks")
    assert inside.capitalised
    assert not opening.capitalised
    assert not after_quote.capitalised


def test_the_word_mentioned_as_a_word_is_a_mention() -> None:
    pos = ["DET", "NOUN", "PUNCT", "NOUN", "PUNCT", "AUX", "ADJ"]
    quoted = traits(["the", "term", '"', "attention", '"', "is", "vague"], pos, "attention")
    named = traits(
        ["the", "concept", "attention", "is", "vague"],
        ["DET", "NOUN", "NOUN", "AUX", "ADJ"],
        "attention",
    )
    used = traits(
        ["their", "attention", "is", "vague"], ["PRON", "NOUN", "AUX", "ADJ"], "attention"
    )
    assert quoted.mention
    assert named.mention
    assert not used.mention


def sentence(text: str) -> Doc:
    """A document of the text's space-separated tokens."""
    return Doc(VOCAB, words=text.split())


def test_a_title_or_a_table_of_contents_is_a_list_like_sentence() -> None:
    assert is_list_like(
        sentence("Neural Networks for Visual Pattern Recognition and Control Systems")
    )
    assert is_list_like(sentence("Introduction.- Attention.- Memory.- Index"))
    assert not is_list_like(
        sentence("We study how attention shapes memory in young children over time")
    )
