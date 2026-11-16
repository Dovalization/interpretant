"""Whether spaCy knows an English word (decision agenda 2.3, the word repairs).

The word repairs leave a word alone when spaCy knows it, so a real word is never split or joined
only because the corpus also writes its pieces as an ordinary phrase (*abroad* and *a broad*).
spaCy knows a word when it is one of its English stop words or a lemma in the lemma index of the
small English model's rule lemmatizer, directly or after the lemmatizer's suffix rules and
exceptions, read as a noun, verb, adjective or adverb (*children* reduces to *child*). The index
holds WordNet's lemmas as the model ships them; it lacks many technical words, names and foreign
words, which the corpus counts decide instead.
"""

from __future__ import annotations

from functools import cache
from typing import cast

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.pipeline import Lemmatizer
from spacy.tokens import Doc

# The model the spaCy pass loads (``annotation.build_pipeline``'s default).
SPACY_MODEL = "en_core_web_sm"
# The parts of speech the lemma index lists.
INDEX_POS = ("noun", "verb", "adj", "adv")


@cache
def is_dictionary_word(word: str) -> bool:
    """Whether spaCy knows the word, in any case.

    It does when the word is a stop word, or a lemma of the index directly or after the rule
    lemmatizer.
    """
    folded = word.lower()
    if folded in STOP_WORDS or any(folded in _index(pos) for pos in INDEX_POS):
        return True
    return any(lemma in _index(pos) for pos in INDEX_POS for lemma in _lemmas(folded, pos))


@cache
def _lemmatizer() -> Lemmatizer:
    """The rule lemmatizer of the small English model; the other components are not loaded."""
    pipeline = spacy.load(SPACY_MODEL, exclude=["tok2vec", "tagger", "parser", "attribute_ruler"])
    return cast(Lemmatizer, pipeline.get_pipe("lemmatizer"))


@cache
def _index(pos: str) -> frozenset[str]:
    return frozenset(_lemmatizer().lookups.get_table("lemma_index")[pos])


def _lemmas(word: str, pos: str) -> list[str]:
    """The rule lemmatizer's lemmas of ``word`` read as ``pos``, out of context.

    Rule mode reads the token's part of speech, so a one-word document carries it.
    """
    token = Doc(_lemmatizer().vocab, words=[word])[0]
    token.pos_ = pos.upper()
    return _lemmatizer().rule_lemmatize(token)
