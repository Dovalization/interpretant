"""The static view's tokens of one cached spaCy document (decision agenda 2.2, 3.2; thin version).

Each word token becomes one static token: lowercase, accents folded (Unicode NFKD, combining marks
removed), the en dash written as a hyphen, and spaCy's in-context lemma for plural nouns (`NNS`)
and third-person singular verbs (`VBZ`), auxiliaries left as they are (*is*, *has*). The stage
applies the lemma exceptions (`corpus/lemma_exceptions.py`) to the document first, so the lemma
read here is spaCy's or the exception's. The written word is then spelled by the spelling map
(agenda 3.4, `corpus/spelling.py`), per hyphen segment, except in tokens whose text carries an
accent: those are mostly French words folded into English look-alikes (*utilisé*), so the map
leaves them as they are (`docs/spellings.md`, 2026-11-17). Proper nouns are spelled like any word.
Punctuation, whitespace and tokens with no letter are dropped. Each token keeps its character
offset, so any build's removed ranges apply to it later without reading the documents again, and
whether it is a finite verb, for the no-finite-verb check of agenda 2.3 (run 13). Phrases and the
word tables come with the full view (agenda 9.1, stage 6).
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass

from spacy.tokens import Doc, Token

from interpretant.corpus.spelling import spell

# Fine-grained tags whose in-context lemma the view writes (agenda 3.2): plural nouns and
# third-person singular present verbs, auxiliaries aside; other forms keep their own word.
_LEMMATISED_TAGS = frozenset({"NNS", "VBZ"})
_EN_DASH = "–"


@dataclass(frozen=True)
class StaticTokens:
    """One document's static tokens, their character offsets and their finite-verb flags."""

    tokens: list[str]
    starts: list[int]
    finite_verbs: list[bool]


def static_tokens(doc: Doc, spellings: Mapping[str, str]) -> StaticTokens:
    """The static view's tokens of one document, in text order, spelled by ``spellings``."""
    tokens: list[str] = []
    starts: list[int] = []
    finite_verbs: list[bool] = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        word = fold(_form(token))
        if not any(character.isalpha() for character in word):
            continue
        tokens.append(word if _is_accented(token.text) else spell(word, spellings))
        starts.append(token.idx)
        finite_verbs.append("Fin" in token.morph.get("VerbForm", None))
    return StaticTokens(tokens=tokens, starts=starts, finite_verbs=finite_verbs)


def fold(word: str) -> str:
    """Lowercase, accents folded, the en dash written as a hyphen."""
    decomposed = unicodedata.normalize("NFKD", word.lower().replace(_EN_DASH, "-"))
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def _is_accented(text: str) -> bool:
    return any(
        unicodedata.combining(character) for character in unicodedata.normalize("NFKD", text)
    )


def _form(token: Token) -> str:
    lemmatised = token.tag_ in _LEMMATISED_TAGS and token.pos_ != "AUX" and bool(token.lemma_)
    return token.lemma_ if lemmatised else token.text
