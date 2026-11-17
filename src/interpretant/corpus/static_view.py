"""The static view's tokens of one cached spaCy document (decision agenda 2.2, 3.2; thin version).

Each word token becomes one static token: lowercase, accents folded (Unicode NFKD, combining marks
removed), the en dash written as a hyphen, and spaCy's in-context lemma for plural nouns (`NNS`)
and third-person singular verbs (`VBZ`). Punctuation, whitespace and tokens with no letter are
dropped. Each token keeps its character offset, so any build's removed ranges apply to it later
without reading the documents again, and whether it is a finite verb, for the no-finite-verb
check of agenda 2.3 (run 13). Phrases and the word tables come with the full view (agenda 9.1,
stage 6).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from spacy.tokens import Doc, Token

# Fine-grained tags whose in-context lemma the view writes (agenda 3.2): plural nouns and
# third-person singular present verbs; other forms keep their own word.
_LEMMATISED_TAGS = frozenset({"NNS", "VBZ"})
_EN_DASH = "–"


@dataclass(frozen=True)
class StaticTokens:
    """One document's static tokens, their character offsets and their finite-verb flags."""

    tokens: list[str]
    starts: list[int]
    finite_verbs: list[bool]


def static_tokens(doc: Doc) -> StaticTokens:
    """The static view's tokens of one document, in text order."""
    tokens: list[str] = []
    starts: list[int] = []
    finite_verbs: list[bool] = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        word = fold(_form(token))
        if not any(character.isalpha() for character in word):
            continue
        tokens.append(word)
        starts.append(token.idx)
        finite_verbs.append("Fin" in token.morph.get("VerbForm", None))
    return StaticTokens(tokens=tokens, starts=starts, finite_verbs=finite_verbs)


def fold(word: str) -> str:
    """Lowercase, accents folded, the en dash written as a hyphen."""
    decomposed = unicodedata.normalize("NFKD", word.lower().replace(_EN_DASH, "-"))
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def _form(token: Token) -> str:
    return token.lemma_ if token.tag_ in _LEMMATISED_TAGS and token.lemma_ else token.text
