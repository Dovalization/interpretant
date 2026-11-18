"""The static view's tokens of one cached spaCy document (decision agenda 2.2, 3.2, 3.4).

Each word token becomes one entry: lowercase, accents folded (Unicode NFKD, combining marks
removed), the en dash written as a hyphen, and spaCy's in-context lemma for plural nouns (`NNS`)
and third-person singular verbs (`VBZ`), auxiliaries left as they are (*is*, *has*). The stage
applies the lemma exceptions (`corpus/lemma_exceptions.py`) to the document first, so the lemma
read here is spaCy's or the exception's. The written word is then spelled by the spelling map
(agenda 3.4, `corpus/spelling.py`), per hyphen segment, except in tokens whose text carries an
accent: those are mostly French words folded into English look-alikes (*utilisé*), so the map
leaves them as they are (`docs/spellings.md`, 2026-11-17). Proper nouns are spelled like any word.

The compound map (phase step 3, `corpus/compound_spelling.py`) comes next, accented tokens again
left alone. A hyphenated or closed form is rewritten whole, its standard possibly holding a space
(*user-interface* is written *user interface*). Then each spaced noun-noun compound, taken left to
right without overlaps, writes its standard in its first entry and an empty string in its second.
So the view keeps one entry per word token and stays aligned with a view without the maps; lines
read the entries as words (`corpus/cell_versions.segment_words`). An empty entry only follows a
non-empty one in the same sentence, and an entry holds at most one space.

Punctuation, whitespace and tokens with no letter are dropped. Each entry keeps its token's
character offset, so any build's removed ranges apply to it later without reading the documents
again, and whether it is a finite verb, for the no-finite-verb check of agenda 2.3 (run 13).
Single words stay unjoined: there is no phrase list (agenda 3.5, 2026-11-18).
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass

from spacy.tokens import Doc, Token

from interpretant.corpus.compound_spelling import CompoundMap, is_noun_compound
from interpretant.corpus.spelling import has_accent, spell

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


def static_tokens(doc: Doc, spellings: Mapping[str, str], compounds: CompoundMap) -> StaticTokens:
    """The static view's entries of one document, in text order, spelled and compounded."""
    kept: list[Token] = []
    words: list[str] = []
    finite_verbs: list[bool] = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        word = fold(_form(token))
        if not any(character.isalpha() for character in word):
            continue
        if not has_accent(token.text):
            word = spell(word, spellings)
            word = compounds.one_token.get(word, word)
        kept.append(token)
        words.append(word)
        finite_verbs.append("Fin" in token.morph.get("VerbForm", None))
    return StaticTokens(
        tokens=_merge_spaced(kept, words, compounds),
        starts=[token.idx for token in kept],
        finite_verbs=finite_verbs,
    )


def fold(word: str) -> str:
    """Lowercase, accents folded, the en dash written as a hyphen."""
    decomposed = unicodedata.normalize("NFKD", word.lower().replace(_EN_DASH, "-"))
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def _merge_spaced(tokens: list[Token], words: list[str], compounds: CompoundMap) -> list[str]:
    """The words with each spaced compound written as its standard, then an empty entry."""
    if not compounds.spaced:
        return words
    merged = list(words)
    index = 0
    while index < len(tokens) - 1:
        first, second = tokens[index], tokens[index + 1]
        standard = compounds.spaced.get((words[index], words[index + 1]))
        accented = has_accent(first.text) or has_accent(second.text)
        if standard is not None and not accented and is_noun_compound(first, second):
            merged[index], merged[index + 1] = standard, ""
            index += 2
        else:
            index += 1
    return merged


def _form(token: Token) -> str:
    lemmatised = token.tag_ in _LEMMATISED_TAGS and token.pos_ != "AUX" and bool(token.lemma_)
    return token.lemma_ if lemmatised else token.text
