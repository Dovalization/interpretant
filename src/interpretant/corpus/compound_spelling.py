"""Compound spelling variants written as one form (decision agenda 3.4; phase step 3).

A two-part compound is written open (*data base*), hyphenated (*data-base*) or closed
(*database*). The map (`curated/compounds.tsv`, audited in `docs/compounds.md`) sends each variant
to the form the corpus uses most often over kept works (the author, 2026-11-18):

- a one-token form, hyphenated or closed, is rewritten whole, and its standard may be spaced
  (*user-interface* is written *user interface*);
- a spaced pair is merged only as a noun-noun compound: two tokens tagged as nouns, adjacent in
  the document and in one sentence. So *is a* or *how to* never merge, nor an adjective-first pair
  (*long term*): before a noun, an adjective-first compound's hyphen is grammar, not spelling.

The static view, the map's builder and its audit all use this module, so they count and merge the
same pairs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from spacy.tokens import Token

NOUN_TAGS = frozenset({"NOUN", "PROPN"})


@dataclass(frozen=True)
class CompoundMap:
    """The one-token forms and the spaced pairs, each with the standard it is written as."""

    one_token: Mapping[str, str] = field(default_factory=dict)
    spaced: Mapping[tuple[str, str], str] = field(default_factory=dict)


# The map that changes nothing.
NO_COMPOUNDS = CompoundMap()


def build_compound_map(pairs: Iterable[tuple[str, str]]) -> CompoundMap:
    """The map of (form, standard) pairs, a form with a space being a spaced pair."""
    one_token: dict[str, str] = {}
    spaced: dict[tuple[str, str], str] = {}
    for form, standard in pairs:
        if " " in form:
            first, second = form.split(" ")
            spaced[(first, second)] = standard
        else:
            one_token[form] = standard
    return CompoundMap(one_token=one_token, spaced=spaced)


def compound_key(form: str) -> str:
    """The form's letters without its hyphen or space: the key its family shares."""
    return form.replace("-", "").replace(" ", "")


def is_noun_compound(first: Token, second: Token) -> bool:
    """Whether two tokens can be a spaced noun-noun compound.

    They must be adjacent in the document (no punctuation, number or space between), apart by
    whitespace (not one word the tokenizer split), in one sentence, and both tagged as nouns.
    """
    return (
        second.i == first.i + 1
        and bool(first.whitespace_)
        and not second.is_sent_start
        and first.pos_ in NOUN_TAGS
        and second.pos_ in NOUN_TAGS
    )


def compounds_written_away(compounds: CompoundMap, words: Iterable[str]) -> list[str]:
    """The words the map would write as something else (none of the word table may be).

    A word is written away if it is a one-token form, a word of a spaced form, or, for a phrase,
    a whole spaced form.
    """
    written = set(compounds.one_token)
    written |= {word for pair in compounds.spaced for word in pair}
    written |= {" ".join(pair) for pair in compounds.spaced}
    return sorted(word for word in words if word in written)
