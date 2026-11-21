"""What a usage's context says about it: its role, and flags for the drift checks.

Recorded for the checks step 7 pre-registers, not for scope (`docs/usages_audit.md`; the author,
2026-11-20):

- **The role:** a compound part (a part of a hyphenated word), a noun modifier (a noun or
  adjective-tagged target followed by a noun, *network analysis*), or a head. Contextual vectors
  cluster by syntactic role (Kutuzov et al. 2022), and the role mix moves over the decades, so the
  measures are reported with and without modifiers and compound parts.
- **Capitalised:** the target starts with a capital and does not open its sentence: titles,
  headings and capitalised terms, where the name rules are least reliable.
- **A mention:** the word mentioned as a word, between quotation marks or after *term*, *word*,
  *concept*, *notion*, *label*, *expression* or *phrase* (*the term "attention"*).
- **A list-like sentence:** two or more table-of-contents separators (``.-``), or at least six
  content words, three in five of them capitalised (titles, lists of chapters or instruments).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from spacy.tokens import Doc

_NOUNS = frozenset({"NOUN", "PROPN"})
_MODIFIER_TAGS = _NOUNS | {"ADJ"}
_OPENING = " \"'‘“([-…"
_QUOTES_OPEN = frozenset("\"'“‘«")
_QUOTES_CLOSE = frozenset("\"'”’»")
_MENTION_BEFORE = re.compile(
    r"\b(?:terms?|words?|concepts?|notions?|labels?|expressions?|phrases?)\s*[\"'“‘«]?\s*$",
    re.IGNORECASE,
)
LIST_SEPARATOR = ".-"
LIST_SEPARATORS = 2
LIST_MIN_WORDS = 6
LIST_CAPITAL_SHARE = 0.6


@dataclass(frozen=True)
class UsageTraits:
    """A usage's role and flags."""

    role: str
    capitalised: bool
    mention: bool
    list_like: bool


def usage_traits(
    doc: Doc, start_char: int, end_char: int, head: int, *, hyphen_part: bool
) -> UsageTraits:
    """The traits of the usage at ``[start_char, end_char)`` of a sentence, ``head`` its token."""
    text = doc.text
    before, after = text[:start_char], text[end_char:]
    quoted = before[-1:] in _QUOTES_OPEN and after[:1] in _QUOTES_CLOSE
    return UsageTraits(
        role=_role(doc, head, hyphen_part=hyphen_part),
        capitalised=text[start_char : start_char + 1].isupper() and bool(before.strip(_OPENING)),
        mention=quoted or bool(_MENTION_BEFORE.search(before)),
        list_like=is_list_like(doc),
    )


def is_list_like(doc: Doc) -> bool:
    """Whether a sentence reads as a list or a title rather than prose (the module's rule)."""
    if doc.text.count(LIST_SEPARATOR) >= LIST_SEPARATORS:
        return True
    words = [token for token in doc if token.is_alpha and not token.is_stop]
    if len(words) < LIST_MIN_WORDS:
        return False
    capitalised = sum(token.text[0].isupper() for token in words)
    return capitalised / len(words) >= LIST_CAPITAL_SHARE


def _role(doc: Doc, head: int, *, hyphen_part: bool) -> str:
    if hyphen_part:
        return "compound"
    following = doc[head + 1] if head + 1 < len(doc) else None
    modifies = following is not None and following.pos_ in _NOUNS
    return "modifier" if modifies and doc[head].pos_ in _MODIFIER_TAGS else "head"
