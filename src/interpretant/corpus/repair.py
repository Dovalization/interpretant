"""Repair before spaCy: undo extraction artefacts so the stored text is clean, then never edit it.

The decision agenda, step 2.3: NFKC, control and format characters by Unicode category, soft
hyphens; abstracts with mojibake are dropped, not repaired, so detection reads the raw text.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

# A literal backslash-n left in the text by an upstream extractor.
_LITERAL_NEWLINE = re.compile(r"\\n")
# A soft hyphen marks where a word was split across lines; any space after it is part of the
# split, so both go and the word is whole again.
_SOFT_HYPHEN = re.compile(r"­\s?")
_SPACES = re.compile(r"\s+")
# Characters that separate words although they are not spaces: the zero-width space (a break
# opportunity) and the replacement character (a lost character).
_WORD_SEPARATORS = frozenset({"​", "�"})
_WHITESPACE_CONTROLS = frozenset("\t\n\r\x0b\x0c")


def repair_text(raw: str) -> str:
    """The abstract without extraction artefacts, in Unicode normal form NFKC.

    Line breaks, tabs, zero-width spaces and replacement characters become spaces; other control
    and format characters (category Cc or Cf, such as the byte-order mark or the zero-width
    joiner) are removed; soft hyphens rejoin the word they split; runs of whitespace collapse to
    one space. NFKC also expands ligatures (*ﬁ* to *fi*), maps full-width punctuation and
    no-break spaces to their plain forms and writes superscripts as plain digits.
    """
    text = _SOFT_HYPHEN.sub("", _LITERAL_NEWLINE.sub(" ", raw))
    kept = []
    for character in text:
        if character in _WHITESPACE_CONTROLS or character in _WORD_SEPARATORS:
            kept.append(" ")
        elif unicodedata.category(character) not in ("Cc", "Cf"):
            kept.append(character)
    normal = unicodedata.normalize("NFKC", "".join(kept))
    return _SPACES.sub(" ", normal).strip()


def compile_signatures(patterns: Sequence[str]) -> re.Pattern[str]:
    """One regular expression matching any of the given mojibake signature patterns."""
    return re.compile("|".join(f"(?:{pattern})" for pattern in patterns))


def has_mojibake(raw: str, signatures: re.Pattern[str]) -> bool:
    """Whether the raw text shows a mis-decoding signature, such as *Ã©* for *é*."""
    return signatures.search(raw) is not None
