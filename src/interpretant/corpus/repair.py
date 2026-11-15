"""Repair before spaCy: undo extraction artefacts so the stored text is clean, then never edit it.

The decision agenda, step 2.3: NFKC, control and format characters by Unicode category, soft
hyphens, the space before punctuation and the missing space after it; abstracts with mojibake are
dropped, not repaired, so detection reads the raw text.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from spacy.lang.char_classes import (
    ALPHA_LOWER,
    ALPHA_UPPER,
    CONCAT_QUOTES,
    LATIN,
    LATIN_LOWER,
    LATIN_UPPER,
)
from spacy.lang.lex_attrs import like_email, like_url

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
# The Unicode hyphen characters (U+2010; NFKC turns the non-breaking U+2011 into it), written as
# the ASCII hyphen-minus, so "post‐training" and "post-training" are one word (52,394 in run 2).
_HYPHENS = str.maketrans({"\u2010": "-", "\u2011": "-"})
# Sentence punctuation with no space before the next sentence: "very rare.It begins",
# "(r = 0.922).In addition", 'learns."Acquaintance', 'night.\u201cFading\u201d' (55,202 of run 3's
# 705,283 works). It follows a lowercase letter, a digit, a closing bracket or a quote and may
# carry a closing quote or bracket; the space goes before the next sentence's opening quote or
# bracket and its capitalised word, so "U.S.Army", "E.coli", "3.5" and "weeks.4" stay as they
# are. The letter classes are spaCy's, as in the tokenizer's rules.
_QUOTES = CONCAT_QUOTES.replace(",", "")
_GLUED_SENTENCE = re.compile(
    rf"(?<=[{ALPHA_LOWER}0-9\)\]{_QUOTES}])[.?!][\"'\u201d\u2019\u00bb\)\]]?"
    rf"(?=[\u201c\u2018\u00ab\(\[]?[{ALPHA_UPPER}][{ALPHA_LOWER}])"
)
# A comma glued to the next word: "paper,the results", "body,spleen,intestine" (62,889 in 17,909
# of run 3's kept works). A single letter on both sides keeps "N,N-dimethyl" and "(i,j)" whole.
_GLUED_COMMA = re.compile(rf"(?<=[{LATIN}]{{2}}),(?=[{LATIN}])|(?<=[{LATIN}]),(?=[{LATIN}]{{2}})")
# A semicolon glued to the next word: "form;comprehensively", "(Ragni, 2005;Ragni, 2007)" (4,439
# in 2,340 kept works).
_GLUED_SEMICOLON = re.compile(rf"(?<=[{LATIN}0-9\)\]]);(?=[{LATIN}])")
# A colon glued to a capitalised word: "Objective:To study", "New York:Routledge" (4,952 in 3,926
# kept works). Before a lowercase word it is left alone, since ratios ("light:dark", "h:min") and
# vowel length ("a:s\u0268bi") are written that way, and so is a colon after one letter ("x:Gx").
_GLUED_COLON = re.compile(rf"(?<=[{LATIN}]{{2}}):(?=[{LATIN_UPPER}][{LATIN_LOWER}])")
_GLUED_PUNCTUATION = (_GLUED_SENTENCE, _GLUED_COMMA, _GLUED_SEMICOLON, _GLUED_COLON)
# A space before punctuation that belongs to the word before it: "spectacle , and", "et al .
# found", "real ; the only", "Abstract : This" (in run 3's kept works: commas in 13,324, full
# stops in 15,696, semicolons in 2,416, colons in 6,701), and spaces just inside brackets, "( 2010
# )" and "( N = 610)" (5,175 and 6,346). A full stop keeps its space unless a space, a closing mark
# or the end follows, so ".86" and spaced ellipses (". . .") stay, and a colon between digits
# keeps it ("1 : 2"). Question and exclamation marks are left alone: after a space they are mostly
# a character the source lost (a micro sign, a pound sign, a dash) or OCR for "l" ("!arge").
_WORD_END = r"[^\W_]|[\)\]\"'”’%]"
_SPACES_BEFORE_PUNCTUATION = (
    re.compile(rf"(?<={_WORD_END}) +(?=[,;])"),
    re.compile(rf"(?<={_WORD_END}) +(?=\.(?:$|[\"'”’\)\]]| (?!\.)))"),
    re.compile(r"(?<=[^\W\d_]|[\)\]\"'”’%]) +(?=:)"),
    re.compile(r"(?<=[0-9]) +(?=:(?! *[0-9]))"),
    re.compile(rf"(?<={_WORD_END}|\.) +(?=[\)\]])"),
    re.compile(r"(?<=[\(\[]) +(?=[^\W_])"),
)
# Punctuation around a web or e-mail address, stripped before spaCy's tests read it: ``like_url``
# rejects anything that ends in a full stop.
_ADDRESS_EDGES = "\"'\u201c\u201d\u2018\u2019\u00ab\u00bb()[]{}<>,;:.!?"
# The original generic domains, which end addresses written with capitals ("Berkeley.Edu") and
# never begin an English sentence; spaCy's longer list holds country codes such as "in" and "it".
_GENERIC_DOMAINS = frozenset({"com", "org", "net", "edu", "gov", "int", "mil", "info"})
# The end of an HTML entity, whole or with its ampersand lost ("&rsquo;s", "Hegelrsquo;s"), whose
# semicolon is part of it.
_ENTITY_END = re.compile(r"(?:&#?[A-Za-z0-9]+|[lr][sd]quo|dquo|nbsp|[nm]dash|hellip)$")


def repair_text(raw: str) -> str:
    """The abstract without extraction artefacts, in Unicode normal form NFKC.

    Line breaks, tabs, zero-width spaces and replacement characters become spaces; other control
    and format characters (category Cc or Cf, such as the byte-order mark or the zero-width
    joiner) are removed; soft hyphens rejoin the word they split; runs of whitespace collapse to
    one space. NFKC also expands ligatures (*ﬁ* to *fi*), maps full-width punctuation and
    no-break spaces to their plain forms and writes superscripts as plain digits; the Unicode
    hyphen characters become the ASCII hyphen-minus. A space before a comma, full stop, semicolon
    or colon that belongs to the word before it goes ("et al . found" to "et al. found"), as do
    spaces just inside brackets. Then punctuation glued to the next word gets the missing space:
    sentence punctuation before a new sentence ("rare.It begins" to "rare. It begins"), commas
    and semicolons before a word, colons before a capitalised word; never inside web and e-mail
    addresses or HTML entities.
    """
    text = _SOFT_HYPHEN.sub("", _LITERAL_NEWLINE.sub(" ", raw))
    kept = []
    for character in text:
        if character in _WHITESPACE_CONTROLS or character in _WORD_SEPARATORS:
            kept.append(" ")
        elif unicodedata.category(character) not in ("Cc", "Cf"):
            kept.append(character)
    normal = unicodedata.normalize("NFKC", "".join(kept)).translate(_HYPHENS)
    text = _SPACES.sub(" ", normal).strip()
    for pattern in _SPACES_BEFORE_PUNCTUATION:
        text = pattern.sub("", text)
    for pattern in _GLUED_PUNCTUATION:
        text = _add_missing_spaces(pattern, text)
    return text


def _add_missing_spaces(pattern: re.Pattern[str], text: str) -> str:
    def spaced(match: re.Match[str]) -> str:
        if inside_address(text, match.start(), match.end()):
            return match.group()
        if _ENTITY_END.search(text, 0, match.start()):
            return match.group()
        return match.group() + " "

    return pattern.sub(spaced, text)


def inside_address(text: str, start: int, end: int) -> bool:
    """Whether the punctuation or word at ``start:end`` lies inside a web or e-mail address.

    It does when the run of text between spaces around it is an address, unless the address ends
    there: the text before it is a whole address (not just a scheme or "www") and a plain word
    follows that is no generic domain ("www.comphistsem.org).The best" ends before "The"; 26
    removals in 21 of run 4's kept works took such a word with the address). So "www.Nature.com",
    "http://www.ArsDisputandi" and "TTyler@Garnet.Berkeley.Edu" stay whole.
    """
    left = text.rfind(" ", 0, start) + 1
    right = text.find(" ", end)
    stop = len(text) if right == -1 else right
    chunk = text[left:stop].strip(_ADDRESS_EDGES)
    if not (like_url(chunk) or like_email(chunk)):
        return False
    before = text[left:start].strip(_ADDRESS_EDGES)
    after = text[end:stop].strip(_ADDRESS_EDGES)
    ends_here = (
        after.isalpha()
        and after.lower() not in _GENERIC_DOMAINS
        and not before.lower().endswith("www")
        and (like_url(before) or like_email(before))
    )
    return not ends_here


def compile_signatures(patterns: Sequence[str]) -> re.Pattern[str]:
    """One regular expression matching any of the given mojibake signature patterns."""
    return re.compile("|".join(f"(?:{pattern})" for pattern in patterns))


def has_mojibake(raw: str, signatures: re.Pattern[str]) -> bool:
    """Whether the raw text shows a mis-decoding signature, such as *Ã©* for *é*."""
    return signatures.search(raw) is not None
