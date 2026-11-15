"""Repair before spaCy: undo extraction artefacts so the stored text is clean, then never edit it.

The decision agenda, step 2.3: HTML character references, NFKC, control and format characters by
Unicode category, soft hyphens, the space before punctuation and the missing space after it;
abstracts with mojibake are dropped, not repaired, so detection reads the raw text.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Sequence
from html.entities import html5

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
# A DOI, bare or after "doi:", which spaCy's URL test does not know ("10.21437/SpeechProsody.2020-
# 122"); the spaCy pass's tokenizer keeps one as a single token with the same pattern.
DOI_PATTERN = r"^(?:doi:)?10\.\d{4,9}/\S+$"
_DOI = re.compile(DOI_PATTERN)
# HTML character references left by an upstream extractor (131 works in run 7): numeric ones and
# HTML5 names with their semicolon, decoded as the standard library reads them, and names whose
# ampersand was lost ("Hegelrsquo;s", "ldquo;bad faithrdquo;", "casenbsp;study"). Two are no HTML
# at all: SAGE's "&dquo;" for a straight double quote (406 in 95 works) and "&s;" for an
# apostrophe ("Ruisdael&s;s", 20 in 10). A name HTML5 lacks stays as it is ("&subd;"; checked
# against the table, since ``html.unescape`` reads "&copyright;" as "&copy" and "right;"), and so
# does "amp;" without its ampersand, which mostly ends a word ("camp;", "Kamp;").
_CHARACTER_REFERENCE = re.compile(
    r"&(?:#[0-9]{1,7}|#[xX][0-9A-Fa-f]{1,6}|[A-Za-z][A-Za-z0-9]{0,31});"
)
_LOST_AMPERSAND = re.compile(r"(?:[lr][sd]quo|[nm]dash|nbsp|hellip);")
_NONSTANDARD_REFERENCES = {"&dquo;": '"', "&s;": "'"}
# The end of an HTML entity left undecoded, whose semicolon is part of it.
_ENTITY_END = re.compile(r"&#?[A-Za-z0-9]+$")


def repair_text(raw: str) -> str:
    """The abstract without extraction artefacts, in Unicode normal form NFKC.

    HTML character references are decoded first ("&rsquo;" and "rsquo;" to "’"). Line breaks,
    tabs, zero-width spaces and replacement characters become spaces; other control and format
    characters (category Cc or Cf, such as the byte-order mark or the zero-width joiner) are
    removed; soft hyphens rejoin the word they split; runs of whitespace collapse to
    one space. NFKC also expands ligatures (*ﬁ* to *fi*), maps full-width punctuation and
    no-break spaces to their plain forms and writes superscripts as plain digits; the Unicode
    hyphen characters become the ASCII hyphen-minus. A space before a comma, full stop, semicolon
    or colon that belongs to the word before it goes ("et al . found" to "et al. found"), as do
    spaces just inside brackets. Then punctuation glued to the next word gets the missing space:
    sentence punctuation before a new sentence ("rare.It begins" to "rare. It begins"), commas
    and semicolons before a word, colons before a capitalised word; never inside web and e-mail
    addresses or HTML entities.
    """
    text = _SOFT_HYPHEN.sub("", _LITERAL_NEWLINE.sub(" ", _decode_references(raw)))
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


def _decode_references(text: str) -> str:
    """The text with its HTML character references decoded, as far as they are known."""

    def decoded(match: re.Match[str]) -> str:
        reference = match.group()
        if reference in _NONSTANDARD_REFERENCES:
            return _NONSTANDARD_REFERENCES[reference]
        if reference.startswith("&#") or reference[1:] in html5:
            return html.unescape(reference)
        return reference

    text = _CHARACTER_REFERENCE.sub(decoded, text)
    return _LOST_AMPERSAND.sub(lambda match: html5[match.group()], text)


def _add_missing_spaces(pattern: re.Pattern[str], text: str) -> str:
    def spaced(match: re.Match[str]) -> str:
        if inside_address(text, match.start(), match.end()):
            return match.group()
        if _ENTITY_END.search(text, 0, match.start()):
            return match.group()
        return match.group() + " "

    return pattern.sub(spaced, text)


def inside_address(text: str, start: int, end: int) -> bool:
    """Whether the punctuation or word at ``start:end`` lies inside a web or e-mail address or DOI.

    It does when the run of text between spaces around it is an address, unless the address ends
    there: the text before it is a whole address (not just a scheme or "www") and a plain word
    follows that is no generic domain ("www.comphistsem.org).The best" ends before "The"; 26
    removals in 21 of run 4's kept works took such a word with the address). So "www.Nature.com",
    "http://www.ArsDisputandi", "TTyler@Garnet.Berkeley.Edu" and "10.21437/SpeechProsody.2020-122"
    stay whole.
    """
    left = text.rfind(" ", 0, start) + 1
    right = text.find(" ", end)
    stop = len(text) if right == -1 else right
    if not _is_address(text[left:stop].strip(_ADDRESS_EDGES)):
        return False
    before = text[left:start].strip(_ADDRESS_EDGES)
    after = text[end:stop].strip(_ADDRESS_EDGES)
    ends_here = (
        after.isalpha()
        and after.lower() not in _GENERIC_DOMAINS
        and not before.lower().endswith("www")
        and _is_address(before)
    )
    return not ends_here


def _is_address(text: str) -> bool:
    """Whether the text is a web or e-mail address or a DOI."""
    return like_url(text) or like_email(text) or _DOI.match(text) is not None


def compile_signatures(patterns: Sequence[str]) -> re.Pattern[str]:
    """One regular expression matching any of the given mojibake signature patterns."""
    return re.compile("|".join(f"(?:{pattern})" for pattern in patterns))


def has_mojibake(raw: str, signatures: re.Pattern[str]) -> bool:
    """Whether the raw text shows a mis-decoding signature, such as *Ã©* for *é*."""
    return signatures.search(raw) is not None
