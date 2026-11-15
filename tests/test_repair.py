"""Repair before spaCy."""

from __future__ import annotations

import pytest

from interpretant.corpus.repair import compile_signatures, has_mojibake, repair_text

SIGNATURES = compile_signatures([r"Ã[\x80-\xbf]", "â€", r"Â[\xa0-\xbf]"])


@pytest.mark.parametrize(
    ("raw", "repaired"),
    [
        ("line one\nline two\ttabbed", "line one line two tabbed"),
        ("literal\\nnewline", "literal newline"),
        ("exam­ ple and inter­nal", "example and internal"),
        ("zero​width space", "zero width space"),
        ("byte﻿order and join‍er", "byteorder and joiner"),
        ("lost � character", "lost character"),
        ("post\u2010training and non\u2011breaking", "post-training and non-breaking"),
        ("the ﬁrst efﬂuent", "the first effluent"),
        ("full， width： colon", "full, width: colon"),
        ("no break", "no break"),
        ("m² and CO₂", "m2 and CO2"),
        ("  padded   with   spaces  ", "padded with spaces"),
    ],
    ids=["line breaks and tabs", "literal backslash-n", "soft hyphens", "zero-width space",
         "format characters", "replacement character", "unicode hyphens", "ligatures",
         "full-width punctuation", "no-break space", "superscripts and subscripts",
         "runs of spaces"],
)  # fmt: skip
def test_repair_text_undoes_extraction_artefacts(raw: str, repaired: str) -> None:
    assert repair_text(raw) == repaired


@pytest.mark.parametrize(
    ("raw", "repaired"),
    [
        ("Narcolepsy is very rare.It begins", "Narcolepsy is very rare. It begins"),
        ("in 1950.French delegates", "in 1950. French delegates"),
        ("(r = 0.922).In addition", "(r = 0.922). In addition"),
        ('he learns."Acquaintance follows', 'he learns." Acquaintance follows'),
        ("the night.“Fading” light", "the night. “Fading” light"),
        ("Is it rare?Yes, and", "Is it rare? Yes, and"),
        ("In this paper,the results", "In this paper, the results"),
        ("the carotid body,spleen,intestine", "the carotid body, spleen, intestine"),
        ("the teaching form;comprehensively", "the teaching form; comprehensively"),
        ("(Ragni, 2005;Ragni, 2007)", "(Ragni, 2005; Ragni, 2007)"),
        ("Objective:To study", "Objective: To study"),
        ("see www.comphistsem.org).The next", "see www.comphistsem.org). The next"),
        ("see (www.inamori-f.or.jp.Given) now", "see (www.inamori-f.or.jp. Given) now"),
    ],
    ids=["after a letter", "after a digit", "after a bracket", "closing quote",
         "opening quote", "question mark", "comma", "comma list", "semicolon",
         "semicolon after a year", "colon before a capital", "web address ending a sentence",
         "bracketed web address ending a sentence"],
)  # fmt: skip
def test_repair_text_spaces_punctuation_glued_to_the_next_word(raw: str, repaired: str) -> None:
    assert repair_text(raw) == repaired


@pytest.mark.parametrize(
    ("raw", "repaired"),
    [
        ("a public spectacle , and it", "a public spectacle, and it"),
        ("Nakahata et al . found that", "Nakahata et al. found that"),
        ("on reading Jerusalem . For him", "on reading Jerusalem. For him"),
        ("the end of the essay .", "the end of the essay."),
        ("called real ; the only", "called real; the only"),
        ("Abstract : This paper", "Abstract: This paper"),
        ("Abusch's ( 2010 ) account", "Abusch's (2010) account"),
        ("in this technology ,this article", "in this technology, this article"),
    ],
    ids=["comma", "full stop before a word", "full stop before a sentence", "final full stop",
         "semicolon", "colon", "inside brackets", "space before and none after"],
)  # fmt: skip
def test_repair_text_removes_the_space_before_punctuation(raw: str, repaired: str) -> None:
    assert repair_text(raw) == repaired


@pytest.mark.parametrize(
    "text",
    ["IQ, .86 for performance", "they endeavour . . . to help", "Contents . . . . 12",
     "a ratio of 1 : 2", "a ?yes? bias", "in us ?Or how", "found !that waves"],
    ids=["decimal", "spaced ellipsis", "leader dots", "ratio", "lost characters",
         "question mark", "exclamation mark"],
)  # fmt: skip
def test_repair_text_keeps_the_space_where_it_belongs(text: str) -> None:
    assert repair_text(text) == text


@pytest.mark.parametrize(
    "text",
    ["the U.S.Army", "E.coli strains", "a 3.5 mg dose", "three weeks.4 Just prior",
     "e.g.the model", "see www.Nature.com today", "at http://Wiley.Biolucida.net/Atlas now",
     "at TTyler@Garnet.Berkeley.Edu. We", "[http://www.ArsDisputandi.org]",
     "write to J.Smith@uni.edu now", "see 10.1234/abc.Def-5 now",
     "N,N-dimethyl", "the (i,j)th entry", "a 1,000 sample", "the light:dark cycle",
     "at 10:30 hours", "[!x:Gx]", "the author&rsquo;s view", "Hegelrsquo;s system"],
    ids=["initials", "species", "decimal", "footnote", "lowercase next word", "web address",
         "address with a scheme", "e-mail address with a capitalised domain",
         "address after www", "e-mail address", "doi",
         "comma between single letters", "indices", "thousands", "ratio", "time",
         "colon after one letter", "html entity", "html entity without its ampersand"],
)  # fmt: skip
def test_repair_text_leaves_other_full_stops_as_they_are(text: str) -> None:
    assert repair_text(text) == text


def test_repair_text_leaves_clean_prose_unchanged() -> None:
    prose = "Attention mechanisms (Vaswani et al., 2017) improved translation — by 2.1 BLEU."
    assert repair_text(prose) == prose


@pytest.mark.parametrize("raw", ["cafÃ© society", "the authorâ€™s view", "priceÂ£ 5"])
def test_has_mojibake_finds_mis_decoded_text(raw: str) -> None:
    assert has_mojibake(raw, SIGNATURES)


def test_has_mojibake_ignores_correct_accents_and_quotes() -> None:
    assert not has_mojibake("café society, the author’s view, Ångström", SIGNATURES)
