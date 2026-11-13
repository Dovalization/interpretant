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
        ("the ﬁrst efﬂuent", "the first effluent"),
        ("full，width： colon", "full,width: colon"),
        ("no break", "no break"),
        ("m² and CO₂", "m2 and CO2"),
        ("  padded   with   spaces  ", "padded with spaces"),
    ],
    ids=["line breaks and tabs", "literal backslash-n", "soft hyphens", "zero-width space",
         "format characters", "replacement character", "ligatures", "full-width punctuation",
         "no-break space", "superscripts and subscripts", "runs of spaces"],
)  # fmt: skip
def test_repair_text_undoes_extraction_artefacts(raw: str, repaired: str) -> None:
    assert repair_text(raw) == repaired


def test_repair_text_leaves_clean_prose_unchanged() -> None:
    prose = "Attention mechanisms (Vaswani et al., 2017) improved translation — by 2.1 BLEU."
    assert repair_text(prose) == prose


@pytest.mark.parametrize("raw", ["cafÃ© society", "the authorâ€™s view", "priceÂ£ 5"])
def test_has_mojibake_finds_mis_decoded_text(raw: str) -> None:
    assert has_mojibake(raw, SIGNATURES)


def test_has_mojibake_ignores_correct_accents_and_quotes() -> None:
    assert not has_mojibake("café society, the author’s view, Ångström", SIGNATURES)
