"""The word table: the words the project studies, each with the readings counted as its uses.

`curated/words.tsv` (decision agenda 4.7): one row per word, a lemma (a phrase target's words
joined by a space), the parts of speech whose uses are in scope, whether the word is matched by
its lemma or only by its form (*having*), and whether it is held out for the confirming run (6.2).
A word lists a set of parts of speech, not one: the small tagger reads many noun uses as other
parts of speech (*learning* in "the learning process" as a verb, *Transformer* as a name), so
each noun accepts `PROPN`, with real names left to entity recognition, plus the extra tags its
data shows to be tagger errors (the author, 2026-11-17). Tab-separated, after `#` comment lines.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import re
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from interpretant.catalog.config import CatalogConfig

COLUMNS = ["lemma", "pos", "match", "held_out"]
# Universal Dependencies' part-of-speech tags, as spaCy's `pos_` gives them.
UniversalTag = Literal[
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]  # fmt: skip
_LEMMA = re.compile(r"[a-z]+(?:[ -][a-z]+)*")


class Word(BaseModel):
    """One word of the table."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lemma: str
    pos: tuple[UniversalTag, ...] = Field(min_length=1)
    match: Literal["lemma", "form"]
    held_out: bool

    @field_validator("lemma")
    @classmethod
    def _lemma_is_lowercase_words(cls, lemma: str) -> str:
        if not _LEMMA.fullmatch(lemma):
            raise ValueError(f"lemma {lemma!r} is not lowercase letters, spaces and hyphens")
        return lemma

    @property
    def is_phrase(self) -> bool:
        """Whether the word is a phrase target (several words)."""
        return " " in self.lemma

    @property
    def reported_name(self) -> str:
        """The name a committed output may show: the lemma, or a hash for a held-out word.

        Held-out words are sealed until the confirmation plan (agenda 6.2): never printed, only
        hashed, so outputs can still be joined back to the table.
        """
        if not self.held_out:
            return self.lemma
        return "held-out:" + hashlib.sha256(self.lemma.encode()).hexdigest()[:12]


def load_words(catalog: CatalogConfig) -> list[Word]:
    """Read the word table, in its order.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the header is not lemma, pos, match, held_out, a lemma repeats, or
            held_out is not yes or no.
        pydantic.ValidationError: if a row is malformed (an unknown tag, an uppercase lemma).
    """
    with catalog.words.path.open(encoding="utf-8", newline="") as handle:
        lines = itertools.dropwhile(lambda line: line.startswith("#"), handle)
        rows = csv.reader(lines, delimiter="\t")
        header = next(rows, None)
        if header != COLUMNS:
            raise ValueError(f"expected the columns {COLUMNS}, got {header}")
        words = [_word(dict(zip(COLUMNS, row, strict=True))) for row in rows]
    seen: set[str] = set()
    for word in words:
        if word.lemma in seen:
            raise ValueError(f"the word {word.lemma!r} appears twice")
        seen.add(word.lemma)
    return words


def _word(row: dict[str, str]) -> Word:
    held_out = {"yes": True, "no": False}.get(row["held_out"])
    if held_out is None:
        raise ValueError(f"held_out must be yes or no, got {row['held_out']!r}")
    return Word.model_validate(
        {
            "lemma": row["lemma"],
            "pos": tuple(row["pos"].split("|")),
            "match": row["match"],
            "held_out": held_out,
        }
    )


def seal_held_out(text: str, words: Iterable[Word]) -> str:
    """The text with each held-out word shown by its reported name, in any of its common forms.

    The forms are the lemma with -s, -es, -ed, -d, -ing or 's, and for a lemma ending in -e or -y
    those that drop it (-ing; -ies, -ied); case is ignored. Free text a development output keeps,
    such as a judge's note, is sealed this way before it is committed (`docs/held_out.md`).
    """
    names: dict[str, str] = {}
    for word in words:
        if word.held_out:
            for form in _common_forms(word.lemma):
                names[form] = word.reported_name
    if not names:
        return text
    alternatives = "|".join(re.escape(form) for form in sorted(names, key=len, reverse=True))
    pattern = re.compile(rf"\b(?:{alternatives})\b", re.IGNORECASE)
    return pattern.sub(lambda match: names[match.group(0).lower()], text)


def _common_forms(lemma: str) -> set[str]:
    forms = {lemma + ending for ending in ("", "s", "es", "ed", "d", "ing", "'s")}
    if lemma.endswith("e"):
        forms.add(lemma[:-1] + "ing")
    if lemma.endswith("y"):
        forms |= {lemma[:-1] + "ies", lemma[:-1] + "ied"}
    return {form.lower() for form in forms}
