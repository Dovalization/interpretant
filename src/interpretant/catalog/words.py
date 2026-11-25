"""The word table: the words the project studies, each with the readings counted as its uses.

`curated/words.tsv` (decision agenda 4.7): one row per development word, a lemma (a phrase target's
words joined by a space), the parts of speech whose uses are in scope, and whether the word is
matched by its lemma or only by its form (*having*). The words held out for the confirming run
(6.2) are kept apart in `curated/held_out.tsv`, tracked by DVC and never by git, with the same
columns plus the sealed name a committed output shows for each (`docs/held_out.md`).
A word lists a set of parts of speech, not one: the small tagger reads many noun uses as other
parts of speech (*learning* in "the learning process" as a verb, *Transformer* as a name), so
each noun accepts `PROPN`, with real names left to entity recognition, plus the extra tags its
data shows to be tagger errors (the author, 2026-11-17). Tab-separated, after `#` comment lines.
"""

from __future__ import annotations

import csv
import itertools
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from interpretant.catalog.config import CatalogConfig

COLUMNS = ["lemma", "pos", "match"]
HELD_OUT_COLUMNS = [*COLUMNS, "sealed_name"]
_SEALED_NAME = re.compile(r"held-out:[0-9a-f]{12}")
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
    sealed_name: str | None = None

    @field_validator("lemma")
    @classmethod
    def _lemma_is_lowercase_words(cls, lemma: str) -> str:
        if not _LEMMA.fullmatch(lemma):
            raise ValueError(f"lemma {lemma!r} is not lowercase letters, spaces and hyphens")
        return lemma

    @model_validator(mode="after")
    def _held_out_words_have_a_sealed_name(self) -> Self:
        if self.held_out != (self.sealed_name is not None):
            raise ValueError("a held-out word, and only one, has a sealed name")
        if self.sealed_name is not None and not _SEALED_NAME.fullmatch(self.sealed_name):
            raise ValueError(
                f"a sealed name is held-out: and 12 hex digits, got {self.sealed_name!r}"
            )
        return self

    @property
    def is_phrase(self) -> bool:
        """Whether the word is a phrase target (several words)."""
        return " " in self.lemma

    @property
    def reported_name(self) -> str:
        """The name a committed output may show: the lemma, or a held-out word's sealed name.

        Held-out words are sealed until the confirmation plan (agenda 6.2): outputs show a random
        name kept beside the word in the held-out file, so they can still be joined back to it but
        can't be read by hashing candidate words.
        """
        return self.sealed_name if self.sealed_name is not None else self.lemma


def load_words(catalog: CatalogConfig) -> list[Word]:
    """Read the word table, development words first, then the held-out words, each in its order.

    Raises:
        FileNotFoundError: if either file is missing; the held-out file comes from DVC.
        ValueError: if a file's header isn't its columns, or a lemma appears twice.
        pydantic.ValidationError: if a row is malformed (an unknown tag, an uppercase lemma, a
            malformed sealed name).
    """
    held_out_path = catalog.held_out_words.path
    if not held_out_path.exists():
        raise FileNotFoundError(f"{held_out_path} is missing: it is tracked by DVC (dvc pull)")
    development = [_word(row, held_out=False) for row in _read_rows(catalog.words.path, COLUMNS)]
    held_out = [_word(row, held_out=True) for row in _read_rows(held_out_path, HELD_OUT_COLUMNS)]
    words = development + held_out
    seen: set[str] = set()
    for word in words:
        if word.lemma in seen:
            raise ValueError(f"the word {word.lemma!r} appears twice")
        seen.add(word.lemma)
    return words


def _read_rows(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        lines = itertools.dropwhile(lambda line: line.startswith("#"), handle)
        rows = csv.reader(lines, delimiter="\t")
        header = next(rows, None)
        if header != columns:
            raise ValueError(f"{path.name}: expected the columns {columns}, got {header}")
        return [dict(zip(columns, row, strict=True)) for row in rows]


def _word(row: dict[str, str], *, held_out: bool) -> Word:
    return Word.model_validate(
        {
            "lemma": row["lemma"],
            "pos": tuple(row["pos"].split("|")),
            "match": row["match"],
            "held_out": held_out,
            "sealed_name": row.get("sealed_name"),
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
