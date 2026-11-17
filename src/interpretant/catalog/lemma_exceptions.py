"""The lemma exceptions file: the forms whose static-view lemma differs from spaCy's.

Built from the lemma review by ``scripts/evidence/lemma_exceptions.py`` (decision agenda 3.2,
3.3; ``docs/lemma_review.md``) and applied to the cached documents through spaCy's attribute
ruler (3.6, ``corpus/lemma_exceptions.py``). Tab-separated, written by the ``csv`` module (so a
form holding a quote is quoted), after a header of ``#`` comment lines.
"""

from __future__ import annotations

import csv
import itertools
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from interpretant.catalog.config import CatalogConfig

COLUMNS = ["form", "lemma", "tag", "source"]
Source = Literal["override", "adjective_rule", "author", "author_call", "agents", "hyphen_rule"]


class LemmaException(BaseModel):
    """One form's lemma under one reading, and where the decision comes from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # The token's text lowercased, accents not folded, as spaCy's LOWER attribute holds it.
    form: str = Field(min_length=1)
    lemma: str
    tag: Literal["NNS", "VBZ"]
    source: Source

    @field_validator("lemma")
    @classmethod
    def _lemma_has_a_letter(cls, lemma: str) -> str:
        # The static view drops tokens with no letter, so a lemma without one would drop a token
        # and shift the view against a view without the exceptions.
        if not any(character.isalpha() for character in lemma):
            raise ValueError(f"lemma {lemma!r} has no letter")
        return lemma


def load_lemma_exceptions(catalog: CatalogConfig) -> list[LemmaException]:
    """Read the lemma exceptions file, in its order.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the header is not form, lemma, tag, source, or a (form, tag) pair repeats.
        pydantic.ValidationError: if a row is malformed (an empty tag, a lemma with no letter).
    """
    with catalog.lemma_exceptions.path.open(encoding="utf-8", newline="") as handle:
        lines = itertools.dropwhile(lambda line: line.startswith("#"), handle)
        rows = csv.reader(lines, delimiter="\t")
        header = next(rows, None)
        if header != COLUMNS:
            raise ValueError(f"expected the columns {COLUMNS}, got {header}")
        exceptions = [LemmaException(**dict(zip(COLUMNS, row, strict=True))) for row in rows]
    seen: set[tuple[str, str]] = set()
    for exception in exceptions:
        key = (exception.form, exception.tag)
        if key in seen:
            raise ValueError(f"the form {exception.form!r} appears twice as {exception.tag}")
        seen.add(key)
    return exceptions
