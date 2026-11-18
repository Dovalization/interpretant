"""The compound map: each spelling of a two-part compound and the form it is written as.

Built by ``scripts/review/score_compounds.py`` from the audited proposal (decision agenda 3.4;
``docs/compounds.md``) and applied by the static view (``corpus/compound_spelling.py``): a form
with a space is a spaced pair, merged only as a noun-noun compound; any other form is one token,
rewritten whole. Tab-separated, after a header of ``#`` comment lines.
"""

from __future__ import annotations

import csv
import itertools
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from interpretant.catalog.config import CatalogConfig

COLUMNS = ["form", "standard", "source"]
# Two lowercase letter parts joined by a hyphen or a space, or one closed word.
_FORM = r"^[a-z]+(?:[- ][a-z]+)?$"
_JOINT = re.compile(r"[- ]")


class Compound(BaseModel):
    """One spelling of a compound, its standard, and why that standard won.

    The standard is the form the corpus uses most often (``frequency``), or a word-table lemma,
    which keeps its own spelling (``word_table``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    form: str = Field(pattern=_FORM)
    standard: str = Field(pattern=_FORM)
    source: Literal["frequency", "word_table"]

    @model_validator(mode="after")
    def _one_compound(self) -> Compound:
        if self.form == self.standard:
            raise ValueError(f"the form {self.form!r} is its own standard")
        if _letters(self.form) != _letters(self.standard):
            raise ValueError(f"{self.form!r} and {self.standard!r} have different letters")
        parts = [_JOINT.split(text) for text in (self.form, self.standard)]
        if all(len(part) == 2 for part in parts) and parts[0] != parts[1]:
            raise ValueError(f"{self.form!r} and {self.standard!r} split at different points")
        return self

    @property
    def is_spaced(self) -> bool:
        """Whether the form is a spaced pair of tokens."""
        return " " in self.form


def load_compounds(catalog: CatalogConfig) -> list[Compound]:
    """Read the compound map, in its order.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the header is not form, standard, source; a form repeats; a standard is
            also a form (a chain); one compound has two standards; or a word of a spaced form is
            itself a one-token form, so the two passes of the view would depend on their order.
        pydantic.ValidationError: if a row is malformed (three parts, a digit, two forms of
            different letters).
    """
    with catalog.compounds.path.open(encoding="utf-8", newline="") as handle:
        lines = itertools.dropwhile(lambda line: line.startswith("#"), handle)
        rows = csv.reader(lines, delimiter="\t")
        header = next(rows, None)
        if header != COLUMNS:
            raise ValueError(f"expected the columns {COLUMNS}, got {header}")
        compounds = [Compound(**dict(zip(COLUMNS, row, strict=True))) for row in rows]
    forms: set[str] = set()
    standards: dict[str, str] = {}
    for compound in compounds:
        if compound.form in forms:
            raise ValueError(f"the form {compound.form!r} appears twice")
        forms.add(compound.form)
        key = _letters(compound.form)
        if standards.setdefault(key, compound.standard) != compound.standard:
            raise ValueError(f"the compound {key!r} has two standards")
    chained = sorted(compound.standard for compound in compounds if compound.standard in forms)
    if chained:
        raise ValueError(f"standards that are also forms: {chained}")
    one_token = {compound.form for compound in compounds if not compound.is_spaced}
    crossing = sorted(
        word
        for compound in compounds
        if compound.is_spaced
        for word in compound.form.split(" ")
        if word in one_token
    )
    if crossing:
        raise ValueError(f"words of spaced forms that are also one-token forms: {crossing}")
    return compounds


def _letters(form: str) -> str:
    return _JOINT.sub("", form)
