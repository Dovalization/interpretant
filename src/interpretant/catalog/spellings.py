"""The spelling map: each spelling variant and the standard spelling it is written as.

Built by ``scripts/review/score_spellings.py`` from the audited proposal (decision agenda 3.4;
``docs/spellings.md``) and applied to each hyphen segment of the static view's tokens
(``corpus/spelling.py``). Tab-separated, after a header of ``#`` comment lines.
"""

from __future__ import annotations

import csv
import itertools
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from interpretant.catalog.config import CatalogConfig

COLUMNS = ["form", "standard", "source"]


class Spelling(BaseModel):
    """One spelling variant, its standard, and where the pair comes from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Both are folded words as the static view writes them: lowercase letters only, so a pair never
    # splits or joins a token and the view's offsets stay as they are.
    form: str = Field(pattern=r"^[a-z]+$")
    standard: str = Field(pattern=r"^[a-z]+$")
    source: Literal["norm_table", "rules"]


def load_spellings(catalog: CatalogConfig) -> list[Spelling]:
    """Read the spelling map, in its order.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the header is not form, standard, source; a form repeats or is its own
            standard; or a standard is also a form, which would make the map depend on the order
            it is applied in.
        pydantic.ValidationError: if a row is malformed (an uppercase letter, a space, a hyphen).
    """
    with catalog.spellings.path.open(encoding="utf-8", newline="") as handle:
        lines = itertools.dropwhile(lambda line: line.startswith("#"), handle)
        rows = csv.reader(lines, delimiter="\t")
        header = next(rows, None)
        if header != COLUMNS:
            raise ValueError(f"expected the columns {COLUMNS}, got {header}")
        spellings = [Spelling(**dict(zip(COLUMNS, row, strict=True))) for row in rows]
    forms: set[str] = set()
    for spelling in spellings:
        if spelling.form in forms:
            raise ValueError(f"the form {spelling.form!r} appears twice")
        if spelling.form == spelling.standard:
            raise ValueError(f"the form {spelling.form!r} is its own standard")
        forms.add(spelling.form)
    chained = sorted(spelling.standard for spelling in spellings if spelling.standard in forms)
    if chained:
        raise ValueError(f"standards that are also forms: {chained}")
    return spellings
