"""How often given word forms appear in the corpus's texts, counted with Polars.

The word repairs (decision agenda 2.3) decide from such counts: a form is looked for between
spaces after case folding, with every character other than a letter, digit, white space or hyphen
read as a space, so *in vestigation*, *self-attention* and *of the* are each one form, and a
hyphen keeps two words together as written.
"""

from __future__ import annotations

import polars as pl

# In Polars' syntax (the regex crate): the characters read as a space, and runs of white space.
_NOT_TEXT_PATTERN = r"[^\p{L}\p{N}\s-]+"
_SPACES_PATTERN = r"\s+"


def count_forms(texts: pl.DataFrame, forms: pl.Series) -> pl.DataFrame:
    """How often each form appears between spaces, case folded, other characters as spaces.

    ``texts`` has a ``text`` column; the result has a ``form`` and a ``count`` column, for the
    forms found only (none for no forms). One lazy query, which Polars runs in parallel (as
    ``Series`` methods it ran on one core).
    """
    normalised = (
        " "
        + pl.col("text")
        .str.to_lowercase()
        .str.replace_all(_NOT_TEXT_PATTERN, " ")
        .str.replace_all(_SPACES_PATTERN, " ")
        + " "
    )
    if forms.is_empty():
        return pl.DataFrame(schema={"form": pl.String(), "count": pl.UInt32()})
    patterns = (" " + forms.unique() + " ").to_list()
    return (
        texts.lazy()
        .select(form=normalised.str.extract_many(patterns, overlapping=True))
        .explode("form")
        .drop_nulls()
        .group_by(pl.col("form").str.strip_chars(" "))
        .len("count")
        .collect(engine="streaming")
    )
