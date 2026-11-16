"""The lower-case glued words table: every candidate glued word, its counts and its decision.

Written by the count-word-repairs stage, which splits the words marked ``split`` before the spaCy
pass (decision agenda 2.3, run 8); a word marked ``kept_joined`` is on the curated list. Rows of
the pattern ``hyphenated_chunk`` give a hyphenated chunk's separator for the glued word it holds.
One file, written whole or not at all.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from interpretant.catalog.config import CatalogConfig

LOWER_GLUED_WORDS_SCHEMA = pl.Schema(
    {
        "word": pl.String(),
        "pattern": pl.String(),
        "parts": pl.List(pl.String()),
        "separators": pl.List(pl.String()),
        "joined": pl.Int64(),
        "apart": pl.Int64(),
        "uses": pl.Int64(),
        "capitalised_uses": pl.Int64(),
        "kept_joined": pl.Boolean(),
        "split": pl.Boolean(),
    }
)


def save_lower_glued_words(catalog: CatalogConfig, lower_glued_words: pl.DataFrame) -> Path:
    """Write the lower-case glued words table, replacing it.

    Raises:
        ValueError: if the columns are not the lower-case glued-word columns.
    """
    if lower_glued_words.schema != LOWER_GLUED_WORDS_SCHEMA:
        raise ValueError(
            f"expected the lower-case glued-word columns, got {lower_glued_words.schema}"
        )
    target = catalog.lower_glued_words.path
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(".parquet.partial")
    lower_glued_words.write_parquet(staging)
    staging.replace(target)
    return target


def load_lower_glued_words(catalog: CatalogConfig) -> pl.DataFrame:
    """The lower-case glued words table.

    Raises:
        FileNotFoundError: if the lower-case glued words have not been counted.
    """
    path = catalog.lower_glued_words.path
    if not path.exists():
        raise FileNotFoundError(
            f"no lower-case glued words table at {path}; "
            "run `interpretant count-word-repairs` first"
        )
    return pl.read_parquet(path)
