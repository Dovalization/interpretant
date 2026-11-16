"""The split words table: every broken word of the corpus, its counts and whether it is joined.

Written by the count-word-repairs stage and read by the spaCy pass, which joins the breaks marked
``join`` or ``join_hyphenated`` before annotating (decision agenda 2.3, run 7c); a break marked
``kept_apart`` is on the curated list of real phrases. One file, written whole or not at all.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from interpretant.catalog.config import CatalogConfig

SPLIT_WORDS_SCHEMA = pl.Schema(
    {
        "word": pl.String(),
        "left": pl.String(),
        "right": pl.String(),
        "left_is_word": pl.Boolean(),
        "right_is_word": pl.Boolean(),
        "joined": pl.Int64(),
        "space": pl.Int64(),
        "hyphen_space": pl.Int64(),
        "hyphen": pl.Int64(),
        "kept_apart": pl.Boolean(),
        "join": pl.Boolean(),
        "join_hyphenated": pl.Boolean(),
    }
)


def save_split_words(catalog: CatalogConfig, split_words: pl.DataFrame) -> Path:
    """Write the split words table, replacing it.

    Raises:
        ValueError: if the columns are not the split-word columns.
    """
    if split_words.schema != SPLIT_WORDS_SCHEMA:
        raise ValueError(f"expected the split-word columns, got {split_words.schema}")
    target = catalog.split_words.path
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(".parquet.partial")
    split_words.write_parquet(staging)
    staging.replace(target)
    return target


def load_split_words(catalog: CatalogConfig) -> pl.DataFrame:
    """The split words table.

    Raises:
        FileNotFoundError: if the split words have not been counted.
    """
    path = catalog.split_words.path
    if not path.exists():
        raise FileNotFoundError(
            f"no split words table at {path}; run `interpretant count-word-repairs` first"
        )
    return pl.read_parquet(path)
