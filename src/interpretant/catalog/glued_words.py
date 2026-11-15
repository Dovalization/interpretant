"""The glued words table: every glued word of the corpus, its counts and whether it is split.

Written by the count-glued-words stage and read by the spaCy pass, which splits the words marked
``split`` before annotating (decision agenda 2.3, run 7). One file, written whole or not at all.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from interpretant.catalog.config import CatalogConfig

GLUED_WORDS_SCHEMA = pl.Schema(
    {
        "word": pl.String(),
        "left": pl.String(),
        "right": pl.String(),
        "works": pl.Int64(),
        "occurrences": pl.Int64(),
        "joined": pl.Int64(),
        "apart": pl.Int64(),
        "left_alone": pl.Int64(),
        "opener_rank": pl.Int64(),
        "reason": pl.String(),
        "split": pl.Boolean(),
    }
)


def save_glued_words(catalog: CatalogConfig, glued_words: pl.DataFrame) -> Path:
    """Write the glued words table, replacing it.

    Raises:
        ValueError: if the columns are not the glued-word columns.
    """
    if glued_words.schema != GLUED_WORDS_SCHEMA:
        raise ValueError(f"expected the glued-word columns, got {glued_words.schema}")
    target = catalog.glued_words.path
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(".parquet.partial")
    glued_words.write_parquet(staging)
    staging.replace(target)
    return target


def load_glued_words(catalog: CatalogConfig) -> pl.DataFrame:
    """The glued words table.

    Raises:
        FileNotFoundError: if the glued words have not been counted.
    """
    path = catalog.glued_words.path
    if not path.exists():
        raise FileNotFoundError(
            f"no glued words table at {path}; run `interpretant count-glued-words` first"
        )
    return pl.read_parquet(path)
