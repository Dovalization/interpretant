"""The sentence languages: lingua's verdict on every sentence of the annotation cache, per part.

One row per sentence: the work, the sentence's character range and its language (an ISO 639-1
code, or ``und``) with lingua's confidence. A part is written whole or not at all.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.parts import part_path

SENTENCE_LANGUAGES_SCHEMA = pl.Schema(
    {
        "work_id": pl.String(),
        "start": pl.Int64(),
        "end": pl.Int64(),
        "language": pl.String(),
        "confidence": pl.Float64(),
    }
)
_SUFFIX = ".parquet"


def save_sentence_languages(catalog: CatalogConfig, languages: pl.DataFrame, part: str) -> Path:
    """Write one part's sentence languages, replacing the part.

    Raises:
        ValueError: if the part name is not safe as a file name, or the columns are not the
            sentence-language columns.
    """
    if languages.schema != SENTENCE_LANGUAGES_SCHEMA:
        raise ValueError(f"expected the sentence-language columns, got {languages.schema}")
    target = part_path(catalog.sentence_languages.path, part, _SUFFIX)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(f"{_SUFFIX}.partial")
    languages.write_parquet(staging)
    staging.replace(target)
    return target


def load_sentence_languages(catalog: CatalogConfig, part: str) -> pl.DataFrame:
    """One part's sentence languages.

    Raises:
        ValueError: if the part name is not safe as a file name.
        FileNotFoundError: if the part's languages have not been detected.
    """
    path = part_path(catalog.sentence_languages.path, part, _SUFFIX)
    if not path.exists():
        raise FileNotFoundError(f"no sentence languages for part {part!r} at {path}")
    return pl.read_parquet(path)
