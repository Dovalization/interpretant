"""The spaCy pass's sentences: each cached document's sentence ranges, per part.

One row per sentence, in the cache's order: the work and the sentence's character range in the
repaired text. The language stage reads these with the repaired texts instead of deserialising
the DocBins (docs/pipeline_review.md, C5). A part is written whole or not at all.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.parts import part_path

SENTENCES_SCHEMA = pl.Schema({"work_id": pl.String(), "start": pl.Int64(), "end": pl.Int64()})
_SUFFIX = ".parquet"


def save_sentences(catalog: CatalogConfig, sentences: pl.DataFrame, part: str) -> Path:
    """Write one part's sentences, replacing the part.

    Raises:
        ValueError: if the part name is not safe as a file name, or the columns are not the
            sentence columns.
    """
    if sentences.schema != SENTENCES_SCHEMA:
        raise ValueError(f"expected the sentence columns, got {sentences.schema}")
    target = part_path(catalog.sentences.path, part, _SUFFIX)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(f"{_SUFFIX}.partial")
    sentences.write_parquet(staging)
    staging.replace(target)
    return target


def load_sentences(catalog: CatalogConfig, part: str) -> pl.DataFrame:
    """One part's sentences, in the cache's order.

    Raises:
        ValueError: if the part name is not safe as a file name.
        FileNotFoundError: if the part has not been annotated.
    """
    path = part_path(catalog.sentences.path, part, _SUFFIX)
    if not path.exists():
        raise FileNotFoundError(f"no sentences for part {part!r} at {path}")
    return pl.read_parquet(path)
