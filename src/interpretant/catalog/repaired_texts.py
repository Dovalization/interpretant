"""The texts of the works, per part: as repair leaves them, and with every word repair made.

The repair-texts stage merges the cells and repairs each text (spacing around punctuation, HTML
entities, web addresses); its base texts are read by the word repairs. The count-word-repairs
stage makes the glued-word splits, the split-word joins and the lower-case splits (decision agenda
2.3); its repaired texts are read by the spaCy pass. Both are made once, in worker processes. One
row per work, in the part's order; a part is written whole or not at all, and a part with no
works is written empty.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.parts import part_path

REPAIRED_TEXTS_SCHEMA = pl.Schema({"work_id": pl.String(), "text": pl.String()})
_SUFFIX = ".parquet"


def save_base_texts(catalog: CatalogConfig, texts: pl.DataFrame, part: str) -> Path:
    """Write one part's base texts, replacing the part.

    Raises:
        ValueError: if the part name is not safe as a file name, or the columns are not the
            text columns.
    """
    return _save(catalog.base_texts.path, texts, part)


def load_base_texts(catalog: CatalogConfig, part: str) -> pl.DataFrame:
    """One part's base texts, in the part's order.

    Raises:
        ValueError: if the part name is not safe as a file name.
        FileNotFoundError: if the part's texts have not been repaired.
    """
    return pl.read_parquet(_existing_path(catalog.base_texts.path, part, "repair-texts"))


def save_repaired_texts(catalog: CatalogConfig, texts: pl.DataFrame, part: str) -> Path:
    """Write one part's repaired texts, replacing the part.

    Raises:
        ValueError: if the part name is not safe as a file name, or the columns are not the
            text columns.
    """
    return _save(catalog.repaired_texts.path, texts, part)


def load_repaired_texts(catalog: CatalogConfig, part: str) -> pl.DataFrame:
    """One part's repaired texts, in the part's order.

    Raises:
        ValueError: if the part name is not safe as a file name.
        FileNotFoundError: if the part's word repairs have not been made.
    """
    return pl.read_parquet(_existing_path(catalog.repaired_texts.path, part, "count-word-repairs"))


def count_repaired_texts(catalog: CatalogConfig, part: str) -> int:
    """How many repaired texts a part holds, read from the file's metadata.

    Raises:
        ValueError: if the part name is not safe as a file name.
        FileNotFoundError: if the part's word repairs have not been made.
    """
    path = _existing_path(catalog.repaired_texts.path, part, "count-word-repairs")
    count: int = pl.scan_parquet(path).select(pl.len()).collect().item()
    return count


def _save(directory: Path, texts: pl.DataFrame, part: str) -> Path:
    if texts.schema != REPAIRED_TEXTS_SCHEMA:
        raise ValueError(f"expected the text columns, got {texts.schema}")
    target = part_path(directory, part, _SUFFIX)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(f"{_SUFFIX}.partial")
    texts.write_parquet(staging)
    staging.replace(target)
    return target


def _existing_path(directory: Path, part: str, command: str) -> Path:
    path = part_path(directory, part, _SUFFIX)
    if not path.exists():
        raise FileNotFoundError(
            f"no texts for part {part!r} at {path}; run `interpretant {command}` first"
        )
    return path
