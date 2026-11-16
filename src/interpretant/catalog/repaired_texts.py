"""The repaired texts: each work's text as the count-word-repairs stage leaves it, per part.

Repair, the glued-word splits and the split-word joins (decision agenda 2.3) are made once, in
that stage's worker processes, and the spaCy pass reads the result instead of repairing in its
main process (run 7c's profile, step 9). One row per work, in the part's order; a part is
written whole or not at all, and a part with no works is written empty.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.parts import part_path

REPAIRED_TEXTS_SCHEMA = pl.Schema({"work_id": pl.String(), "text": pl.String()})
_SUFFIX = ".parquet"


def save_repaired_texts(catalog: CatalogConfig, texts: pl.DataFrame, part: str) -> Path:
    """Write one part's repaired texts, replacing the part.

    Raises:
        ValueError: if the part name is not safe as a file name, or the columns are not the
            repaired-text columns.
    """
    if texts.schema != REPAIRED_TEXTS_SCHEMA:
        raise ValueError(f"expected the repaired-text columns, got {texts.schema}")
    target = part_path(catalog.repaired_texts.path, part, _SUFFIX)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(f"{_SUFFIX}.partial")
    texts.write_parquet(staging)
    staging.replace(target)
    return target


def load_repaired_texts(catalog: CatalogConfig, part: str) -> pl.DataFrame:
    """One part's repaired texts, in the part's order.

    Raises:
        ValueError: if the part name is not safe as a file name.
        FileNotFoundError: if the part's texts have not been repaired.
    """
    path = part_path(catalog.repaired_texts.path, part, _SUFFIX)
    if not path.exists():
        raise FileNotFoundError(
            f"no repaired texts for part {part!r} at {path}; "
            "run `interpretant count-word-repairs` first"
        )
    return pl.read_parquet(path)
