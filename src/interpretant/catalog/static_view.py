"""The static view's tokens of every cached document, per part (thin version, agenda 9.1 stage 6).

One row per cached document, in the cache's order, kept and dropped works alike: the work, its
static tokens, each token's character offset in the repaired text and whether it is a finite verb.
A build's removed ranges apply to the offsets when a cell's text is made, so cleaning changes never
rewrite this table; it changes only with the spaCy pass. A part is written whole or not at all.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.parts import part_path

STATIC_VIEW_SCHEMA = pl.Schema(
    {
        "work_id": pl.String(),
        "tokens": pl.List(pl.String()),
        "starts": pl.List(pl.Int64()),
        "finite_verbs": pl.List(pl.Boolean()),
    }
)
_SUFFIX = ".parquet"


def save_static_view(catalog: CatalogConfig, view: pl.DataFrame, part: str) -> Path:
    """Write one part's static view, replacing the part.

    Raises:
        ValueError: if the part name is not safe as a file name, or the columns are not the
            static view's.
    """
    if view.schema != STATIC_VIEW_SCHEMA:
        raise ValueError(f"expected the static view's columns, got {view.schema}")
    target = part_path(catalog.static_view.path, part, _SUFFIX)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(f"{_SUFFIX}.partial")
    view.write_parquet(staging)
    staging.replace(target)
    return target


def load_static_view(catalog: CatalogConfig, part: str) -> pl.DataFrame:
    """One part's static view, in the cache's order.

    Raises:
        ValueError: if the part name is not safe as a file name.
        FileNotFoundError: if the part's static view has not been built.
    """
    path = part_path(catalog.static_view.path, part, _SUFFIX)
    if not path.exists():
        raise FileNotFoundError(f"no static view for part {part!r} at {path}")
    return pl.read_parquet(path)
