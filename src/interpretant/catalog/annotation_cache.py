"""The transformer's annotation of target sentences, cached per part (`docs/performance.md`).

Re-annotating every target sentence takes the transformer about a quarter of an hour, and most
reruns of the usages stage change rules, not sentences. Each row holds one sentence's annotation,
under a key that hashes its tokens with everything that shapes the annotation (the model and the
package versions, its settings, the lemma exceptions, a version): every token's tag, part of
speech, morphology and lemma, and its entity tag as IOB (``B-ORG``, ``I-ORG``, ``O``). Each set of
settings has its own folder, so a change of settings starts an empty cache and the stale folder can
be deleted whole. The cache is not a stage output: a rerun reads what it finds and adds what it
annotates. A part is written whole or not at all.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.parts import part_path

ANNOTATION_CACHE_COLUMNS = pl.Schema(
    {
        "key": pl.String(),
        "tags": pl.List(pl.String()),
        "pos": pl.List(pl.String()),
        "morphs": pl.List(pl.String()),
        "lemmas": pl.List(pl.String()),
        "ents": pl.List(pl.String()),
    }
)
_SUFFIX = ".parquet"


def load_annotation_cache(catalog: CatalogConfig, settings: str, part: str) -> pl.DataFrame:
    """One part's cached annotations under one set of settings; empty if none were saved.

    Raises:
        ValueError: if the settings or part name is not safe as a file name.
    """
    path = _cache_path(catalog, settings, part)
    if not path.exists():
        return pl.DataFrame(schema=ANNOTATION_CACHE_COLUMNS)
    return pl.read_parquet(path, schema=ANNOTATION_CACHE_COLUMNS)


def save_annotation_cache(
    catalog: CatalogConfig, cache: pl.DataFrame, settings: str, part: str
) -> Path:
    """Write one part's cached annotations under one set of settings, replacing the part.

    Raises:
        ValueError: if the settings or part name is not safe as a file name, or a key repeats.
    """
    if cache["key"].n_unique() != cache.height:
        raise ValueError(f"the annotation cache of {part} repeats a key")
    target = _cache_path(catalog, settings, part)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(f"{_SUFFIX}.partial")
    cache.select(ANNOTATION_CACHE_COLUMNS.names()).write_parquet(staging)
    staging.replace(target)
    return target


def _cache_path(catalog: CatalogConfig, settings: str, part: str) -> Path:
    if not settings.isalnum():
        raise ValueError(f"the annotation settings {settings!r} are not a plain folder name")
    return part_path(catalog.annotation_cache.path / settings, part, _SUFFIX)
