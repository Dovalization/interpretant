"""Tables an earlier build wrote, read from the DVC cache at the git revision that recorded them.

Each commit's ``dvc.lock`` records the hashes of the outputs of the run it records, and DVC's file
system reads the files of a revision from the local cache, or from the default remote when they
are not there; so comparing with an earlier build needs no copy of it set aside.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from dvc.api import DVCFileSystem

from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.schemas import ABSTRACTS_SCHEMA

# The word repairs' decision tables, by catalog entry.
DECISION_TABLES = ("glued_words", "split_words", "lower_glued_words")


def load_abstracts_at(catalog: CatalogConfig, repository: Path, revision: str) -> pl.DataFrame:
    """The abstracts table as the build recorded at ``revision`` wrote it, its columns checked.

    ``repository`` is the project's root, the folder of the catalog file.

    Raises:
        FileNotFoundError: if that revision records no part of the table.
    """
    files = DVCFileSystem(str(repository), rev=revision)
    folder = _relative(catalog.abstracts.path, repository)
    parts = sorted(path for path in files.find(folder) if path.endswith(".parquet"))
    if not parts:
        raise FileNotFoundError(f"revision {revision} records no part of the abstracts table")
    table = pl.concat([_read(files, path) for path in parts])
    ABSTRACTS_SCHEMA.validate(table.lazy())
    return table


def load_decision_tables_at(
    catalog: CatalogConfig, repository: Path, revision: str
) -> dict[str, pl.DataFrame]:
    """Each word-repair decision table recorded at ``revision``; one it lacks is left out."""
    files = DVCFileSystem(str(repository), rev=revision)
    tables: dict[str, pl.DataFrame] = {}
    for name in DECISION_TABLES:
        path = _relative(getattr(catalog, name).path, repository)
        if files.exists(path):
            tables[name] = _read(files, path)
    return tables


def _read(files: DVCFileSystem, path: str) -> pl.DataFrame:
    with files.open(path, "rb") as handle:
        return pl.read_parquet(handle)


def _relative(path: Path, repository: Path) -> str:
    return path.resolve().relative_to(repository.resolve()).as_posix()
