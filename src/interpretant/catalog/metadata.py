"""OpenAlex's record of each downloaded work beyond its abstract, one file per raw cell.

Each cell is ``<field>/<decade>.jsonl.gz``, laid out like the raw cells, one record per line as
OpenAlex returned it (the fields of decision agenda 4.8) plus the date it was fetched. A work
OpenAlex no longer returns is recorded as missing, so a rerun does not ask for it again. Like the
downloads, the files are registered, never regenerated; an interrupted fetch appends where it
stopped.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import date
from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.json_lines import append_json_lines, read_cell_files, read_json_lines
from interpretant.catalog.raw import RawCell

_SUFFIX = ".jsonl.gz"


class WorkMetadata(BaseModel):
    """The parts of a work's OpenAlex record the Crossref and PubMed lookups read.

    The corpus rules and the report read OpenAlex's staging table instead, which Polars builds
    from these files (``build-linked-tables``).
    """

    # The file keeps OpenAlex's whole record; the model reads only what the lookups use.
    model_config = ConfigDict(extra="ignore", frozen=True)

    work_id: str = Field(alias="id", pattern=r"^https://openalex\.org/W\d+$")
    fetched: date
    missing: bool = False
    doi: str | None = None
    # The work's identifiers in other systems by system ("doi", "pmid", "pmcid", "mag").
    ids: Mapping[str, str] = Field(default_factory=dict)


def fetched_work_ids(catalog: CatalogConfig, cell: RawCell) -> set[str]:
    """The works of a cell whose record is already in its metadata file (none without a file)."""
    if not _path(catalog, cell).exists():
        return set()
    return {record.work_id for record in load_work_metadata(catalog, cell)}


def append_work_metadata(
    catalog: CatalogConfig, cell: RawCell, records: Iterable[Mapping[str, object]]
) -> None:
    """Append records to a cell's metadata file, each checked before anything is written.

    Raises:
        pydantic.ValidationError: if a record lacks a work id or a fetch date.
    """
    checked = list(records)
    for record in checked:
        WorkMetadata.model_validate(record)
    append_json_lines(_path(catalog, cell), checked)


def load_work_metadata(catalog: CatalogConfig, cell: RawCell) -> Iterator[WorkMetadata]:
    """A cell's metadata records, each checked as it is read.

    Raises:
        FileNotFoundError: if the cell's metadata has not been fetched.
        pydantic.ValidationError: if a line is not a metadata record.
    """
    for line in read_json_lines(_path(catalog, cell)):
        yield WorkMetadata.model_validate_json(line)


def metadata_work_ids(catalog: CatalogConfig, cells: Sequence[RawCell]) -> set[str]:
    """The work ids of the cells' metadata records, read by Polars without checking the records.

    A stage that only needs to know every work has a record reads the ids alone: about 3 s for
    the whole corpus where checking every record takes about 30 (run 7c's profile).

    Raises:
        FileNotFoundError: if a cell's metadata has not been fetched (the first such cell).
    """
    paths = [_path(catalog, cell) for cell in cells]
    for cell, path in zip(cells, paths, strict=True):
        if not path.exists():
            raise FileNotFoundError(f"no work metadata for {cell.field} {cell.decade}")
    if not paths:
        return set()
    ids = pl.read_ndjson(paths, schema=pl.Schema({"id": pl.String()}))["id"]
    return set(ids.drop_nulls().to_list())


def load_work_metadata_table(catalog: CatalogConfig, schema: pl.Schema) -> pl.DataFrame:
    """Every cell's metadata records as one table, read by Polars with the given fields.

    A work downloaded in several cells has a row per cell; fields a record lacks are null.
    """
    return read_cell_files(catalog.work_metadata.path, schema)


def _path(catalog: CatalogConfig, cell: RawCell) -> Path:
    return catalog.work_metadata.path / cell.field / f"{cell.decade}{_SUFFIX}"
