"""OpenAlex's record of each downloaded work beyond its abstract, one file per raw cell.

Each cell is ``<field>/<decade>.jsonl.gz``, laid out like the raw cells, one record per line as
OpenAlex returned it (the fields of decision agenda 4.8) plus the date it was fetched. A work
OpenAlex no longer returns is recorded as missing, so a rerun does not ask for it again. Like the
downloads, the files are registered, never regenerated; an interrupted fetch appends where it
stopped. Each append is one gzip member, compressed in memory and written in one call, so a fetch
stopped between batches leaves a whole file (gzip reads the members as one stream).
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator, Mapping
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.raw import RawCell

_SUFFIX = ".jsonl.gz"


class Source(BaseModel):
    """Where a copy of a work is hosted: a journal, a repository, a conference and so on."""

    # The file keeps OpenAlex's whole record; the models read only what the rules use.
    model_config = ConfigDict(extra="ignore", frozen=True)

    display_name: str | None = None
    type: str | None = None


class Location(BaseModel):
    """One copy of a work and its source."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    source: Source | None = None


class WorkMetadata(BaseModel):
    """The parts of a work's OpenAlex record the corpus rules read."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    work_id: str = Field(alias="id", pattern=r"^https://openalex\.org/W\d+$")
    fetched: date
    missing: bool = False
    doi: str | None = None
    indexed_in: tuple[str, ...] = ()
    primary_location: Location | None = None
    is_paratext: bool = False
    is_retracted: bool = False


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
    lines = []
    for record in records:
        WorkMetadata.model_validate(record)
        lines.append(json.dumps(record, ensure_ascii=False) + "\n")
    member = gzip.compress("".join(lines).encode("utf-8"))
    path = _path(catalog, cell)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(member)


def load_work_metadata(catalog: CatalogConfig, cell: RawCell) -> Iterator[WorkMetadata]:
    """A cell's metadata records, each checked as it is read.

    Raises:
        FileNotFoundError: if the cell's metadata has not been fetched.
        pydantic.ValidationError: if a line is not a metadata record.
    """
    with gzip.open(_path(catalog, cell), "rt", encoding="utf-8") as lines:
        for line in lines:
            yield WorkMetadata.model_validate_json(line)


def _path(catalog: CatalogConfig, cell: RawCell) -> Path:
    return catalog.work_metadata.path / cell.field / f"{cell.decade}{_SUFFIX}"
