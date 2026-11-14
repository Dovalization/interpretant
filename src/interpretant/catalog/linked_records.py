"""Each downloaded work's record in another source (Crossref, PubMed), one file per raw cell.

Laid out like the work metadata: ``<field>/<decade>.jsonl.gz`` under the source's catalog path,
one record per line as the source returned it, plus the work id, the identifier it was looked up
by and the date it was fetched. A work the source does not hold is recorded as missing, so a
rerun does not ask for it again (decision agenda 4.8, linked sources).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.json_lines import append_json_lines, read_json_lines
from interpretant.catalog.metadata import WorkMetadata
from interpretant.catalog.raw import RawCell

LinkedSource = Literal["crossref", "pubmed"]
LINKED_SOURCES: tuple[LinkedSource, ...] = ("crossref", "pubmed")


class LinkedRecord(BaseModel):
    """The fields every linked record carries, whatever the source."""

    # The file keeps the source's whole record; the model checks only what every record needs.
    model_config = ConfigDict(extra="ignore", frozen=True)

    work_id: str = Field(pattern=r"^https://openalex\.org/W\d+$")
    fetched: date
    missing: bool = False


@dataclass(frozen=True)
class LinkedWork:
    """A work and the identifier a source knows it by: its DOI, or its PubMed id."""

    work_id: str
    identifier: str


def select_linked_works(
    records: Iterable[WorkMetadata], identifier_of: Callable[[WorkMetadata], str]
) -> list[LinkedWork]:
    """The works with an identifier, in order and each once; works OpenAlex lost are left out."""
    works: dict[str, LinkedWork] = {}
    for record in records:
        identifier = "" if record.missing else identifier_of(record)
        if identifier and record.work_id not in works:
            works[record.work_id] = LinkedWork(record.work_id, identifier)
    return list(works.values())


def build_linked_record(
    work: LinkedWork, item: Mapping[str, Any] | None, *, fetched: date
) -> dict[str, Any]:
    """The line stored for a work: the source's item, or a record marked missing without one."""
    record = {
        "work_id": work.work_id,
        "identifier": work.identifier,
        "fetched": fetched.isoformat(),
    }
    if item is None:
        return {**record, "missing": True}
    return {**record, **item}


def fetched_linked_work_ids(
    catalog: CatalogConfig, cell: RawCell, source: LinkedSource
) -> set[str]:
    """The works of a cell already looked up in the source (none without a file)."""
    path = _path(catalog, cell, source)
    if not path.exists():
        return set()
    return {LinkedRecord.model_validate_json(line).work_id for line in read_json_lines(path)}


def load_linked_records(
    catalog: CatalogConfig, cell: RawCell, source: LinkedSource
) -> Iterator[dict[str, Any]]:
    """A cell's records from the source as stored, each checked to carry a work id and a date.

    A cell none of whose works the source could be asked about has no file and gives nothing;
    whether every work that should have a record has one is the reader's check.
    """
    path = _path(catalog, cell, source)
    if not path.exists():
        return
    for line in read_json_lines(path):
        record: dict[str, Any] = json.loads(line)
        LinkedRecord.model_validate(record)
        yield record


def append_linked_records(
    catalog: CatalogConfig,
    cell: RawCell,
    source: LinkedSource,
    records: Iterable[Mapping[str, object]],
) -> None:
    """Append a source's records to a cell's file, each checked before anything is written.

    Raises:
        pydantic.ValidationError: if a record lacks a work id or a fetch date.
    """
    checked = list(records)
    for record in checked:
        LinkedRecord.model_validate(record)
    append_json_lines(_path(catalog, cell, source), checked)


def _path(catalog: CatalogConfig, cell: RawCell, source: LinkedSource) -> Path:
    entry = catalog.crossref_records if source == "crossref" else catalog.pubmed_records
    return entry.path / cell.field / f"{cell.decade}.jsonl.gz"
