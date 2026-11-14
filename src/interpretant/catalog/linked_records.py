"""Each downloaded work's record in another source (Crossref, PubMed), one file per raw cell.

Laid out like the work metadata: ``<field>/<decade>.jsonl.gz`` under the source's catalog path,
one record per line as the source returned it, plus the work id and the date it was fetched. A
work the source does not hold is recorded as missing, so a rerun does not ask for it again. Like
the downloads, the files are registered, never regenerated (decision agenda 4.8, linked sources).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.json_lines import append_json_lines, read_json_lines
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


def fetched_linked_work_ids(
    catalog: CatalogConfig, cell: RawCell, source: LinkedSource
) -> set[str]:
    """The works of a cell already looked up in the source (none without a file)."""
    path = _path(catalog, cell, source)
    if not path.exists():
        return set()
    return {LinkedRecord.model_validate_json(line).work_id for line in read_json_lines(path)}


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
