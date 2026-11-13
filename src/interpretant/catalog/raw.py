"""The raw abstracts: the regular OpenAlex samples as downloaded, one file per cell.

Each cell is ``<field>/<decade>.jsonl.gz``, one work per line with the names the download wrote
(``id``, ``year``, ``type``, ``authors``, ``abstract``). Downloads are registered, never
regenerated (decision agenda 4.8), so the catalog only reads them.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from interpretant.catalog.config import CatalogConfig

_SUFFIX = ".jsonl.gz"


class RawWork(BaseModel):
    """One work of a raw cell file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    work_id: str = Field(alias="id", pattern=r"^https://openalex\.org/W\d+$")
    year: int
    document_type: str = Field(alias="type")
    author_ids: tuple[str, ...] = Field(alias="authors")
    text: str = Field(alias="abstract")


@dataclass(frozen=True, order=True)
class RawCell:
    """A field in a decade, as the raw files name it."""

    field: str
    decade: int


def list_raw_cells(catalog: CatalogConfig) -> list[RawCell]:
    """The cells that have a raw file, sorted by field and decade."""
    folder = catalog.raw_abstracts.path
    return sorted(
        RawCell(path.parent.name, int(path.name.removesuffix(_SUFFIX)))
        for path in folder.glob(f"*/*{_SUFFIX}")
    )


def load_raw_cell(catalog: CatalogConfig, cell: RawCell) -> Iterator[RawWork]:
    """One cell's works, each line checked as it is read.

    Raises:
        FileNotFoundError: if the cell has no raw file.
        pydantic.ValidationError: if a line is not a work record.
    """
    path = catalog.raw_abstracts.path / cell.field / f"{cell.decade}{_SUFFIX}"
    with gzip.open(path, "rt", encoding="utf-8") as lines:
        for line in lines:
            yield RawWork.model_validate_json(line)
