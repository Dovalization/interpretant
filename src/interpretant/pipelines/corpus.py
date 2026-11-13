"""What both corpus stages share: the selected cells, the parts, the metadata and the counts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from interpretant.catalog import (
    CatalogConfig,
    RawCell,
    WorkMetadata,
    list_raw_cells,
    load_work_metadata,
)
from interpretant.pipelines.params import CorpusSelection


@dataclass(frozen=True)
class CorpusResult:
    """What a corpus stage wrote, and how many abstracts each rule dropped, removed or flagged."""

    abstract_count: int
    parts: list[str]
    dropped_counts: dict[str, int]
    removal_counts: dict[str, int]
    flag_counts: dict[str, int]


def corpus_cells(catalog: CatalogConfig, selection: CorpusSelection) -> list[RawCell]:
    """The selected fields' and decades' raw cells, sorted by field and decade.

    Raises:
        ValueError: if no raw cell matches the selection.
    """
    cells = [
        cell
        for cell in list_raw_cells(catalog)
        if cell.field in selection.fields and cell.decade in selection.decades
    ]
    if not cells:
        raise ValueError(
            f"no raw cell for fields {selection.fields} in decades {selection.decades}"
        )
    return cells


def part_name(field: str, decade: int) -> str:
    """A part of the corpus tables: the abstracts whose first field is ``field``, in a decade."""
    return f"{field}_{decade}"


def load_corpus_metadata(
    catalog: CatalogConfig, cells: Sequence[RawCell]
) -> Iterator[WorkMetadata]:
    """Every cell's metadata records, cell after cell.

    Raises:
        ValueError: if a cell's metadata has not been fetched.
    """
    for cell in cells:
        try:
            yield from load_work_metadata(catalog, cell)
        except FileNotFoundError as exc:
            raise ValueError(
                f"no work metadata for {cell.field} {cell.decade}: "
                "run `interpretant fetch-metadata`"
            ) from exc


def summarise(tables: Mapping[str, pl.DataFrame]) -> CorpusResult:
    """The counts of a stage's parts: rows, and each drop, removal and flag reason."""
    table = pl.concat(tables.values())
    return CorpusResult(
        abstract_count=table.height,
        parts=list(tables),
        dropped_counts=_counts(table["dropped_reason"].drop_nulls()),
        removal_counts=_counts(table["removals"].explode().drop_nulls().struct.field("reason")),
        flag_counts=_counts(table["flags"].explode().drop_nulls()),
    )


def _counts(reasons: pl.Series) -> dict[str, int]:
    return {str(reason): count for reason, count in Counter(reasons.to_list()).most_common()}
