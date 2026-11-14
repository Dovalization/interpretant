"""The works of the selected cells: one record per work, listing every field it was sampled for.

The same work in two cells is one record listing both fields, so it is kept in every cell it
belongs to (decision agenda 2.5). The text is the raw text as collected; the spaCy pass repairs it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from interpretant.catalog import RawWork


@dataclass(frozen=True)
class RawAbstract:
    """One abstract as collected, before repair, with the metadata the table keeps."""

    work_id: str
    year: int
    fields: tuple[str, ...]
    source: str
    document_type: str
    author_ids: tuple[str, ...]
    text: str


def merge_cells(
    cells: Iterable[tuple[str, Iterable[RawWork]]], *, source: str
) -> list[RawAbstract]:
    """One abstract per work id, in order of first appearance, listing every field it came from.

    A work's text, year, type and authors come from its first appearance; a work fetched twice
    carries the same OpenAlex record.
    """
    first_seen: dict[str, RawWork] = {}
    fields_by_work: dict[str, set[str]] = {}
    for field, works in cells:
        for work in works:
            first_seen.setdefault(work.work_id, work)
            fields_by_work.setdefault(work.work_id, set()).add(field)
    return [
        RawAbstract(
            work_id=work_id,
            year=work.year,
            fields=tuple(sorted(fields_by_work[work_id])),
            source=source,
            document_type=work.document_type,
            author_ids=work.author_ids,
            text=work.text,
        )
        for work_id, work in first_seen.items()
    ]


def part_of(abstract: RawAbstract) -> tuple[str, int]:
    """The part an abstract belongs to: its first field and its decade."""
    return abstract.fields[0], abstract.year - abstract.year % 10
