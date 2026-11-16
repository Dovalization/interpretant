"""What the corpus stages share: the selected cells, their works and parts, and the counts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.metadata import metadata_work_ids
from interpretant.catalog.raw import RawCell, list_raw_cells, load_raw_cell
from interpretant.corpus.works import RawAbstract, merge_cells, part_of
from interpretant.pipelines.params import CorpusSelection
from interpretant.pipelines.progress import Progress

# The source of the regular OpenAlex samples, as the table records it.
REGULAR = "regular"


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


def corpus_parts(cells: Sequence[RawCell]) -> list[str]:
    """The parts the cells make, one per field and decade, in order."""
    return [part_name(cell.field, cell.decade) for cell in cells]


def load_corpus_works(
    catalog: CatalogConfig, cells: Sequence[RawCell], progress: Progress
) -> list[RawAbstract]:
    """The cells' works merged, one record per work, once every work's metadata is checked.

    Raises:
        ValueError: if a cell's metadata has not been fetched or lacks a work.
    """
    progress.stage("merging the raw cells")
    abstracts = merge_cells(
        ((cell.field, load_raw_cell(catalog, cell)) for cell in cells), source=REGULAR
    )
    progress.stage("checking the work metadata")
    try:
        fetched = metadata_work_ids(catalog, cells)
    except FileNotFoundError as exc:
        raise ValueError(f"{exc}: run `interpretant fetch-metadata`") from exc
    unfetched = {abstract.work_id for abstract in abstracts} - fetched
    if unfetched:
        raise ValueError(
            f"{len(unfetched)} works have no metadata record: run `interpretant fetch-metadata`"
        )
    return abstracts


def works_by_part(abstracts: Iterable[RawAbstract]) -> dict[str, list[RawAbstract]]:
    """The abstracts grouped by part, in their order within each part."""
    groups: dict[str, list[RawAbstract]] = {}
    for abstract in abstracts:
        groups.setdefault(part_name(*part_of(abstract)), []).append(abstract)
    return groups


def part_name(field: str, decade: int) -> str:
    """A part of the corpus tables: the abstracts whose first field is ``field``, in a decade."""
    return f"{field}_{decade}"


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


def combine_results(results: Iterable[CorpusResult]) -> CorpusResult:
    """One result for several parts' results: rows and every reason's count added up."""
    results = list(results)
    return CorpusResult(
        abstract_count=sum(result.abstract_count for result in results),
        parts=[part for result in results for part in result.parts],
        dropped_counts=_added(result.dropped_counts for result in results),
        removal_counts=_added(result.removal_counts for result in results),
        flag_counts=_added(result.flag_counts for result in results),
    )


def _added(counts: Iterable[Mapping[str, int]]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for count in counts:
        total.update(count)
    return dict(total.most_common())


def _counts(reasons: pl.Series) -> dict[str, int]:
    return {str(reason): count for reason, count in Counter(reasons.to_list()).most_common()}
