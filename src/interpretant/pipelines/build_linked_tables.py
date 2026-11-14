"""Build the staging tables, one per source, from the raw files (decision agenda 4.8, layers).

A DVC stage: the tables follow from the raw files, which are registered snapshots, so a rebuild
gives the same tables. Each work appears once per table, however many cells downloaded it; the
DOI corrections are applied to OpenAlex's metadata here, so every later step reads corrected
values. Before building, every work a source could be asked about is checked to have a record.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, TypeVar

from interpretant.catalog import (
    LINKED_SOURCES,
    CatalogConfig,
    LinkedSource,
    RawCell,
    list_raw_cells,
    load_doi_corrections,
    load_linked_records,
    load_work_metadata,
    save_staging_table,
)
from interpretant.corpus.crossref import build_crossref_works, select_crossref_works
from interpretant.corpus.pubmed import build_pubmed_works, select_pubmed_works
from interpretant.corpus.work_metadata import build_openalex_works, correct_dois
from interpretant.pipelines.progress import SILENT, Progress

Item = TypeVar("Item")


@dataclass(frozen=True)
class LinkedTablesResult:
    """How many rows each staging table holds."""

    openalex: int
    crossref: int
    pubmed: int


def build_linked_tables(
    catalog: CatalogConfig, *, progress: Progress = SILENT
) -> LinkedTablesResult:
    """Build and save the OpenAlex, Crossref and PubMed staging tables.

    Raises:
        ValueError: if a work the source could be asked about has no record from it (the lookup
            did not finish: run ``interpretant fetch-linked-sources``).
    """
    cells = list_raw_cells(catalog)
    corrections = load_doi_corrections(catalog)
    progress.stage("reading the work metadata", total=len(cells))
    metadata = _first_per_work(
        (work.work_id, work)
        for cell in _advancing(cells, progress)
        for work in correct_dois(load_work_metadata(catalog, cell), corrections)
    )
    openalex = build_openalex_works(metadata[work] for work in sorted(metadata))
    found = list(metadata.values())
    expected: dict[LinkedSource, set[str]] = {
        "crossref": {work.work_id for work in select_crossref_works(found)},
        "pubmed": {work.work_id for work in select_pubmed_works(found)},
    }
    records = {
        source: _linked_records(catalog, cells, source, expected[source], progress)
        for source in LINKED_SOURCES
    }
    crossref = build_crossref_works(
        records["crossref"][work] for work in sorted(records["crossref"])
    )
    pubmed = build_pubmed_works(records["pubmed"][work] for work in sorted(records["pubmed"]))
    progress.stage("saving the staging tables")
    save_staging_table(catalog, "openalex", openalex)
    save_staging_table(catalog, "crossref", crossref)
    save_staging_table(catalog, "pubmed", pubmed)
    return LinkedTablesResult(
        openalex=openalex.height, crossref=crossref.height, pubmed=pubmed.height
    )


def _linked_records(
    catalog: CatalogConfig,
    cells: list[RawCell],
    source: LinkedSource,
    expected: set[str],
    progress: Progress,
) -> dict[str, dict[str, Any]]:
    progress.stage(f"reading the {source} records", total=len(cells))
    records = _first_per_work(
        (record["work_id"], record)
        for cell in _advancing(cells, progress)
        for record in load_linked_records(catalog, cell, source)
    )
    unrecorded = expected - records.keys()
    if unrecorded:
        raise ValueError(
            f"{len(unrecorded)} works have no {source} record: run "
            "`interpretant fetch-linked-sources`"
        )
    return records


def _first_per_work(pairs: Iterable[tuple[str, Item]]) -> dict[str, Item]:
    first: dict[str, Item] = {}
    for work_id, item in pairs:
        first.setdefault(work_id, item)
    return first


def _advancing(cells: list[RawCell], progress: Progress) -> Iterator[RawCell]:
    for cell in cells:
        yield cell
        progress.advance()
