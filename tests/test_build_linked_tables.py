"""The build-linked-tables pipeline: one staging table per source from the raw files."""

from __future__ import annotations

import pytest

from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.linked_records import append_linked_records
from interpretant.catalog.metadata import append_work_metadata
from interpretant.catalog.raw import RawCell
from interpretant.catalog.staging import load_staging_table
from interpretant.pipelines.build_linked_tables import build_linked_tables
from tests.metadata_records import linked_metadata_record, missing_record
from tests.raw_cells import write_raw_cell

FETCHED = "2026-11-14"


def write_corrections(catalog: CatalogConfig, entries: str = "") -> None:
    catalog.doi_corrections.path.parent.mkdir(parents=True, exist_ok=True)
    catalog.doi_corrections.path.write_text(f"corrections:{entries or ' []'}\n")


def crossref_record(work: int) -> dict[str, object]:
    return {
        "work_id": f"https://openalex.org/W{work}",
        "identifier": f"10.1000/Work.{work}",
        "fetched": FETCHED,
        "DOI": f"10.1000/work.{work}",
        "page": "1-4",
    }


def prepare_cells(catalog: CatalogConfig) -> None:
    """Work 1 sits in two cells, work 2 in one, work 3 is lost to OpenAlex; 1 has a PubMed id."""
    write_raw_cell(catalog, "ai", 2010, [1, 2, 3])
    write_raw_cell(catalog, "compsci", 2010, [1])
    append_work_metadata(
        catalog,
        RawCell("ai", 2010),
        [linked_metadata_record(1, pmid=True), linked_metadata_record(2), missing_record(3)],
    )
    append_work_metadata(catalog, RawCell("compsci", 2010), [linked_metadata_record(1, pmid=True)])
    append_linked_records(
        catalog, RawCell("ai", 2010), "crossref", [crossref_record(1), crossref_record(2)]
    )
    append_linked_records(catalog, RawCell("compsci", 2010), "crossref", [crossref_record(1)])
    pubmed = {"work_id": "https://openalex.org/W1", "identifier": "901", "fetched": FETCHED}
    append_linked_records(catalog, RawCell("ai", 2010), "pubmed", [pubmed | {"title": "One"}])
    append_linked_records(catalog, RawCell("compsci", 2010), "pubmed", [pubmed | {"title": "One"}])


def test_build_linked_tables_writes_one_row_per_work_and_source(catalog: CatalogConfig) -> None:
    prepare_cells(catalog)
    write_corrections(
        catalog,
        "\n  - work_id: https://openalex.org/W2\n    openalex_doi: https://doi.org/10.1000/Work.2"
        "\n    doi: null\n    evidence: PubMed lists no DOI",
    )
    result = build_linked_tables(catalog)
    assert (result.openalex, result.crossref, result.pubmed) == (3, 2, 1)
    openalex = load_staging_table(catalog, "openalex").collect().sort("work_id")
    assert openalex["doi"].to_list() == ["10.1000/work.1", None, None]
    assert openalex["is_missing"].to_list() == [False, False, True]
    crossref = load_staging_table(catalog, "crossref").collect()
    assert crossref["page_count"].to_list() == [4, 4]
    assert load_staging_table(catalog, "pubmed").collect()["title"].to_list() == ["One"]


def test_build_linked_tables_refuses_a_lookup_that_did_not_finish(catalog: CatalogConfig) -> None:
    prepare_cells(catalog)
    write_raw_cell(catalog, "hci", 2010, [4])
    append_work_metadata(catalog, RawCell("hci", 2010), [linked_metadata_record(4)])
    write_corrections(catalog)
    with pytest.raises(ValueError, match="1 works have no crossref record"):
        build_linked_tables(catalog)
