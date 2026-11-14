"""Looking works up in Crossref by DOI, with a fake in place of habanero's client."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from interpretant.catalog import LinkedWork, WorkMetadata
from interpretant.corpus.crossref import (
    CROSSREF_RECORD_SCHEMA,
    batch_crossref_works,
    build_crossref_works,
    fetch_crossref_records,
    parse_jats_text,
    select_crossref_works,
)
from tests.metadata_records import linked_metadata_record, missing_record, read_records

FETCHED = date(2026, 11, 14)


def test_select_crossref_works_takes_each_work_with_a_doi_once() -> None:
    records = [
        WorkMetadata.model_validate(record)
        for record in [
            linked_metadata_record(1),
            linked_metadata_record(2, doi=False, pmid=True),
            linked_metadata_record(1),
            missing_record(3),
        ]
    ]
    assert select_crossref_works(records) == [
        LinkedWork("https://openalex.org/W1", "10.1000/Work.1")
    ]


class FakeCrossref:
    """Answers a DOI filter with the DOIs it holds, upper-cased as Crossref may return them."""

    def __init__(self, held: set[str]) -> None:
        self.held = held
        self.filters: list[list[str]] = []
        self.single: list[str] = []

    def works(self, **arguments: Any) -> dict[str, Any] | None:
        if "ids" in arguments:
            self.single.append(arguments["ids"])
            if arguments["ids"] not in self.held:
                return None
            return {"message": {"DOI": arguments["ids"], "page": "1-2", "reference": ["dropped"]}}
        dois = arguments["filter"]["doi"]
        self.filters.append(dois)
        items = [{"DOI": doi.upper(), "page": "5"} for doi in dois if doi in self.held]
        return {"message": {"items": items}}


def test_fetch_crossref_records_keeps_the_order_and_marks_what_crossref_lacks() -> None:
    works = [
        LinkedWork("https://openalex.org/W1", "10.1000/a"),
        LinkedWork("https://openalex.org/W2", "10.1000/b"),
        LinkedWork("https://openalex.org/W3", "10.1386/x.4.1,2,3.69_1"),
    ]
    crossref = FakeCrossref(held={"10.1000/b", "10.1386/x.4.1,2,3.69_1"})
    records = fetch_crossref_records(crossref, works, fetched=FETCHED)
    assert crossref.filters == [["10.1000/a", "10.1000/b"]]
    assert crossref.single == ["10.1386/x.4.1,2,3.69_1"]
    assert [record["work_id"] for record in records] == [work.work_id for work in works]
    assert records[0] == {
        "work_id": "https://openalex.org/W1",
        "identifier": "10.1000/a",
        "fetched": "2026-11-14",
        "missing": True,
    }
    assert records[1]["page"] == "5"
    assert records[2]["page"] == "1-2"
    assert "reference" not in records[2]


def test_fetch_crossref_records_refuses_more_than_one_request_holds() -> None:
    works = [LinkedWork(f"https://openalex.org/W{n}", f"10.1/{n}") for n in range(201)]
    with pytest.raises(ValueError, match="at most 200"):
        fetch_crossref_records(FakeCrossref(set()), works, fetched=FETCHED)


def test_batch_crossref_works_caps_each_request_by_count_and_by_length() -> None:
    short = [LinkedWork(f"https://openalex.org/W{n}", f"10.1/{n}") for n in range(450)]
    assert [len(batch) for batch in batch_crossref_works(short)] == [200, 200, 50]
    long_doi = "10.1000/" + "x" * 120
    long = [LinkedWork(f"https://openalex.org/W{n}", f"{long_doi}{n}") for n in range(100)]
    batches = batch_crossref_works(long)
    assert len(batches) > 1
    assert [work for batch in batches for work in batch] == long


def test_fetch_crossref_records_does_not_send_a_doi_without_crossrefs_form() -> None:
    works = [
        LinkedWork("https://openalex.org/W1", "10.153/scnp00400195"),
        LinkedWork("https://openalex.org/W2", "10.1000/b"),
    ]
    crossref = FakeCrossref(held={"10.1000/b"})
    records = fetch_crossref_records(crossref, works, fetched=FETCHED)
    assert crossref.filters == [["10.1000/b"]]
    assert records[0]["missing"] is True
    assert records[0]["invalid_doi"] is True
    assert "invalid_doi" not in records[1]


def test_parse_jats_text_keeps_the_paragraphs_without_the_heading() -> None:
    markup = (
        "<jats:title>Abstract</jats:title>\n  <jats:p>Brain networks &amp; "
        "<jats:italic>IQ</jats:italic>\n differ.</jats:p><jats:p>A second point.</jats:p>"
    )
    assert parse_jats_text(markup) == "Brain networks & IQ differ. A second point."


def test_build_crossref_works_flattens_found_and_missing_records() -> None:
    found = {
        "work_id": "https://openalex.org/W1",
        "identifier": "10.1002/HBM.24074",
        "fetched": "2026-11-14",
        "DOI": "10.1002/hbm.24074",
        "type": "journal-article",
        "title": ["Brain networks"],
        "container-title": ["Human Brain Mapping"],
        "ISSN": ["1065-9471", "1065-9470"],
        "published": {"date-parts": [[2018, 4, 17]]},
        "abstract": "<jats:p>Networks differ.</jats:p>",
        "page": "3253-3262",
        "references-count": 64,
        "relation": {"has-preprint": [{"id": "10.1101/143891"}]},
        "update-to": [{"type": "correction"}],
        "updated-by": [{"type": "retraction"}],
        "license": [{"URL": "https://example.org/tdm", "content-version": "tdm"}],
        "link": [
            {"URL": "https://example.org/pdf", "intended-application": "text-mining"},
            {"URL": "https://example.org/check", "intended-application": "similarity-checking"},
        ],
    }
    missing = {
        "work_id": "https://openalex.org/W2",
        "identifier": "10.153/x",
        "fetched": "2026-11-14",
        "missing": True,
        "invalid_doi": True,
    }
    table = build_crossref_works(read_records([found, missing], CROSSREF_RECORD_SCHEMA))
    first, second = table.to_dicts()
    assert first["doi"] == "10.1002/hbm.24074"
    assert (first["is_found"], first["page_count"], first["published_year"]) == (True, 10, 2018)
    assert first["abstract"] == "Networks differ."
    assert first["issns"] == ["1065-9471"]
    assert (first["relation_types"], first["update_types"]) == (["has-preprint"], ["correction"])
    assert first["is_updated"] and first["has_text_mining_licence"]
    assert first["text_mining_links"] == ["https://example.org/pdf"]
    assert (second["is_found"], second["is_invalid_doi"], second["abstract"]) == (False, True, None)
    assert second["relation_types"] == []
