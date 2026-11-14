"""Looking works up in Crossref by DOI, with a fake in place of habanero's client."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from interpretant.catalog import LinkedWork, WorkMetadata
from interpretant.corpus.crossref import (
    batch_crossref_works,
    fetch_crossref_records,
    select_crossref_works,
)
from tests.metadata_records import linked_metadata_record, missing_record

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
