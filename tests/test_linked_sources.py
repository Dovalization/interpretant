"""Looking works up in Crossref and PubMed, with fakes in place of both services."""

from __future__ import annotations

import gzip
import json
import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from Bio import Entrez
from click.testing import CliRunner

from interpretant.__main__ import cli
from interpretant.catalog import (
    CatalogConfig,
    RawCell,
    WorkMetadata,
    append_work_metadata,
    fetched_linked_work_ids,
)
from interpretant.cli import fetch_linked_sources as fetch_linked_sources_cli
from interpretant.corpus.linked_sources import (
    LinkedWork,
    fetch_crossref_records,
    fetch_pubmed_records,
    parse_pubmed_articles,
    select_linked_works,
)
from interpretant.pipelines.fetch_linked_sources import (
    SourcePlan,
    Throttle,
    fetch_cell_linked_records,
    is_transient,
    retry_transient,
)
from tests.metadata_records import metadata_record, missing_record

FETCHED = date(2026, 11, 14)


def linked_metadata(work: int, *, doi: bool = True, pmid: bool = False) -> dict[str, Any]:
    record = metadata_record(work)
    record["doi"] = f"https://doi.org/10.1000/Work.{work}" if doi else None
    record["ids"] = {"pmid": f"https://pubmed.ncbi.nlm.nih.gov/{900 + work}"} if pmid else {}
    return record


def write_raw_works(catalog: CatalogConfig, field: str, decade: int, works: range) -> None:
    path = catalog.raw_abstracts.path / field / f"{decade}.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as cell:
        for work in works:
            record = {
                "id": f"https://openalex.org/W{work}",
                "year": decade,
                "type": "article",
                "authors": [],
                "abstract": "Models learn senses.",
            }
            cell.write(json.dumps(record) + "\n")


def test_select_linked_works_takes_each_work_with_the_sources_identifier_once() -> None:
    records = [
        WorkMetadata.model_validate(record)
        for record in [
            linked_metadata(1, pmid=True),
            linked_metadata(2, doi=False),
            linked_metadata(1, pmid=True),
            missing_record(3),
            linked_metadata(4, doi=False, pmid=True),
        ]
    ]
    assert select_linked_works(records, "crossref") == [
        LinkedWork("https://openalex.org/W1", "10.1000/Work.1")
    ]
    assert select_linked_works(records, "pubmed") == [
        LinkedWork("https://openalex.org/W1", "901"),
        LinkedWork("https://openalex.org/W4", "904"),
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


class Element(str):
    """A parsed XML string with its attributes, as Biopython's parser gives it."""

    attributes: dict[str, str]

    def __new__(cls, value: str, **attributes: str) -> Element:
        element = super().__new__(cls, value)
        element.attributes = attributes
        return element


class Branch(dict[str, Any]):
    """A parsed XML element holding others, with its attributes."""

    def __init__(self, attributes: dict[str, str] | None = None, **children: Any) -> None:
        super().__init__(children)
        self.attributes = attributes or {}


def pubmed_answer() -> dict[str, Any]:
    heading = Branch(
        DescriptorName=Element("Narcolepsy", UI="D009290", MajorTopicYN="Y"),
        QualifierName=[Element("diagnosis")],
    )
    article = Branch(
        ArticleTitle=Element("Narcolepsy in a child"),
        Abstract=Branch(AbstractText=[Element("Rare onset.", Label="BACKGROUND")]),
        PublicationTypeList=[Element("Case Reports"), Element("Journal Article")],
        Language=[Element("eng")],
        Pagination=Branch(MedlinePgn="538-9"),
        Journal=Branch(Title=Element("Archives of disease in childhood")),
    )
    citation = Branch(
        {"Status": "MEDLINE"},
        PMID=Element("13831000"),
        Article=article,
        MedlineJournalInfo=Branch(NlmUniqueID="0372434"),
        MeshHeadingList=[heading],
        CommentsCorrectionsList=[Branch({"RefType": "ErratumIn"}, PMID=Element("13832000"))],
    )
    book = Branch(PMID=Element("20301000"), PublicationType=[Element("Review")])
    return {
        "PubmedArticle": [Branch(MedlineCitation=citation)],
        "PubmedBookArticle": [Branch(BookDocument=book)],
    }


def test_parse_pubmed_articles_reads_abstract_indexing_and_corrections() -> None:
    articles = parse_pubmed_articles(pubmed_answer())
    article = articles["13831000"]
    assert article["status"] == "MEDLINE"
    assert article["abstract"] == [{"label": "BACKGROUND", "category": "", "text": "Rare onset."}]
    assert article["publication_types"] == ["Case Reports", "Journal Article"]
    assert article["mesh_headings"] == [
        {
            "descriptor": "Narcolepsy",
            "descriptor_id": "D009290",
            "is_major": True,
            "qualifiers": ["diagnosis"],
        }
    ]
    assert (article["pages"], article["journal_nlm_id"]) == ("538-9", "0372434")
    assert article["comments_corrections"] == [{"type": "ErratumIn", "pmid": "13832000"}]
    assert articles["20301000"] == {
        "pmid": "20301000",
        "is_book": True,
        "publication_types": ["Review"],
    }
    json.dumps(articles)


def test_fetch_pubmed_records_marks_the_ids_pubmed_leaves_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Stream:
        closed = False

        def close(self) -> None:
            self.closed = True

    stream = Stream()
    requested: list[list[str]] = []

    def fake_efetch(**arguments: Any) -> Stream:
        requested.append(arguments["id"])
        return stream

    monkeypatch.setattr(Entrez, "efetch", fake_efetch)
    monkeypatch.setattr(Entrez, "read", lambda _stream: pubmed_answer())
    works = [
        LinkedWork("https://openalex.org/W1", "13831000"),
        LinkedWork("https://openalex.org/W2", "77"),
    ]
    records = fetch_pubmed_records(works, fetched=FETCHED)
    assert requested == [["13831000", "77"]]
    assert stream.closed
    assert records[0]["pmid"] == "13831000"
    assert records[0]["work_id"] == "https://openalex.org/W1"
    assert records[1]["missing"] is True


class FakeHTTPStatusError(Exception):
    """An HTTP client's error that carries the response, as habanero lets some through."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.response = type("Response", (), {"status_code": status_code})()


def test_is_transient_repeats_rate_limits_server_failures_and_lost_connections() -> None:
    assert is_transient(FakeHTTPStatusError(429))
    assert is_transient(FakeHTTPStatusError(503))
    assert not is_transient(FakeHTTPStatusError(400))
    assert is_transient(RuntimeError("HTTP Exception for https://api.crossref.org - timed out"))
    assert is_transient(ConnectionResetError())
    assert not is_transient(KeyError("items"))


def test_retry_transient_repeats_until_an_answer_and_raises_the_rest() -> None:
    failures = [FakeHTTPStatusError(429), FakeHTTPStatusError(503)]

    def flaky() -> list[dict[str, Any]]:
        if failures:
            raise failures.pop(0)
        return [{"answer": 1}]

    assert retry_transient(flaky, first_pause=0) == [{"answer": 1}]

    def refused() -> list[dict[str, Any]]:
        raise FakeHTTPStatusError(400)

    with pytest.raises(FakeHTTPStatusError):
        retry_transient(refused, first_pause=0)

    def always_limited() -> list[dict[str, Any]]:
        raise FakeHTTPStatusError(429)

    with pytest.raises(FakeHTTPStatusError):
        retry_transient(always_limited, attempts=3, first_pause=0)


def test_throttle_spaces_the_calls() -> None:
    throttle = Throttle(0.05)
    start = time.monotonic()
    for _ in range(3):
        throttle.wait()
    assert time.monotonic() - start >= 0.1


class FakeFetch:
    """Returns a record per work, missing for the works it is told the source lacks."""

    def __init__(self, missing: frozenset[str] = frozenset()) -> None:
        self.missing = missing
        self.batches: list[int] = []

    def __call__(self, works: Sequence[LinkedWork]) -> list[dict[str, Any]]:
        self.batches.append(len(works))
        return [self.record(work) for work in works]

    def record(self, work: LinkedWork) -> dict[str, Any]:
        record: dict[str, Any] = {"work_id": work.work_id, "fetched": "2026-11-14"}
        if work.identifier in self.missing:
            record["missing"] = True
        return record


def test_fetch_cell_linked_records_asks_in_batches_and_resumes(catalog: CatalogConfig) -> None:
    cell = RawCell("ai", 2010)
    records = [linked_metadata(work, doi=work != 5) for work in range(1, 9)]
    append_work_metadata(catalog, cell, records)
    fetch = FakeFetch(missing=frozenset({"10.1000/Work.2"}))
    plan = SourcePlan(fetch=fetch, batch_size=3, workers=2)
    result = fetch_cell_linked_records(catalog, cell, "crossref", plan)
    assert sorted(fetch.batches) == [1, 3, 3]
    assert (result.already_fetched, result.found, result.missing) == (0, 6, 1)
    assert len(fetched_linked_work_ids(catalog, cell, "crossref")) == 7
    again = fetch_cell_linked_records(catalog, cell, "crossref", plan)
    assert (again.already_fetched, again.found, sorted(fetch.batches)) == (7, 0, [1, 3, 3])


def test_fetch_linked_sources_command_prints_a_line_per_source_and_cell(
    catalog: CatalogConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_raw_works(catalog, "ai", 2010, range(1, 4))
    append_work_metadata(
        catalog, RawCell("ai", 2010), [linked_metadata(work, pmid=True) for work in range(1, 4)]
    )
    contacts: list[str | None] = []

    def fake_plan(source: str, settings: Any, *, today: date) -> SourcePlan:
        contacts.append(settings.contact_email)
        return SourcePlan(fetch=FakeFetch(), batch_size=200, workers=1)

    monkeypatch.setenv("CONTACT_EMAIL", "someone@example.org")
    monkeypatch.setattr(fetch_linked_sources_cli, "build_source_plan", fake_plan)
    arguments = ["fetch-linked-sources", "--catalog", str(tmp_path / "catalog.yaml")]
    outcome = CliRunner().invoke(cli, arguments)
    assert outcome.exit_code == 0, outcome.output
    assert "crossref [1/1] ai 2010: 3 found, 0 missing, 0 already there" in outcome.output
    assert "pubmed [1/1] ai 2010: 3 found, 0 missing, 0 already there" in outcome.output
    assert contacts == ["someone@example.org", "someone@example.org"]
