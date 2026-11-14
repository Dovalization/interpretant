"""The fetch-linked-sources pipeline and command, with fakes in place of the sources."""

from __future__ import annotations

import http.client
import time
import urllib.error
from collections.abc import Sequence
from datetime import date
from email.message import Message
from pathlib import Path
from typing import Any

import pytest
from Bio.Entrez.Parser import CorruptedXMLError
from click.testing import CliRunner

from interpretant.__main__ import cli
from interpretant.catalog import (
    CatalogConfig,
    DOICorrection,
    LinkedWork,
    RawCell,
    append_work_metadata,
    fetched_linked_work_ids,
)
from interpretant.cli import fetch_linked_sources as fetch_linked_sources_cli
from interpretant.corpus.crossref import batch_crossref_works, select_crossref_works
from interpretant.corpus.pubmed import batch_pubmed_works, select_pubmed_works
from interpretant.pipelines.fetch_linked_sources import (
    SourcePlan,
    Throttle,
    fetch_cell_linked_records,
    is_transient_crossref_error,
    is_transient_pubmed_error,
    retry_transient,
)
from tests.metadata_records import linked_metadata_record
from tests.raw_cells import write_raw_cell


class FakeHTTPStatusError(Exception):
    """An HTTP client's error that carries the response, as habanero lets some through."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.response = type("Response", (), {"status_code": status_code})()


def test_is_transient_crossref_error_repeats_rate_limits_failures_and_lost_connections() -> None:
    assert is_transient_crossref_error(FakeHTTPStatusError(429))
    assert is_transient_crossref_error(FakeHTTPStatusError(503))
    assert not is_transient_crossref_error(FakeHTTPStatusError(400))
    assert is_transient_crossref_error(
        RuntimeError("HTTP Exception for https://api.crossref.org - timed out")
    )
    assert is_transient_crossref_error(ConnectionResetError())
    assert not is_transient_crossref_error(KeyError("items"))


def test_is_transient_pubmed_error_repeats_refusals_and_cut_off_answers() -> None:
    refusal = urllib.error.HTTPError(
        "https://eutils.ncbi.nlm.nih.gov", 400, "Bad Request", Message(), None
    )
    assert is_transient_pubmed_error(refusal)
    assert is_transient_pubmed_error(CorruptedXMLError("mismatched tag: line 760, column 6"))
    assert is_transient_pubmed_error(http.client.IncompleteRead(b"<PubmedArticleSet>"))
    assert not is_transient_pubmed_error(KeyError("MedlineCitation"))


def test_retry_transient_repeats_until_an_answer_and_raises_the_rest() -> None:
    failures = [FakeHTTPStatusError(429), FakeHTTPStatusError(503)]

    def flaky() -> list[dict[str, Any]]:
        if failures:
            raise failures.pop(0)
        return [{"answer": 1}]

    assert retry_transient(flaky, is_transient=is_transient_crossref_error, first_pause=0) == [
        {"answer": 1}
    ]

    def refused() -> list[dict[str, Any]]:
        raise FakeHTTPStatusError(400)

    with pytest.raises(FakeHTTPStatusError):
        retry_transient(refused, is_transient=is_transient_crossref_error, first_pause=0)

    def always_limited() -> list[dict[str, Any]]:
        raise FakeHTTPStatusError(429)

    with pytest.raises(FakeHTTPStatusError):
        retry_transient(
            always_limited, is_transient=is_transient_crossref_error, attempts=3, first_pause=0
        )


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


def in_threes(works: Sequence[LinkedWork]) -> list[list[LinkedWork]]:
    return [list(works[start : start + 3]) for start in range(0, len(works), 3)]


def test_fetch_cell_linked_records_asks_in_batches_and_resumes(catalog: CatalogConfig) -> None:
    cell = RawCell("ai", 2010)
    records = [linked_metadata_record(work, doi=work != 5) for work in range(1, 9)]
    append_work_metadata(catalog, cell, records)
    fetch = FakeFetch(missing=frozenset({"10.1000/Work.2"}))
    plan = SourcePlan(select=select_crossref_works, batch=in_threes, fetch=fetch, workers=2)
    result = fetch_cell_linked_records(catalog, cell, "crossref", plan, {})
    assert sorted(fetch.batches) == [1, 3, 3]
    assert (result.already_fetched, result.found, result.missing) == (0, 6, 1)
    assert len(fetched_linked_work_ids(catalog, cell, "crossref")) == 7
    again = fetch_cell_linked_records(catalog, cell, "crossref", plan, {})
    assert (again.already_fetched, again.found, sorted(fetch.batches)) == (7, 0, [1, 3, 3])


def test_fetch_cell_linked_records_asks_with_the_corrected_dois(catalog: CatalogConfig) -> None:
    cell = RawCell("ai", 2010)
    append_work_metadata(catalog, cell, [linked_metadata_record(work) for work in (1, 2)])
    correction = DOICorrection(
        work_id="https://openalex.org/W1",
        openalex_doi="https://doi.org/",
        doi=None,
        evidence="PubMed lists no DOI",
    )
    fetch = FakeFetch()
    plan = SourcePlan(select=select_crossref_works, batch=in_threes, fetch=fetch, workers=1)
    result = fetch_cell_linked_records(
        catalog, cell, "crossref", plan, {correction.work_id: correction}
    )
    assert (result.found, fetch.batches) == (1, [1])
    assert fetched_linked_work_ids(catalog, cell, "crossref") == {"https://openalex.org/W2"}


def test_fetch_linked_sources_command_prints_a_line_per_source_and_cell(
    catalog: CatalogConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_raw_cell(catalog, "ai", 2010, range(1, 4))
    append_work_metadata(
        catalog,
        RawCell("ai", 2010),
        [linked_metadata_record(work, pmid=True) for work in range(1, 4)],
    )
    contacts: list[str | None] = []

    def fake_plan(source: str, settings: Any, *, today: date) -> SourcePlan:
        contacts.append(settings.contact_email)
        if source == "crossref":
            return SourcePlan(select_crossref_works, batch_crossref_works, FakeFetch(), workers=1)
        return SourcePlan(select_pubmed_works, batch_pubmed_works, FakeFetch(), workers=1)

    monkeypatch.setenv("CONTACT_EMAIL", "someone@example.org")
    catalog.doi_corrections.path.parent.mkdir(parents=True, exist_ok=True)
    catalog.doi_corrections.path.write_text("corrections: []\n")
    monkeypatch.setattr(fetch_linked_sources_cli, "build_source_plan", fake_plan)
    arguments = ["fetch-linked-sources", "--catalog", str(tmp_path / "catalog.yaml")]
    outcome = CliRunner().invoke(cli, arguments)
    assert outcome.exit_code == 0, outcome.output
    assert "crossref [1/1] ai 2010: 3 found, 0 missing, 0 already there" in outcome.output
    assert "pubmed [1/1] ai 2010: 3 found, 0 missing, 0 already there" in outcome.output
    assert contacts == ["someone@example.org", "someone@example.org"]
