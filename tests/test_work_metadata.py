"""OpenAlex's work metadata: the fetch through pyalex, the repository-only rule and the flags.

pyalex is replaced by a fake query, so the tests never call OpenAlex.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl
import pytest
from pyalex import config

from interpretant.catalog.curated import DOICorrection
from interpretant.catalog.metadata import WorkMetadata
from interpretant.corpus import work_metadata
from interpretant.corpus.removals import MALFORMED
from interpretant.corpus.work_metadata import (
    IDS_PER_REQUEST,
    METADATA_FIELDS,
    NOT_IN_OPENALEX,
    OPENALEX_METADATA_SCHEMA,
    PARATEXT,
    REPOSITORY_ONLY,
    RETRACTED,
    build_openalex_works,
    check_work_metadata,
    configure_openalex,
    correct_dois,
    fetch_work_metadata,
    is_repository_only,
    mark_metadata_flags,
)
from tests.abstracts_tables import abstract_row, abstracts_table
from tests.metadata_records import FETCHED, metadata_record, missing_record, read_records


class FakeQuery:
    """pyalex's ``Works`` query as the fetch uses it: an OR filter on ids, a select, a get."""

    def __init__(self, known: dict[str, dict[str, Any]], calls: list[tuple[str, Any]]) -> None:
        self.known = known
        self.calls = calls
        self.ids: list[str] = []

    def filter_or(self, **filters: list[str]) -> FakeQuery:
        self.calls.append(("filter_or", filters))
        self.ids = filters["openalex_id"]
        return self

    def select(self, fields: list[str]) -> FakeQuery:
        self.calls.append(("select", fields))
        return self

    def get(self, per_page: int) -> list[dict[str, Any]]:
        self.calls.append(("get", per_page))
        return [self.known[short_id] for short_id in self.ids if short_id in self.known]


def openalex_works(records: list[dict[str, Any]]) -> pl.DataFrame:
    """OpenAlex's staging table built from the given metadata records."""
    return build_openalex_works(read_records(records, OPENALEX_METADATA_SCHEMA), {})


def test_fetch_work_metadata_asks_by_short_id_and_dates_every_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []
    known = {"W1": {"id": "https://openalex.org/W1", "doi": None}}
    monkeypatch.setattr(work_metadata, "Works", lambda: FakeQuery(known, calls))
    work_ids = ["https://openalex.org/W2", "https://openalex.org/W1"]
    records = fetch_work_metadata(work_ids, fetched=date(2026, 11, 13))
    assert records == [
        {"id": "https://openalex.org/W2", "fetched": FETCHED, "missing": True},
        {"id": "https://openalex.org/W1", "doi": None, "fetched": FETCHED},
    ]
    assert calls == [
        ("filter_or", {"openalex_id": ["W2", "W1"]}),
        ("select", list(METADATA_FIELDS)),
        ("get", 2),
    ]


def test_fetch_work_metadata_refuses_more_ids_than_one_request_takes() -> None:
    work_ids = [f"https://openalex.org/W{work}" for work in range(IDS_PER_REQUEST + 1)]
    with pytest.raises(ValueError, match="at most"):
        fetch_work_metadata(work_ids, fetched=date(2026, 11, 13))


def test_configure_openalex_sets_the_key_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("api_key", "max_retries", "retry_backoff_factor", "retry_http_codes"):
        monkeypatch.setitem(config, key, config[key])
    configure_openalex("a-key", max_retries=3)
    assert (config.api_key, config.max_retries) == ("a-key", 3)
    assert 429 in config.retry_http_codes


@pytest.mark.parametrize("indexed_in", [(), ("datacite",)], ids=["no index", "DataCite only"])
def test_is_repository_only_flags_a_repository_copy_no_scholarly_index_lists(
    indexed_in: tuple[str, ...],
) -> None:
    record = metadata_record(1, source_type="repository", indexed_in=indexed_in)
    assert openalex_works([record]).filter(is_repository_only()).height == 1


@pytest.mark.parametrize(
    "record",
    [
        metadata_record(1, source_type="repository", indexed_in=("arxiv",)),
        metadata_record(1, source_type="journal", indexed_in=()),
        metadata_record(1, source_type=None, indexed_in=()),
        missing_record(1),
    ],
    ids=["arXiv preprint", "journal", "no source", "missing work"],
)
def test_is_repository_only_spares_other_works(record: dict[str, Any]) -> None:
    assert openalex_works([record]).filter(is_repository_only()).is_empty()


def test_check_work_metadata_refuses_a_work_without_a_record() -> None:
    table = abstracts_table([abstract_row(work, ["Models learn senses."]) for work in (1, 2)])
    with pytest.raises(ValueError, match="1 works have no metadata record"):
        check_work_metadata(table, openalex_works([metadata_record(1)]))


def test_mark_metadata_flags_adds_flags_and_keeps_the_existing_ones() -> None:
    table = abstracts_table(
        [
            abstract_row(1, ["Models learn senses."], flags=[MALFORMED]),
            abstract_row(2, ["Models learn senses."]),
            abstract_row(3, ["Models learn senses."]),
        ]
    )
    works = openalex_works(
        [
            metadata_record(1, source_type="repository", indexed_in=()),
            metadata_record(2),
            missing_record(3),
        ]
    )
    flagged = mark_metadata_flags(table, works)
    assert flagged["flags"].to_list() == [[MALFORMED, REPOSITORY_ONLY], [], [NOT_IN_OPENALEX]]


def test_mark_metadata_flags_marks_front_matter_and_retracted_works() -> None:
    table = abstracts_table([abstract_row(work, ["Models learn senses."]) for work in (1, 2)])
    works = openalex_works(
        [{**metadata_record(1), "is_paratext": True}, {**metadata_record(2), "is_retracted": True}]
    )
    flagged = mark_metadata_flags(table, works)
    assert flagged["flags"].to_list() == [[PARATEXT], [RETRACTED]]


def test_correct_dois_replaces_only_the_corrected_works() -> None:
    records = [
        WorkMetadata.model_validate(
            {**metadata_record(work), "doi": f"https://doi.org/10.1/{work}"}
        )
        for work in (1, 2)
    ]
    correction = DOICorrection(
        work_id="https://openalex.org/W1",
        openalex_doi="https://doi.org/10.1/1",
        doi=None,
        evidence="PubMed lists no DOI",
    )
    corrected = list(correct_dois(records, {correction.work_id: correction}))
    assert [record.doi for record in corrected] == [None, "https://doi.org/10.1/2"]
    assert records[0].doi == "https://doi.org/10.1/1"


def test_build_openalex_works_normalises_the_identifiers_and_drops_bad_issns() -> None:
    record = metadata_record(1)
    record.update(
        doi="https://doi.org/10.1152/JN.1999.81.2.535",
        ids={"pmid": "https://pubmed.ncbi.nlm.nih.gov/123", "pmcid": "https://www.ncbi.nlm.nih.gov/pmc/articles/456"},
        title="Dendrites",
        type="article",
        publication_date="1999-02-01",
        primary_location={"source": {"display_name": "J Neurophysiol", "type": "journal",
                                     "issn_l": "0018-9340", "issn": ["0018-9340", "0016-9340"]}},
    )  # fmt: skip
    rows = build_openalex_works(
        read_records([record, missing_record(2)], OPENALEX_METADATA_SCHEMA), {}
    ).to_dicts()
    found, lost = rows
    assert (found["doi"], found["pmid"], found["pmcid"]) == (
        "10.1152/jn.1999.81.2.535",
        "123",
        "PMC456",
    )
    assert (found["publication_year"], found["issn_l"], found["issns"]) == (
        1999,
        "0018-9340",
        ["0018-9340"],
    )
    assert (lost["is_missing"], lost["doi"], lost["issns"]) == (True, None, [])
