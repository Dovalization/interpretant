"""OpenAlex's work metadata: the fetch through pyalex, the repository-only rule and the flags.

pyalex is replaced by a fake query, so the tests never call OpenAlex.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from pyalex import config

from interpretant.catalog import WorkMetadata
from interpretant.corpus import work_metadata
from interpretant.corpus.annotation import MALFORMED
from interpretant.corpus.work_metadata import (
    IDS_PER_REQUEST,
    METADATA_FIELDS,
    NOT_IN_OPENALEX,
    REPOSITORY_ONLY,
    configure_openalex,
    fetch_work_metadata,
    index_work_metadata,
    is_repository_only,
    mark_metadata_flags,
)
from tests.abstracts_tables import abstract_row, abstracts_table
from tests.metadata_records import FETCHED, metadata_record, missing_record


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


def metadata(record: dict[str, Any]) -> WorkMetadata:
    return WorkMetadata.model_validate(record)


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
    assert is_repository_only(metadata(record))


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
    assert not is_repository_only(metadata(record))


def test_index_work_metadata_refuses_a_work_without_a_record() -> None:
    with pytest.raises(ValueError, match="fetch-metadata"):
        index_work_metadata(
            [metadata(metadata_record(1))], ["https://openalex.org/W1", "https://openalex.org/W2"]
        )


def test_mark_metadata_flags_adds_flags_and_keeps_the_existing_ones() -> None:
    table = abstracts_table(
        [
            abstract_row(1, ["Models learn senses."], flags=[MALFORMED]),
            abstract_row(2, ["Models learn senses."]),
            abstract_row(3, ["Models learn senses."]),
        ]
    )
    records = [
        metadata(metadata_record(1, source_type="repository", indexed_in=())),
        metadata(metadata_record(2)),
        metadata(missing_record(3)),
    ]
    flagged = mark_metadata_flags(table, index_work_metadata(records, table["work_id"]))
    assert flagged["flags"].to_list() == [[MALFORMED, REPOSITORY_ONLY], [], [NOT_IN_OPENALEX]]
