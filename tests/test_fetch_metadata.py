"""The fetch-metadata pipeline and command, with a fake fetch in place of OpenAlex."""

from __future__ import annotations

import gzip
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from interpretant.__main__ import cli
from interpretant.catalog import CatalogConfig, RawCell, fetched_work_ids, load_work_metadata
from interpretant.cli import fetch_metadata as fetch_metadata_cli
from interpretant.pipelines.fetch_metadata import (
    OpenAlexSettings,
    fetch_cell_metadata,
    select_raw_cells,
)
from tests.metadata_records import metadata_record, missing_record


def write_raw_cell(catalog: CatalogConfig, field: str, decade: int, works: range) -> None:
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


class FakeFetch:
    """Returns a record per id, missing for the ids it is told OpenAlex lacks."""

    def __init__(self, missing: frozenset[int] = frozenset()) -> None:
        self.missing = missing
        self.batches: list[int] = []

    def __call__(self, work_ids: Sequence[str]) -> list[dict[str, Any]]:
        self.batches.append(len(work_ids))
        works = [int(work_id.rsplit("W", 1)[1]) for work_id in work_ids]
        return [
            missing_record(work) if work in self.missing else metadata_record(work)
            for work in works
        ]


def test_fetch_cell_metadata_fetches_in_batches_and_resumes(catalog: CatalogConfig) -> None:
    write_raw_cell(catalog, "ai", 2010, range(1, 151))
    fetch = FakeFetch(missing=frozenset({7}))
    result = fetch_cell_metadata(catalog, RawCell("ai", 2010), fetch)
    assert fetch.batches == [100, 50]
    assert (result.already_fetched, result.fetched, result.missing) == (0, 149, 1)
    assert len(fetched_work_ids(catalog, RawCell("ai", 2010))) == 150
    again = fetch_cell_metadata(catalog, RawCell("ai", 2010), fetch)
    assert (again.already_fetched, again.fetched, fetch.batches) == (150, 0, [100, 50])


def test_fetch_cell_metadata_keeps_the_cell_order_with_requests_in_parallel(
    catalog: CatalogConfig,
) -> None:
    write_raw_cell(catalog, "ai", 2010, range(1, 351))
    fetch = FakeFetch()

    def first_batch_slowest(work_ids: Sequence[str]) -> list[dict[str, Any]]:
        if work_ids[0] == "https://openalex.org/W1":
            time.sleep(0.2)
        return fetch(work_ids)

    result = fetch_cell_metadata(catalog, RawCell("ai", 2010), first_batch_slowest, workers=4)
    written = [record.work_id for record in load_work_metadata(catalog, RawCell("ai", 2010))]
    assert written == [f"https://openalex.org/W{work}" for work in range(1, 351)]
    assert (result.fetched, sorted(fetch.batches)) == (350, [50, 100, 100, 100])


def test_fetch_cell_metadata_keeps_the_batches_before_a_failed_request(
    catalog: CatalogConfig,
) -> None:
    write_raw_cell(catalog, "ai", 2010, range(1, 301))
    fetch = FakeFetch()

    def second_batch_fails(work_ids: Sequence[str]) -> list[dict[str, Any]]:
        if work_ids[0] == "https://openalex.org/W101":
            raise ConnectionError("OpenAlex did not answer")
        return fetch(work_ids)

    with pytest.raises(ConnectionError):
        fetch_cell_metadata(catalog, RawCell("ai", 2010), second_batch_fails, workers=2)
    assert len(fetched_work_ids(catalog, RawCell("ai", 2010))) == 100


def test_select_raw_cells_takes_every_cell_unless_told_otherwise(catalog: CatalogConfig) -> None:
    for field, decade in [("ai", 2010), ("ai", 2020), ("compsci", 2010)]:
        write_raw_cell(catalog, field, decade, range(1, 2))
    assert len(select_raw_cells(catalog, fields=(), decades=())) == 3
    assert select_raw_cells(catalog, fields=("ai",), decades=(2020,)) == [RawCell("ai", 2020)]
    with pytest.raises(ValueError, match="no raw cell"):
        select_raw_cells(catalog, fields=("hci",), decades=())


def test_fetch_metadata_command_prints_a_line_per_cell(
    catalog: CatalogConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_raw_cell(catalog, "ai", 2010, range(1, 4))
    keys: list[str] = []

    def fake_fetcher(settings: OpenAlexSettings, *, today: object) -> FakeFetch:
        keys.append(settings.openalex_api_key)
        return FakeFetch()

    monkeypatch.setenv("OPENALEX_API_KEY", "a-test-key")
    monkeypatch.setattr(fetch_metadata_cli, "openalex_fetcher", fake_fetcher)
    arguments = ["fetch-metadata", "--catalog", str(tmp_path / "catalog.yaml")]
    outcome = CliRunner().invoke(cli, arguments)
    assert outcome.exit_code == 0, outcome.output
    assert "[1/1] ai 2010: 3 fetched, 0 missing, 0 already there" in outcome.output
    assert keys == ["a-test-key"]
