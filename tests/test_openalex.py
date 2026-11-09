"""Tests for the OpenAlex source."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from interpretant.corpus.openalex import (
    OpenAlexSource,
    abstract_from_inverted_index,
    cell_path,
    download_cell,
    work_record,
)


def make_work(number: int, abstract: str) -> dict[str, object]:
    index: dict[str, list[int]] = {}
    for position, word in enumerate(abstract.split()):
        index.setdefault(word, []).append(position)
    return {
        "id": f"https://openalex.org/W{number}",
        "publication_year": 1995,
        "authorships": [{"author": {"id": f"https://openalex.org/A{number}"}}, {"author": {}}],
        "abstract_inverted_index": index,
    }


class FakeClient:
    """Serves a fixed population of works in pages of two."""

    def __init__(self, works: list[dict[str, object]]) -> None:
        self.works = works
        self.pages_requested: list[int] = []

    def count(self, filters: str) -> int:
        return len(self.works)

    def sample_page(
        self, filters: str, sample: int, seed: int, page: int
    ) -> list[dict[str, object]]:
        self.pages_requested.append(page)
        return self.works[(page - 1) * 200 : page * 200][:sample]


def test_abstract_from_inverted_index_restores_word_order() -> None:
    index = {"the": [0, 3], "network": [1], "learns": [2], "world": [4]}

    assert abstract_from_inverted_index(index) == "the network learns the world"
    assert abstract_from_inverted_index(None) == ""


def test_work_record_keeps_id_year_authors_and_abstract() -> None:
    record = work_record(make_work(7, "neural networks represent"))

    assert record == {
        "id": "https://openalex.org/W7",
        "year": 1995,
        "authors": ["https://openalex.org/A7"],
        "abstract": "neural networks represent",
    }


def test_download_cell_writes_all_pages_and_skips_finished_cells(tmp_path: Path) -> None:
    works = [make_work(number, f"abstract number {number}") for number in range(450)]
    client = FakeClient(works)

    written = download_cell(client, tmp_path, "hci", 1990, sample_size=10_000)  # type: ignore[arg-type]
    again = download_cell(client, tmp_path, "hci", 1990, sample_size=10_000)  # type: ignore[arg-type]

    assert written == again == 450
    assert client.pages_requested == [1, 2, 3]
    assert not list(tmp_path.rglob("*.partial"))


def test_source_yields_preprocessed_abstracts_per_decade(tmp_path: Path) -> None:
    path = cell_path(tmp_path, "hci", 1990)
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt", encoding="utf-8") as cell:
        cell.write(json.dumps({"abstract": "User Interfaces, Revisited"}) + "\n")

    slices = dict(OpenAlexSource(tmp_path, "hci").iter_decade_slices(1980, 2000, 10))

    assert slices == {1980: [], 1990: ["user interface revisited"]}
