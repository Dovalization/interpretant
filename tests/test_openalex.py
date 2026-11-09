"""Tests for the OpenAlex source."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from interpretant.corpus.openalex import (
    OpenAlexSource,
    abstract_from_inverted_index,
    cell_filter,
    cell_path,
    clean_abstracts,
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
        "type": "article",
        "authorships": [{"author": {"id": f"https://openalex.org/A{number}"}}, {"author": {}}],
        "abstract_inverted_index": index,
    }


class FakeClient:
    """Serves a fixed population of works in pages of 200, by sample page or by cursor."""

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

    def cursor_page(self, filters: str, cursor: str) -> tuple[list[dict[str, object]], str | None]:
        start = 0 if cursor == "*" else int(cursor)
        page = self.works[start : start + 200]
        return page, str(start + 200) if page else None


def write_cell(raw_dir: Path, field: str, decade: int, abstracts: list[str]) -> None:
    path = cell_path(raw_dir, field, decade)
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt", encoding="utf-8") as cell:
        for abstract in abstracts:
            cell.write(json.dumps({"abstract": abstract}) + "\n")


def test_abstract_from_inverted_index_restores_word_order() -> None:
    index = {"the": [0, 3], "network": [1], "learns": [2], "world": [4]}

    assert abstract_from_inverted_index(index) == "the network learns the world"
    assert abstract_from_inverted_index(None) == ""


def test_work_record_keeps_id_year_type_authors_and_abstract() -> None:
    record = work_record(make_work(7, "neural networks represent"))

    assert record == {
        "id": "https://openalex.org/W7",
        "year": 1995,
        "type": "article",
        "authors": ["https://openalex.org/A7"],
        "abstract": "neural networks represent",
    }


def test_cell_filter_restricts_to_research_prose() -> None:
    filters = cell_filter("hci", 1990)

    assert "publication_year:1990-1999" in filters
    assert "type:article|review|book-chapter|book|dissertation|conference-paper" in filters


def test_download_cell_writes_all_pages_and_skips_finished_cells(tmp_path: Path) -> None:
    works = [make_work(number, f"abstract number {number}") for number in range(450)]
    client = FakeClient(works)

    written = download_cell(client, tmp_path, "hci", 1990, sample_size=10_000)  # type: ignore[arg-type]
    again = download_cell(client, tmp_path, "hci", 1990, sample_size=10_000)  # type: ignore[arg-type]

    assert written == again == 450
    assert client.pages_requested == [1, 2, 3]
    assert not list(tmp_path.rglob("*.partial"))


def test_full_download_follows_the_cursor_past_the_sample_cap(tmp_path: Path) -> None:
    works = [make_work(number, f"abstract number {number}") for number in range(450)]

    written = download_cell(FakeClient(works), tmp_path, "hci", 2000, sample_size=100, full=True)  # type: ignore[arg-type]

    assert written == 450


def test_clean_abstracts_drops_duplicates_catalogue_records_and_empty_text() -> None:
    cell = clean_abstracts(
        [
            "Script from the WBAP-TV/NBC station in Fort Worth, Texas, relating a news story.",
            "Letter from Sabin, Albert B. to Rous, Peyton dated 1951-10-13.",
            "Users perceive affordances in interfaces.",
            "Users perceive affordances in interfaces.",
            "  ",
            "A letter from the editor would not open a catalogue record.",
        ]
    )

    assert cell.documents == [
        "user perceive affordance in interface",
        "a letter from the editor would not open a catalogue record",
    ]
    assert (cell.catalogue_records, cell.duplicates, cell.empty) == (2, 1, 1)


def test_source_yields_cleaned_abstracts_per_decade(tmp_path: Path) -> None:
    write_cell(tmp_path, "hci", 1990, ["User Interfaces, Revisited", "User interfaces revisited"])

    slices = dict(OpenAlexSource(tmp_path, "hci").iter_decade_slices(1980, 2000, 10))

    assert slices == {1980: [], 1990: ["user interface revisited"]}
