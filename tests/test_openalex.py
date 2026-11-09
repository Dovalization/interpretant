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
    clean_cell,
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
    abstracts = [
        "Script from the WBAP-TV/NBC station in Fort Worth, Texas, relating a news story.",
        "Letter from Sabin, Albert B. to Rous, Peyton dated 1951-10-13.",
        "Users perceive affordances in interfaces.",
        "Users perceive affordances in interfaces.",
        "  ",
        "A letter from the editor would not open a catalogue record.",
    ]
    cell = clean_cell({"abstract": abstract} for abstract in abstracts)

    assert cell.documents == [
        "user perceive affordance in interface",
        "a letter from the editor would not open a catalogue record",
    ]
    assert (cell.catalogue_records, cell.duplicates, cell.empty) == (2, 1, 1)


def test_source_yields_cleaned_abstracts_per_decade(tmp_path: Path) -> None:
    write_cell(
        tmp_path,
        "hci",
        1990,
        [
            "User Interfaces, Revisited for Older Adults",
            "User interfaces revisited for older adults",
        ],
    )

    slices = dict(OpenAlexSource(tmp_path, "hci").iter_decade_slices(1980, 2000, 10))

    assert slices == {1980: [], 1990: ["user interface revisited for older adults"]}


def test_clean_cell_caps_an_author_who_dominates_the_cell() -> None:
    records = [
        {"abstract": f"consortium paper number {n} on imaging", "authors": ["A1"]}
        for n in range(20)
    ]
    records += [
        {"abstract": f"this is the paper number {n} of a single author", "authors": [f"B{n}"]}
        for n in range(80)
    ]

    cell = clean_cell(records, max_author_share=0.05)

    assert cell.author_capped == 15
    assert sum(doc.startswith("consortium") for doc in cell.documents) == 5


def test_clean_cell_drops_scraped_pages_and_citations_and_strips_urls() -> None:
    abstracts = [
        "No AccessJournal of Speech and Hearing ResearchResearch Article 1 Jun 1958",
        "Journal of Communication, Volume 3, Issue 2, 1 November 1953, Pages 120-123",
        "https://www.sil.org/resources/archives/3576",
        "We train a parser on treebanks. Code: https://github.com/x/y. "
        "© 2019 Elsevier B.V. All rights reserved.",
    ]

    cell = clean_cell({"abstract": abstract} for abstract in abstracts)

    assert cell.documents == ["we train a parser on treebanks code"]
    assert (cell.web_pages, cell.catalogue_records, cell.empty) == (1, 1, 1)


def test_clean_cell_keeps_research_about_social_media() -> None:
    abstracts = [
        "As one of the most popular social media platforms, Twitter attracts many journalists.",
        "Researchers had no access to Facebook data before the platform opened its API.",
        "Share Facebook Twitter LinkedIn Reddit Email Export citation Add to favorites",
    ]

    cell = clean_cell({"abstract": abstract} for abstract in abstracts)

    assert len(cell.documents) == 2
    assert cell.web_pages == 1


def test_clean_cell_strips_appended_footers_and_drops_citation_menus() -> None:
    abstracts = [
        "Bodies that learn language become wordlings. This content downloaded from "
        "207.46.13.145 on Wed, 27 Apr 2016 05:09:54 UTC All use subject to "
        "http://about.jstor.org/terms",
        "Capitalism reorganises the working day. Click here to purchase a PDF version of "
        "this article at the Monthly Review website.",
        "Download citation file: Ris (Zotero) Reference Manager EasyBib Bookends Mendeley",
    ]

    cell = clean_cell({"abstract": abstract} for abstract in abstracts)

    assert cell.documents == [
        "bodies that learn language become wordlings",
        "capitalism reorganises the working day",
    ]
    assert cell.web_pages == 1


def test_clean_cell_drops_non_english_abstracts_and_mastheads() -> None:
    abstracts = [
        "Se encuentran interesantes diferencias asi como similitudes en esta exploracion "
        "de los periodistas",
        "Cette etude analyse les representations de la conscience dans les textes du corpus",
        "Editor Roy Mason editor Doug Hodges photographer Marvin Bayles business manager",
        "This study analyses how the representation of consciousness changed in the corpus.",
    ]

    cell = clean_cell({"abstract": abstract} for abstract in abstracts)

    assert cell.documents == [
        "this study analyses how the representation of consciousness changed in the corpus"
    ]
    assert cell.non_english == 3


def test_clean_cell_keeps_ids_aligned_with_the_documents_it_keeps() -> None:
    records = [
        {"id": "W1", "abstract": "we study how people read interfaces on small screens today"},
        {"id": "W2", "abstract": "we study how people read interfaces on small screens today"},
        {"id": "W3", "abstract": "a second abstract about something else entirely in english"},
    ]

    cell = clean_cell(records)

    assert cell.ids == ["W1", "W3"]
    assert len(cell.ids) == len(cell.documents)


def test_sources_for_batches_ids_and_fills_missing_works() -> None:
    from interpretant.corpus.openalex import IDS_PER_REQUEST, OpenAlexClient

    calls: list[dict[str, object]] = []

    class StubClient(OpenAlexClient):
        def _get(self, params: dict[str, object]) -> dict[str, object]:
            calls.append(params)
            ids = str(params["filter"]).split(":", 1)[1].split("|")
            return {
                "results": [
                    {
                        "id": f"https://openalex.org/{work_id}",
                        "primary_location": {"source": {"id": "S1", "display_name": "CHI"}},
                    }
                    for work_id in ids[:-1]
                ]
            }

    ids = [f"https://openalex.org/W{n}" for n in range(IDS_PER_REQUEST + 1)]
    sources = StubClient("key").sources_for(ids)

    assert len(calls) == 2
    assert sources[ids[0]] == ("S1", "CHI")
    assert sources[ids[IDS_PER_REQUEST - 1]] == (None, None)
    assert sources[ids[IDS_PER_REQUEST]] == (None, None)
