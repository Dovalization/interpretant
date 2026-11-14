"""Looking works up in PubMed by PubMed id, with fakes in place of Biopython's Entrez calls."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest
from Bio import Entrez

from interpretant.catalog import LinkedWork, WorkMetadata
from interpretant.corpus.pubmed import (
    fetch_pubmed_records,
    parse_pubmed_articles,
    select_pubmed_works,
)
from tests.metadata_records import linked_metadata_record, missing_record

FETCHED = date(2026, 11, 14)


def test_select_pubmed_works_takes_each_work_with_a_pubmed_id_once() -> None:
    records = [
        WorkMetadata.model_validate(record)
        for record in [
            linked_metadata_record(1, pmid=True),
            linked_metadata_record(2),
            missing_record(3),
            linked_metadata_record(4, doi=False, pmid=True),
        ]
    ]
    assert select_pubmed_works(records) == [
        LinkedWork("https://openalex.org/W1", "901"),
        LinkedWork("https://openalex.org/W4", "904"),
    ]


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
