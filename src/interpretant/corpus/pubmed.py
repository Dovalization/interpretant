"""Each work's PubMed record, looked up by its PubMed id with Biopython (decision agenda 4.8).

PubMed holds what its indexers assign (MeSH headings, publication types such as letter, editorial,
comment or published erratum) and its own abstract, which it has for almost no record before the
1970s. Biopython's Entrez module keeps to NCBI's rate limit, retries its transient failures and
parses the XML.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from typing import Any

import polars as pl
from Bio import Entrez

from interpretant.catalog import (
    PUBMED_WORKS_COLUMNS,
    LinkedWork,
    WorkMetadata,
    build_linked_record,
    select_linked_works,
)
from interpretant.corpus.bibliographic import count_pages

PMIDS_PER_REQUEST = 200


def select_pubmed_works(records: Iterable[WorkMetadata]) -> list[LinkedWork]:
    """The works PubMed can be asked about, in order and each once: those with a PubMed id."""
    return select_linked_works(records, _pmid)


def batch_pubmed_works(works: Sequence[LinkedWork]) -> list[list[LinkedWork]]:
    """The works in batches of 200 PubMed ids, one request each."""
    return [
        list(works[start : start + PMIDS_PER_REQUEST])
        for start in range(0, len(works), PMIDS_PER_REQUEST)
    ]


def configure_entrez(*, tool: str, email: str | None) -> None:
    """Tell NCBI which tool is asking and whom to contact about it.

    Biopython keeps these settings on its Entrez module, so this applies to every request.
    """
    Entrez.tool = tool
    if email:
        Entrez.email = email  # type: ignore[assignment]  # Biopython types its default, None


def fetch_pubmed_records(works: Sequence[LinkedWork], *, fetched: date) -> list[dict[str, Any]]:
    """PubMed's records of up to 200 works, in the given order, each with the work id and date.

    A PubMed id the answer leaves out (deleted, or merged into another) gives a record marked
    missing.

    Raises:
        ValueError: if more than 200 works are given.
        urllib.error.HTTPError: if NCBI still refuses after Biopython's retries.
    """
    if len(works) > PMIDS_PER_REQUEST:
        raise ValueError(f"one PubMed request takes at most {PMIDS_PER_REQUEST} ids")
    stream = Entrez.efetch(db="pubmed", id=[work.identifier for work in works], retmode="xml")
    try:
        answer = Entrez.read(stream)
    finally:
        stream.close()
    found = parse_pubmed_articles(answer)
    return [
        build_linked_record(work, found.get(work.identifier), fetched=fetched) for work in works
    ]


def parse_pubmed_articles(answer: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Each article in a parsed PubMed answer, by PubMed id: its abstract, indexing and journal."""
    articles = {}
    for article in answer.get("PubmedArticle", []):
        record = _parse_citation(article["MedlineCitation"])
        articles[record["pmid"]] = record
    for book in answer.get("PubmedBookArticle", []):
        document = book["BookDocument"]
        pmid = str(document["PMID"])
        types = [str(kind) for kind in document.get("PublicationType", [])]
        articles[pmid] = {"pmid": pmid, "is_book": True, "publication_types": types}
    return articles


def build_pubmed_works(records: Iterable[Mapping[str, Any]]) -> pl.DataFrame:
    """PubMed's staging table: one row per stored record, flattened (decision agenda 4.8).

    A structured abstract's parts are joined in order, each after its label ("BACKGROUND: ...");
    the page range also gives its length; MeSH headings keep their descriptors, the major ones
    also on their own.
    """
    columns = {name: dtype for name, dtype in PUBMED_WORKS_COLUMNS.items() if name != "page_count"}
    rows = pl.from_dicts([_pubmed_row(record) for record in records], schema=columns)
    with_pages = rows.with_columns(count_pages("page").alias("page_count"))
    return with_pages.select(PUBMED_WORKS_COLUMNS.names())


def _pmid(record: WorkMetadata) -> str:
    return record.ids.get("pmid", "").rsplit("/", 1)[-1]


def _parse_citation(citation: Any) -> dict[str, Any]:  # noqa: ANN401 (Biopython's parsed XML)
    article = citation["Article"]
    abstract_parts = article.get("Abstract", {}).get("AbstractText", [])
    return {
        "pmid": str(citation["PMID"]),
        "status": citation.attributes.get("Status", ""),
        "title": str(article.get("ArticleTitle", "")),
        "abstract": [
            {
                "label": part.attributes.get("Label", ""),
                "category": part.attributes.get("NlmCategory", ""),
                "text": str(part),
            }
            for part in abstract_parts
        ],
        "publication_types": [str(kind) for kind in article.get("PublicationTypeList", [])],
        "languages": [str(language) for language in article.get("Language", [])],
        "pages": str(article.get("Pagination", {}).get("MedlinePgn", "")),
        "journal": str(article.get("Journal", {}).get("Title", "")),
        "journal_nlm_id": str(citation.get("MedlineJournalInfo", {}).get("NlmUniqueID", "")),
        "mesh_headings": [
            {
                "descriptor": str(heading["DescriptorName"]),
                "descriptor_id": heading["DescriptorName"].attributes.get("UI", ""),
                "is_major": heading["DescriptorName"].attributes.get("MajorTopicYN") == "Y",
                "qualifiers": [str(qualifier) for qualifier in heading.get("QualifierName", [])],
            }
            for heading in citation.get("MeshHeadingList", [])
        ],
        "keywords": [
            str(keyword) for group in citation.get("KeywordList", []) for keyword in group
        ],
        "comments_corrections": [
            {"type": comment.attributes.get("RefType", ""), "pmid": str(comment.get("PMID", ""))}
            for comment in citation.get("CommentsCorrectionsList", [])
        ],
    }


def _pubmed_row(record: Mapping[str, Any]) -> dict[str, object]:
    parts = record.get("abstract", [])
    headings = record.get("mesh_headings", [])
    comments = record.get("comments_corrections", [])
    abstract = " ".join(
        f"{part['label']}: {part['text']}" if part["label"] else part["text"] for part in parts
    )
    return {
        "work_id": record["work_id"],
        "pmid": str(record["identifier"]),
        "is_found": not record.get("missing", False),
        "is_book": bool(record.get("is_book")),
        "status": record.get("status") or None,
        "title": record.get("title") or None,
        "publication_types": record.get("publication_types", []),
        "mesh_descriptors": [heading["descriptor"] for heading in headings],
        "mesh_major_descriptors": [
            heading["descriptor"] for heading in headings if heading["is_major"]
        ],
        "keywords": record.get("keywords", []),
        "languages": record.get("languages", []),
        "abstract": abstract or None,
        "is_structured_abstract": any(part["label"] for part in parts),
        "page": record.get("pages") or None,
        "journal": record.get("journal") or None,
        "journal_nlm_id": record.get("journal_nlm_id") or None,
        "comment_types": sorted({comment["type"] for comment in comments}),
        "fetched": date.fromisoformat(record["fetched"]),
    }
