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

from interpretant.catalog.linked_records import LinkedWork, build_linked_record, select_linked_works
from interpretant.catalog.metadata import WorkMetadata
from interpretant.corpus.bibliographic import count_pages

PMIDS_PER_REQUEST = 200
_TEXT, _FLAG, _TEXTS = pl.String(), pl.Boolean(), pl.List(pl.String())
# The fields of a stored PubMed record (as parse_pubmed_articles wrote it) the staging table reads.
PUBMED_RECORD_SCHEMA = pl.Schema(
    {
        "work_id": _TEXT,
        "identifier": _TEXT,
        "fetched": _TEXT,
        "missing": _FLAG,
        "is_book": _FLAG,
        "status": _TEXT,
        "title": _TEXT,
        "abstract": pl.List(pl.Struct({"label": _TEXT, "text": _TEXT})),
        "publication_types": _TEXTS,
        "mesh_headings": pl.List(pl.Struct({"descriptor": _TEXT, "is_major": _FLAG})),
        "keywords": _TEXTS,
        "languages": _TEXTS,
        "pages": _TEXT,
        "journal": _TEXT,
        "journal_nlm_id": _TEXT,
        "comments_corrections": pl.List(pl.Struct({"type": _TEXT})),
    }
)


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


def build_pubmed_works(records: pl.DataFrame) -> pl.DataFrame:
    """PubMed's staging table from stored records read with ``PUBMED_RECORD_SCHEMA``.

    One row per record (decision agenda 4.8, layers). A structured abstract's parts are joined
    in order, each after its label ("BACKGROUND: ..."); the page range also gives its length;
    MeSH headings keep their descriptors, the major ones also on their own; empty text is null.
    """
    part = pl.element().struct
    heading = pl.element().struct
    labelled = pl.when(part.field("label") != "").then(
        part.field("label") + ": " + part.field("text")
    )
    return records.select(
        pl.col("work_id"),
        pl.col("identifier").alias("pmid"),
        pl.col("missing").fill_null(value=False).not_().alias("is_found"),
        pl.col("is_book").fill_null(value=False),
        _null_if_empty(pl.col("status")).alias("status"),
        _null_if_empty(pl.col("title")).alias("title"),
        _empty_if_null(pl.col("publication_types")).alias("publication_types"),
        _empty_if_null(pl.col("mesh_headings").list.eval(heading.field("descriptor"))).alias(
            "mesh_descriptors"
        ),
        _empty_if_null(
            pl.col("mesh_headings").list.eval(
                pl.element().filter(heading.field("is_major")).struct.field("descriptor")
            )
        ).alias("mesh_major_descriptors"),
        _empty_if_null(pl.col("keywords")).alias("keywords"),
        _empty_if_null(pl.col("languages")).alias("languages"),
        _null_if_empty(
            pl.col("abstract").list.eval(labelled.otherwise(part.field("text"))).list.join(" ")
        ).alias("abstract"),
        pl.col("abstract")
        .list.eval(part.field("label") != "")
        .list.any()
        .fill_null(value=False)
        .alias("is_structured_abstract"),
        _null_if_empty(pl.col("pages")).alias("page"),
        count_pages("pages").alias("page_count"),
        _null_if_empty(pl.col("journal")).alias("journal"),
        _null_if_empty(pl.col("journal_nlm_id")).alias("journal_nlm_id"),
        _empty_if_null(
            pl.col("comments_corrections")
            .list.eval(pl.element().struct.field("type"))
            .list.unique()
            .list.sort()
        ).alias("comment_types"),
        pl.col("fetched").str.to_date().alias("fetched"),
    )


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


def _null_if_empty(text: pl.Expr) -> pl.Expr:
    return pl.when(text != "").then(text)


def _empty_if_null(texts: pl.Expr) -> pl.Expr:
    return texts.fill_null(pl.lit([], dtype=_TEXTS))
