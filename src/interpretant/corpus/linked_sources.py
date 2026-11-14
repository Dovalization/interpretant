"""Each work's record in Crossref and PubMed, looked up by its DOI or its PubMed id.

Crossref holds what publishers deposit with a DOI (the abstract, the page range, relations to
other records, licences); PubMed holds what its indexers assign (MeSH headings, publication types)
and its own abstract. OpenAlex builds on both but says neither where an abstract came from nor
keeps the indexers' types, so the corpus rules read them from the sources (decision agenda 4.8,
linked sources). habanero builds Crossref's requests; Biopython's Entrez module fetches PubMed's
records, keeping to NCBI's rate limit and retrying its transient failures, and parses them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from Bio import Entrez
from habanero import Crossref

from interpretant.catalog import LinkedSource, WorkMetadata

# Crossref's works route rejects a field it does not offer. Reference lists are left out:
# OpenAlex already holds each work's matched references, and they are most of a record's size.
CROSSREF_FIELDS = (
    "DOI",
    "type",
    "title",
    "container-title",
    "ISSN",
    "published",
    "abstract",
    "page",
    "relation",
    "update-to",
    "updated-by",
    "license",
    "link",
    "references-count",
)
# A filter of 200 DOIs answers in about a second; 300 make the request's address too long.
DOIS_PER_REQUEST = 200
PMIDS_PER_REQUEST = 200
_DOI_PREFIX = "https://doi.org/"


@dataclass(frozen=True)
class LinkedWork:
    """A work and the identifier a source knows it by: its DOI, or its PubMed id."""

    work_id: str
    identifier: str


def select_linked_works(records: Iterable[WorkMetadata], source: LinkedSource) -> list[LinkedWork]:
    """The works a source can be asked about, in order and each once: those with its identifier."""
    works: dict[str, LinkedWork] = {}
    for record in records:
        identifier = "" if record.missing else _identifier(record, source)
        if identifier and record.work_id not in works:
            works[record.work_id] = LinkedWork(record.work_id, identifier)
    return list(works.values())


def configure_entrez(*, tool: str, email: str | None) -> None:
    """Tell NCBI which tool is asking and whom to contact about it.

    Biopython keeps these settings on its Entrez module, so this applies to every request.
    """
    Entrez.tool = tool
    if email:
        Entrez.email = email  # type: ignore[assignment]  # Biopython types its default, None


def fetch_crossref_records(
    crossref: Crossref, works: Sequence[LinkedWork], *, fetched: date
) -> list[dict[str, Any]]:
    """Crossref's records of up to 200 works, in the given order, each with the work id and date.

    A DOI Crossref does not hold (registered with another agency, or unknown) gives a record
    marked missing. A comma inside a DOI would split the batch filter, so such a DOI is asked for
    on its own.

    Raises:
        ValueError: if more than 200 works are given.
        habanero.exceptions.RequestError: if Crossref refuses a request.
    """
    if len(works) > DOIS_PER_REQUEST:
        raise ValueError(f"one Crossref request takes at most {DOIS_PER_REQUEST} DOIs")
    batched = [work.identifier for work in works if "," not in work.identifier]
    found: dict[str, dict[str, Any]] = {}
    if batched:
        answer = crossref.works(
            filter={"doi": batched}, select=list(CROSSREF_FIELDS), limit=len(batched)
        )
        found.update({item["DOI"].lower(): item for item in answer["message"]["items"]})
    for work in works:
        if "," in work.identifier:
            found.update(_fetch_crossref_work(crossref, work.identifier))
    return [_linked_record(work, found.get(work.identifier.lower()), fetched) for work in works]


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
    return [_linked_record(work, found.get(work.identifier), fetched) for work in works]


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


def _identifier(record: WorkMetadata, source: LinkedSource) -> str:
    if source == "crossref":
        return (record.doi or "").removeprefix(_DOI_PREFIX)
    return record.ids.get("pmid", "").rsplit("/", 1)[-1]


def _fetch_crossref_work(crossref: Crossref, doi: str) -> dict[str, dict[str, Any]]:
    answer = crossref.works(ids=doi, warn=True)
    if answer is None:
        return {}
    item = answer["message"]
    return {doi.lower(): {field: item[field] for field in CROSSREF_FIELDS if field in item}}


def _linked_record(
    work: LinkedWork, item: Mapping[str, Any] | None, fetched: date
) -> dict[str, Any]:
    record = {
        "work_id": work.work_id,
        "identifier": work.identifier,
        "fetched": fetched.isoformat(),
    }
    if item is None:
        return {**record, "missing": True}
    return {**record, **item}


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
