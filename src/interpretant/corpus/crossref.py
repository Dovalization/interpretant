"""Each work's Crossref record, looked up by its DOI with habanero (decision agenda 4.8).

Crossref holds what publishers deposit with a DOI: the abstract, the page range, relations to
other records (a preprint and its published version, corrections, retractions) and licences,
text-mining ones included. OpenAlex builds on it but does not say where an abstract came from, so
the corpus rules read these from Crossref itself.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote

import polars as pl
from habanero import Crossref

from interpretant.catalog import (
    CROSSREF_WORKS_COLUMNS,
    LinkedWork,
    WorkMetadata,
    build_linked_record,
    select_linked_works,
)
from interpretant.corpus.bibliographic import count_pages, is_valid_issn

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
# A filter of 200 DOIs answers in about a second. Crossref refuses a request whose address is too
# long (431, seen above about 7,300 characters with long DOIs), so a batch also stops at this many
# characters of encoded filter.
DOIS_PER_REQUEST = 200
MAX_FILTER_CHARACTERS = 6_000
_DOI_PREFIX = "https://doi.org/"
# Crossref refuses a whole batch if one DOI lacks its form: "10.", a prefix of 4 or more digits, a
# slash and a suffix. Two works in OpenAlex carry a 3-digit prefix.
_DOI_FORM = re.compile(r"^10\.\d{4,}(?:\.\d+)*/\S+$")


def select_crossref_works(records: Iterable[WorkMetadata]) -> list[LinkedWork]:
    """The works Crossref can be asked about, in order and each once: those with a DOI."""
    return select_linked_works(records, _doi)


def batch_crossref_works(works: Sequence[LinkedWork]) -> list[list[LinkedWork]]:
    """The works in request-sized batches: at most 200 DOIs and 6,000 characters of filter each."""
    batches: list[list[LinkedWork]] = []
    characters = 0
    for work in works:
        size = len(quote(f"doi:{work.identifier},", safe=""))
        is_full = bool(batches) and (
            len(batches[-1]) == DOIS_PER_REQUEST or characters + size > MAX_FILTER_CHARACTERS
        )
        if not batches or is_full:
            batches.append([])
            characters = 0
        batches[-1].append(work)
        characters += size
    return batches


def fetch_crossref_records(
    crossref: Crossref, works: Sequence[LinkedWork], *, fetched: date
) -> list[dict[str, Any]]:
    """Crossref's records of up to 200 works, in the given order, each with the work id and date.

    A DOI Crossref does not hold (registered with another agency, or unknown) gives a record
    marked missing. A DOI without Crossref's form is not sent: its record is marked missing and
    ``invalid_doi``. A comma inside a DOI would split the batch filter, so such a DOI is asked for
    on its own.

    Raises:
        ValueError: if more than 200 works are given.
        habanero.exceptions.RequestError: if Crossref refuses a request.
    """
    if len(works) > DOIS_PER_REQUEST:
        raise ValueError(f"one Crossref request takes at most {DOIS_PER_REQUEST} DOIs")
    valid = [work.identifier for work in works if _DOI_FORM.match(work.identifier)]
    batched = [doi for doi in valid if "," not in doi]
    found: dict[str, dict[str, Any]] = {}
    if batched:
        answer = crossref.works(
            filter={"doi": batched}, select=list(CROSSREF_FIELDS), limit=len(batched)
        )
        found.update({item["DOI"].lower(): item for item in answer["message"]["items"]})
    for doi in valid:
        if "," in doi:
            found.update(_fetch_crossref_work(crossref, doi))
    return [_record(work, found, fetched=fetched) for work in works]


def build_crossref_works(records: Iterable[Mapping[str, Any]]) -> pl.DataFrame:
    """Crossref's staging table: one row per stored record, flattened (decision agenda 4.8).

    The abstract, deposited as JATS markup, becomes plain text without its "Abstract" heading;
    the page range also gives its length; relations, updates and licences become their types
    and addresses, with whether the publisher licenses text mining.
    """
    columns = {
        name: dtype for name, dtype in CROSSREF_WORKS_COLUMNS.items() if name != "page_count"
    }
    rows = pl.from_dicts([_crossref_row(record) for record in records], schema=columns)
    with_pages = rows.with_columns(count_pages("page").alias("page_count"))
    return with_pages.select(CROSSREF_WORKS_COLUMNS.names())


def parse_jats_text(markup: str) -> str:
    """The plain text of a JATS abstract: its paragraphs' words, without titles or tags."""
    parser = _JATSText()
    parser.feed(markup)
    parser.close()
    return " ".join(" ".join(parser.parts).split())


def _record(work: LinkedWork, found: dict[str, dict[str, Any]], *, fetched: date) -> dict[str, Any]:
    record = build_linked_record(work, found.get(work.identifier.lower()), fetched=fetched)
    if not _DOI_FORM.match(work.identifier):
        record["invalid_doi"] = True
    return record


def _doi(record: WorkMetadata) -> str:
    return (record.doi or "").removeprefix(_DOI_PREFIX)


def _fetch_crossref_work(crossref: Crossref, doi: str) -> dict[str, dict[str, Any]]:
    answer = crossref.works(ids=doi, warn=True)
    if answer is None:
        return {}
    item = answer["message"]
    return {doi.lower(): {field: item[field] for field in CROSSREF_FIELDS if field in item}}


class _JATSText(HTMLParser):
    """Collects the text of a JATS fragment, skipping its titles."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._title_depth = 0

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        if tag.endswith("title"):
            self._title_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.endswith("title") and self._title_depth:
            self._title_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._title_depth:
            self.parts.append(data)


def _crossref_row(record: Mapping[str, Any]) -> dict[str, object]:
    published = record.get("published", {}).get("date-parts", [[None]])
    licences = record.get("license", [])
    abstract = record.get("abstract")
    return {
        "work_id": record["work_id"],
        "doi": str(record["identifier"]).lower(),
        "is_found": not record.get("missing", False),
        "is_invalid_doi": bool(record.get("invalid_doi")),
        "crossref_type": record.get("type"),
        "title": _first(record.get("title")),
        "container_title": _first(record.get("container-title")),
        "issns": [issn for issn in record.get("ISSN", []) if is_valid_issn(issn)],
        "published_year": published[0][0] if published and published[0] else None,
        "abstract": parse_jats_text(abstract) if abstract else None,
        "page": record.get("page"),
        "references_count": record.get("references-count"),
        "relation_types": sorted(record.get("relation", {})),
        "update_types": sorted({update["type"] for update in record.get("update-to", [])}),
        "is_updated": bool(record.get("updated-by")),
        "licence_urls": [licence["URL"] for licence in licences],
        "has_text_mining_licence": any(
            licence.get("content-version") == "tdm" for licence in licences
        ),
        "text_mining_links": [
            link["URL"]
            for link in record.get("link", [])
            if link.get("intended-application") == "text-mining"
        ],
        "fetched": date.fromisoformat(record["fetched"]),
    }


def _first(values: Sequence[str] | None) -> str | None:
    return values[0] if values else None
