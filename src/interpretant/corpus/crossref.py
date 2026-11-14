"""Each work's Crossref record, looked up by its DOI with habanero (decision agenda 4.8).

Crossref holds what publishers deposit with a DOI: the abstract, the page range, relations to
other records (a preprint and its published version, corrections, retractions) and licences,
text-mining ones included. OpenAlex builds on it but does not say where an abstract came from, so
the corpus rules read these from Crossref itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date
from typing import Any
from urllib.parse import quote

from habanero import Crossref

from interpretant.catalog import LinkedWork, WorkMetadata, build_linked_record, select_linked_works

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
    return [
        build_linked_record(work, found.get(work.identifier.lower()), fetched=fetched)
        for work in works
    ]


def _doi(record: WorkMetadata) -> str:
    return (record.doi or "").removeprefix(_DOI_PREFIX)


def _fetch_crossref_work(crossref: Crossref, doi: str) -> dict[str, dict[str, Any]]:
    answer = crossref.works(ids=doi, warn=True)
    if answer is None:
        return {}
    item = answer["message"]
    return {doi.lower(): {field: item[field] for field in CROSSREF_FIELDS if field in item}}
