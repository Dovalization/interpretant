"""OpenAlex's record of each work beyond its abstract, fetched by id with pyalex (agenda 4.8).

The raw cells keep only a work's id, year, type, author ids and abstract, so an abstract cannot
be told from a digitised archive record (letters, student newspapers, photo captions, which
OpenAlex types as articles). The record's source and indexes can: such works are flagged
repository-only, and the corpus review decides which kinds are dropped (4b).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from typing import Any

import polars as pl
from pyalex import Works, config

from interpretant.catalog import WorkMetadata
from interpretant.corpus.cleaning import add_flag

# The fields of decision agenda 4.8: everything the corpus rules, the review and later measures
# may read, so the corpus is fetched once.
METADATA_FIELDS = (
    "id",
    "doi",
    "title",
    "publication_date",
    "ids",
    "language",
    "primary_location",
    "locations",
    "locations_count",
    "type",
    "indexed_in",
    "is_paratext",
    "is_retracted",
    "authorships",
    "primary_topic",
    "topics",
    "keywords",
    "referenced_works",
    "referenced_works_count",
    "cited_by_count",
    "counts_by_year",
    "updated_date",
)
# OpenAlex takes at most 100 values in one OR filter.
IDS_PER_REQUEST = 100
REPOSITORY_ONLY = "repository_only"
NOT_IN_OPENALEX = "not_in_openalex"
PARATEXT = "paratext"
RETRACTED = "retracted"
# Indexes of scholarly publications. DataCite is left out: it lists research data and archive
# records alike, so the review looks at it first (decision agenda 4b).
SCHOLARLY_INDEXES = frozenset({"crossref", "pubmed", "arxiv", "doaj"})


def configure_openalex(api_key: str, *, max_retries: int = 5) -> None:
    """Give pyalex the API key and retry rate-limit and server errors with a growing pause.

    pyalex keeps its settings in one module-level object, so this applies to every request.
    """
    config.api_key = api_key
    config.max_retries = max_retries
    config.retry_backoff_factor = 1.0
    config.retry_http_codes = [429, 500, 502, 503, 504]


def fetch_work_metadata(work_ids: Sequence[str], *, fetched: date) -> list[dict[str, Any]]:
    """OpenAlex's records of up to 100 works, in the given order, each with the fetch date.

    A work OpenAlex does not return (withdrawn, or merged into another id) gets a record marked
    missing.

    Raises:
        ValueError: if more than 100 ids are given.
        requests.HTTPError: if a request still fails after pyalex's retries.
    """
    if len(work_ids) > IDS_PER_REQUEST:
        raise ValueError(f"OpenAlex takes at most {IDS_PER_REQUEST} ids per request")
    short_ids = [work_id.rsplit("/", 1)[-1] for work_id in work_ids]
    results = (
        Works()
        .filter_or(openalex_id=short_ids)
        .select(list(METADATA_FIELDS))
        .get(per_page=len(short_ids))
    )
    found = {str(work["id"]): dict(work) for work in results}
    return [
        {**found[work_id], "fetched": fetched.isoformat()}
        if work_id in found
        else {"id": work_id, "fetched": fetched.isoformat(), "missing": True}
        for work_id in work_ids
    ]


def is_repository_only(metadata: WorkMetadata) -> bool:
    """Whether a work's primary copy is in a repository and no scholarly index lists it."""
    location = metadata.primary_location
    source_type = location.source.type if location and location.source else None
    return source_type == "repository" and not SCHOLARLY_INDEXES.intersection(metadata.indexed_in)


def index_work_metadata(
    records: Iterable[WorkMetadata], work_ids: Iterable[str]
) -> dict[str, WorkMetadata]:
    """Each work's metadata record, by work id.

    Raises:
        ValueError: if a work has no record (its cell's fetch did not finish).
    """
    by_work = {record.work_id: record for record in records}
    unfetched = set(work_ids) - by_work.keys()
    if unfetched:
        raise ValueError(
            f"{len(unfetched)} works have no metadata record: run `interpretant fetch-metadata`"
        )
    return by_work


def mark_metadata_flags(
    abstracts: pl.DataFrame, metadata: Mapping[str, WorkMetadata]
) -> pl.DataFrame:
    """Flag works for the review from their metadata.

    The flags: repository-only, no longer returned by OpenAlex, front matter (OpenAlex's
    ``is_paratext``: tables of contents, editorial boards) and retracted.
    """
    flags = {
        REPOSITORY_ONLY: [work for work, record in metadata.items() if is_repository_only(record)],
        NOT_IN_OPENALEX: [work for work, record in metadata.items() if record.missing],
        PARATEXT: [work for work, record in metadata.items() if record.is_paratext],
        RETRACTED: [work for work, record in metadata.items() if record.is_retracted],
    }
    for flag, work_ids in flags.items():
        abstracts = add_flag(abstracts, work_ids, flag=flag)
    return abstracts
