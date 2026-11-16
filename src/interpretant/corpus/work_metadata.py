"""OpenAlex's record of each work beyond its abstract, fetched by id with pyalex (agenda 4.8).

The raw cells keep only a work's id, year, type, author ids and abstract, so an abstract cannot
be told from a digitised archive record (letters, student newspapers, photo captions, which
OpenAlex types as articles). The record's source and indexes can: such works are flagged
repository-only, and the corpus review decides which kinds are dropped (4b).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import date
from typing import Any

import polars as pl
from pyalex import Works, config

from interpretant.catalog import DOICorrection, WorkMetadata
from interpretant.corpus.bibliographic import is_valid_issn
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
_DOI_PREFIX = "https://doi.org/"
_TEXT, _FLAG, _TEXTS = pl.String(), pl.Boolean(), pl.List(pl.String())
# The fields of a stored metadata record the staging table reads, as Polars reads them.
OPENALEX_METADATA_SCHEMA = pl.Schema(
    {
        "id": _TEXT,
        "missing": _FLAG,
        "doi": _TEXT,
        "ids": pl.Struct({"pmid": _TEXT, "pmcid": _TEXT}),
        "title": _TEXT,
        "type": _TEXT,
        "publication_date": _TEXT,
        "language": _TEXT,
        "primary_location": pl.Struct(
            {
                "source": pl.Struct(
                    {"display_name": _TEXT, "type": _TEXT, "issn_l": _TEXT, "issn": _TEXTS}
                )
            }
        ),
        "indexed_in": _TEXTS,
        "is_paratext": _FLAG,
        "is_retracted": _FLAG,
    }
)
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


def correct_dois(
    records: Iterable[WorkMetadata], corrections: Mapping[str, DOICorrection]
) -> Iterator[WorkMetadata]:
    """The records with the curated DOI corrections applied; the downloaded files are unchanged."""
    for record in records:
        correction = corrections.get(record.work_id)
        yield record if correction is None else record.model_copy(update={"doi": correction.doi})


def build_openalex_works(
    metadata: pl.DataFrame, corrections: Mapping[str, DOICorrection]
) -> pl.DataFrame:
    """OpenAlex's staging table from metadata read with ``OPENALEX_METADATA_SCHEMA``.

    One row per record (decision agenda 4.8, layers), the curated DOI corrections applied. DOIs
    lose their resolver prefix and are lower-cased, as Crossref's table keeps them; PubMed
    Central ids get the ``PMC`` prefix OpenAlex leaves out; an ISSN that fails its check digit
    is left out (59 in the downloads are typos beside the journal's real ones).
    """
    corrected = pl.DataFrame(
        {
            "id": list(corrections),
            "corrected_doi": [correction.doi for correction in corrections.values()],
        },
        schema={"id": pl.String(), "corrected_doi": pl.String()},
    ).with_columns(pl.lit(value=True).alias("is_corrected"))
    source = pl.col("primary_location").struct.field("source")
    doi = pl.when(pl.col("is_corrected")).then(pl.col("corrected_doi")).otherwise(pl.col("doi"))
    normalised_doi = doi.str.strip_prefix(_DOI_PREFIX).str.to_lowercase()
    pmid = pl.col("ids").struct.field("pmid").str.split("/").list.last()
    pmcid = pl.col("ids").struct.field("pmcid").str.split("/").list.last().str.strip_prefix("PMC")
    issn_l = source.struct.field("issn_l")
    return metadata.join(corrected, on="id", how="left", maintain_order="left").select(
        pl.col("id").alias("work_id"),
        pl.col("missing").fill_null(value=False).alias("is_missing"),
        pl.when(normalised_doi != "").then(normalised_doi).alias("doi"),
        pmid.alias("pmid"),
        pl.when(pmcid.is_not_null()).then("PMC" + pmcid).alias("pmcid"),
        pl.col("title"),
        pl.col("type").alias("openalex_type"),
        pl.col("publication_date")
        .str.to_date(strict=False)
        .dt.year()
        .cast(pl.Int64())
        .alias("publication_year"),
        pl.col("language"),
        source.struct.field("display_name").alias("source_name"),
        source.struct.field("type").alias("source_type"),
        pl.when(is_valid_issn(issn_l)).then(issn_l).alias("issn_l"),
        source.struct.field("issn")
        .list.eval(pl.element().filter(is_valid_issn(pl.element())))
        .fill_null(pl.lit([], dtype=_TEXTS))
        .alias("issns"),
        pl.col("indexed_in").fill_null(pl.lit([], dtype=_TEXTS)),
        pl.col("is_paratext").fill_null(value=False),
        pl.col("is_retracted").fill_null(value=False),
    )


def is_repository_only() -> pl.Expr:
    """Whether a staging row's primary copy is in a repository and no scholarly index lists it."""
    listed = pl.col("indexed_in").list.eval(pl.element().is_in(sorted(SCHOLARLY_INDEXES)))
    return (pl.col("source_type") == "repository") & ~listed.list.any()


def check_work_metadata(abstracts: pl.DataFrame, openalex_works: pl.DataFrame) -> None:
    """Check that every abstract's work has a row in OpenAlex's staging table.

    Raises:
        ValueError: if a work has no metadata record (its cell's fetch did not finish).
    """
    unfetched = abstracts.select("work_id").join(
        openalex_works.select("work_id"), on="work_id", how="anti"
    )
    if unfetched.height:
        raise ValueError(
            f"{unfetched.height} works have no metadata record: run `interpretant fetch-metadata`"
        )


def mark_metadata_flags(abstracts: pl.DataFrame, openalex_works: pl.DataFrame) -> pl.DataFrame:
    """Flag works for the review from OpenAlex's staging table.

    The flags: repository-only, no longer returned by OpenAlex, front matter (OpenAlex's
    ``is_paratext``: tables of contents, editorial boards) and retracted.
    """
    flags = {
        REPOSITORY_ONLY: is_repository_only(),
        NOT_IN_OPENALEX: pl.col("is_missing"),
        PARATEXT: pl.col("is_paratext"),
        RETRACTED: pl.col("is_retracted"),
    }
    for flag, condition in flags.items():
        work_ids = openalex_works.filter(condition)["work_id"].to_list()
        abstracts = add_flag(abstracts, work_ids, flag=flag)
    return abstracts
