"""Each work's Crossref record, looked up by its DOI with habanero (decision agenda 4.8).

Crossref holds what publishers deposit with a DOI: the abstract, the page range, relations to
other records (a preprint and its published version, corrections, retractions) and licences,
text-mining ones included. OpenAlex builds on it but does not say where an abstract came from, so
the corpus rules read these from Crossref itself.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from datetime import date
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote

import polars as pl
from habanero import Crossref

from interpretant.catalog.linked_records import LinkedWork, build_linked_record, select_linked_works
from interpretant.catalog.metadata import WorkMetadata
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
_TEXT, _FLAG, _TEXTS = pl.String(), pl.Boolean(), pl.List(pl.String())
# The fields of a stored Crossref record the staging table reads, as Polars reads them. The
# relation object is read as its JSON text: its keys are relation types, an open list (39 in the
# downloads), which an explicit struct would silently cut to the ones it names.
CROSSREF_RECORD_SCHEMA = pl.Schema(
    {
        "work_id": _TEXT,
        "identifier": _TEXT,
        "fetched": _TEXT,
        "missing": _FLAG,
        "invalid_doi": _FLAG,
        "type": _TEXT,
        "title": _TEXTS,
        "container-title": _TEXTS,
        "ISSN": _TEXTS,
        "published": pl.Struct({"date-parts": pl.List(pl.List(pl.Int64()))}),
        "abstract": _TEXT,
        "page": _TEXT,
        "references-count": pl.Int64(),
        "relation": _TEXT,
        "update-to": pl.List(pl.Struct({"type": _TEXT})),
        "updated-by": _TEXT,
        "license": pl.List(pl.Struct({"URL": _TEXT, "content-version": _TEXT})),
        "link": pl.List(pl.Struct({"URL": _TEXT, "intended-application": _TEXT})),
    }
)


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


def build_crossref_works(records: pl.DataFrame) -> pl.DataFrame:
    """Crossref's staging table from stored records read with ``CROSSREF_RECORD_SCHEMA``.

    One row per record (decision agenda 4.8, layers). The abstract, deposited as JATS markup,
    becomes plain text without its "Abstract" heading and the relation object its types, both
    through Python's standard library, as Polars parses neither; the page range also gives its
    length; updates and licences become their types and addresses, with whether the publisher
    licenses text mining.
    """
    licences = pl.col("license")
    links = pl.col("link")
    return records.select(
        pl.col("work_id"),
        pl.col("identifier").str.to_lowercase().alias("doi"),
        pl.col("missing").fill_null(value=False).not_().alias("is_found"),
        pl.col("invalid_doi").fill_null(value=False).alias("is_invalid_doi"),
        pl.col("type").alias("crossref_type"),
        pl.col("title").list.first().alias("title"),
        pl.col("container-title").list.first().alias("container_title"),
        _empty_if_null(
            pl.col("ISSN").list.eval(pl.element().filter(is_valid_issn(pl.element())))
        ).alias("issns"),
        pl.col("published")
        .struct.field("date-parts")
        .list.first()
        .list.first()
        .alias("published_year"),
        pl.col("abstract")
        .map_elements(parse_jats_text, return_dtype=pl.String(), skip_nulls=True)
        .alias("abstract"),
        pl.col("page"),
        count_pages("page").alias("page_count"),
        pl.col("references-count").alias("references_count"),
        _empty_if_null(
            pl.col("relation").map_elements(_relation_types, return_dtype=_TEXTS, skip_nulls=True)
        ).alias("relation_types"),
        _empty_if_null(
            pl.col("update-to")
            .list.eval(pl.element().struct.field("type"))
            .list.unique()
            .list.sort()
        ).alias("update_types"),
        pl.col("updated-by").is_not_null().alias("is_updated"),
        _empty_if_null(licences.list.eval(pl.element().struct.field("URL"))).alias("licence_urls"),
        licences.list.eval(pl.element().struct.field("content-version") == "tdm")
        .list.any()
        .fill_null(value=False)
        .alias("has_text_mining_licence"),
        _empty_if_null(
            links.list.eval(
                pl.element()
                .filter(pl.element().struct.field("intended-application") == "text-mining")
                .struct.field("URL")
            )
        ).alias("text_mining_links"),
        pl.col("fetched").str.to_date().alias("fetched"),
    )


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


def _relation_types(relation: str) -> list[str]:
    return sorted(json.loads(relation))


def _empty_if_null(texts: pl.Expr) -> pl.Expr:
    return texts.fill_null(pl.lit([], dtype=_TEXTS))
