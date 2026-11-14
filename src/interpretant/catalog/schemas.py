"""The schemas the catalog checks tables against when it writes and reads them."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.polars import PolarsData

SENTENCE = pl.Struct(
    {"start": pl.Int64(), "end": pl.Int64(), "language": pl.String(), "confidence": pl.Float64()}
)
REMOVAL = pl.Struct({"start": pl.Int64(), "end": pl.Int64(), "reason": pl.String()})

ABSTRACTS_COLUMNS = pl.Schema(
    {
        "work_id": pl.String(),
        "year": pl.Int64(),
        "decade": pl.Int64(),
        "fields": pl.List(pl.String()),
        "source": pl.String(),
        "document_type": pl.String(),
        "author_ids": pl.List(pl.String()),
        "text": pl.String(),
        "sentences": pl.List(SENTENCE),
        "removals": pl.List(REMOVAL),
        "flags": pl.List(pl.String()),
        "dropped_reason": pl.String(),
        "duplicate_of": pl.String(),
    }
)


def _spans_are_valid(column: str) -> list[pl.Expr]:
    """Every span starts at or after 0, before its end, and ends inside the text."""
    span = pl.element().struct
    starts_before_ends = (
        pl.col(column)
        .list.eval((span.field("start") >= 0) & (span.field("start") < span.field("end")))
        .list.all()
        .alias(f"{column}_start_before_end")
    )
    last_end = pl.col(column).list.eval(span.field("end")).list.max().fill_null(0)
    ends_inside_text = (last_end <= pl.col("text").str.len_chars()).alias(f"{column}_inside_text")
    return [starts_before_ends, ends_inside_text]


def _decade_is_the_year_rounded_down(data: PolarsData) -> pl.LazyFrame:
    checked: pl.LazyFrame = data.lazyframe.select(
        pl.col("decade") == pl.col("year") - pl.col("year") % 10
    )
    return checked


def _every_work_has_a_field(data: PolarsData) -> pl.LazyFrame:
    checked: pl.LazyFrame = data.lazyframe.select(pl.col("fields").list.len() > 0)
    return checked


def _spans_lie_inside_the_text(data: PolarsData) -> pl.LazyFrame:
    checked: pl.LazyFrame = data.lazyframe.select(
        *_spans_are_valid("sentences"), *_spans_are_valid("removals")
    )
    return checked


ABSTRACTS_SCHEMA = pa.DataFrameSchema(
    {
        "work_id": pa.Column(
            ABSTRACTS_COLUMNS["work_id"],
            unique=True,
            checks=pa.Check.str_startswith("https://openalex.org/W"),
        ),
        "year": pa.Column(ABSTRACTS_COLUMNS["year"], checks=pa.Check.in_range(1900, 2100)),
        "decade": pa.Column(ABSTRACTS_COLUMNS["decade"]),
        "fields": pa.Column(ABSTRACTS_COLUMNS["fields"]),
        "source": pa.Column(
            ABSTRACTS_COLUMNS["source"],
            checks=pa.Check.str_matches(r"^(regular|download:[a-z0-9_-]+)$"),
        ),
        "document_type": pa.Column(ABSTRACTS_COLUMNS["document_type"]),
        "author_ids": pa.Column(ABSTRACTS_COLUMNS["author_ids"]),
        "text": pa.Column(ABSTRACTS_COLUMNS["text"]),
        "sentences": pa.Column(ABSTRACTS_COLUMNS["sentences"]),
        "removals": pa.Column(ABSTRACTS_COLUMNS["removals"]),
        "flags": pa.Column(ABSTRACTS_COLUMNS["flags"]),
        "dropped_reason": pa.Column(ABSTRACTS_COLUMNS["dropped_reason"], nullable=True),
        "duplicate_of": pa.Column(
            ABSTRACTS_COLUMNS["duplicate_of"],
            nullable=True,
            checks=pa.Check.str_startswith("https://openalex.org/W"),
        ),
    },
    checks=[
        pa.Check(_decade_is_the_year_rounded_down, name="decade is the year rounded down"),
        pa.Check(_every_work_has_a_field, name="every work has a field"),
        pa.Check(_spans_lie_inside_the_text, name="spans lie inside the text"),
    ],
    strict=True,
    name="abstracts",
    description=(
        "One row per work, its repaired text never edited after repair. Removals are labelled "
        "character ranges into the text (stand-off annotation, decision agenda 2.2); "
        "dropped_reason is set when a keeping rule drops the whole abstract, so dropped "
        "abstracts stay available for the corpus review (4b); duplicate_of names the earlier "
        "work a near-duplicate was dropped for, and flags mark abstracts left to the review."
    ),
)

OPENALEX_WORKS_COLUMNS = pl.Schema(
    {
        "work_id": pl.String(),
        "is_missing": pl.Boolean(),
        "doi": pl.String(),
        "pmid": pl.String(),
        "pmcid": pl.String(),
        "title": pl.String(),
        "openalex_type": pl.String(),
        "publication_year": pl.Int64(),
        "language": pl.String(),
        "source_name": pl.String(),
        "source_type": pl.String(),
        "issn_l": pl.String(),
        "issns": pl.List(pl.String()),
        "indexed_in": pl.List(pl.String()),
        "is_paratext": pl.Boolean(),
        "is_retracted": pl.Boolean(),
    }
)
CROSSREF_WORKS_COLUMNS = pl.Schema(
    {
        "work_id": pl.String(),
        "doi": pl.String(),
        "is_found": pl.Boolean(),
        "is_invalid_doi": pl.Boolean(),
        "crossref_type": pl.String(),
        "title": pl.String(),
        "container_title": pl.String(),
        "issns": pl.List(pl.String()),
        "published_year": pl.Int64(),
        "abstract": pl.String(),
        "page": pl.String(),
        "page_count": pl.Int64(),
        "references_count": pl.Int64(),
        "relation_types": pl.List(pl.String()),
        "update_types": pl.List(pl.String()),
        "is_updated": pl.Boolean(),
        "licence_urls": pl.List(pl.String()),
        "has_text_mining_licence": pl.Boolean(),
        "text_mining_links": pl.List(pl.String()),
        "fetched": pl.Date(),
    }
)
PUBMED_WORKS_COLUMNS = pl.Schema(
    {
        "work_id": pl.String(),
        "pmid": pl.String(),
        "is_found": pl.Boolean(),
        "is_book": pl.Boolean(),
        "status": pl.String(),
        "title": pl.String(),
        "publication_types": pl.List(pl.String()),
        "mesh_descriptors": pl.List(pl.String()),
        "mesh_major_descriptors": pl.List(pl.String()),
        "keywords": pl.List(pl.String()),
        "languages": pl.List(pl.String()),
        "abstract": pl.String(),
        "is_structured_abstract": pl.Boolean(),
        "page": pl.String(),
        "page_count": pl.Int64(),
        "journal": pl.String(),
        "journal_nlm_id": pl.String(),
        "comment_types": pl.List(pl.String()),
        "fetched": pl.Date(),
    }
)


def _staging_schema(name: str, columns: pl.Schema, description: str) -> pa.DataFrameSchema:
    """A staging table's schema: its columns, all nullable but the work id, one row per work."""
    checked = {
        column: pa.Column(dtype, nullable=column != "work_id") for column, dtype in columns.items()
    }
    checked["work_id"] = pa.Column(
        columns["work_id"], unique=True, checks=pa.Check.str_startswith("https://openalex.org/W")
    )
    return pa.DataFrameSchema(checked, strict=True, name=name, description=description)


OPENALEX_WORKS_SCHEMA = _staging_schema(
    "openalex_works",
    OPENALEX_WORKS_COLUMNS,
    "One row per downloaded work: OpenAlex's metadata flattened, with the curated DOI "
    "corrections applied, PubMed Central ids with their PMC prefix, and only the ISSNs that "
    "pass their check digit.",
)
CROSSREF_WORKS_SCHEMA = _staging_schema(
    "crossref_works",
    CROSSREF_WORKS_COLUMNS,
    "One row per work looked up in Crossref by DOI: the publisher's abstract as plain text, "
    "the page range and its length, relations, updates and licences.",
)
PUBMED_WORKS_SCHEMA = _staging_schema(
    "pubmed_works",
    PUBMED_WORKS_COLUMNS,
    "One row per work looked up in PubMed: the indexers' publication types and MeSH headings, "
    "PubMed's abstract as plain text, the page range and its length.",
)
