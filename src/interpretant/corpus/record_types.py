"""The signals of what kind of record each work is, joined from the staging tables (agenda 2.3).

The record-type batch decides which kept works are not the work's own writing (letters, news,
errata, book reviews, catalogue entries) from several sources that disagree, so each source's
value is kept beside the others: OpenAlex's type and title, Crossref's type and whether the
record is itself an update (a correction or retraction notice), PubMed's publication types, and
the page count, Crossref's where it has one, PubMed's otherwise. The rules that read them come
with the batch.
"""

from __future__ import annotations

import polars as pl

RECORD_SIGNAL_COLUMNS = pl.Schema(
    {
        "work_id": pl.String(),
        "title": pl.String(),
        "openalex_type": pl.String(),
        "is_paratext": pl.Boolean(),
        "crossref_type": pl.String(),
        "update_types": pl.List(pl.String()),
        "pubmed_types": pl.List(pl.String()),
        "page_count": pl.Int64(),
        "page_count_source": pl.String(),
    }
)


def join_record_signals(
    works: pl.DataFrame, openalex: pl.DataFrame, crossref: pl.DataFrame, pubmed: pl.DataFrame
) -> pl.DataFrame:
    """Each work's record-type signals, one row per work in ``works`` and in its order.

    ``works`` needs a ``work_id`` column; the other three are the staging tables. A source that
    holds no record for a work, or did not find it, leaves its columns null.
    """
    found_crossref = crossref.filter("is_found").select(
        "work_id", "crossref_type", "update_types", pl.col("page_count").alias("crossref_pages")
    )
    found_pubmed = pubmed.filter("is_found").select(
        "work_id",
        pl.col("publication_types").alias("pubmed_types"),
        pl.col("page_count").alias("pubmed_pages"),
    )
    joined = (
        works.select("work_id")
        .join(
            openalex.select("work_id", "title", "openalex_type", "is_paratext"),
            on="work_id",
            how="left",
        )
        .join(found_crossref, on="work_id", how="left")
        .join(found_pubmed, on="work_id", how="left")
    )
    return joined.select(
        "work_id",
        "title",
        "openalex_type",
        "is_paratext",
        "crossref_type",
        "update_types",
        "pubmed_types",
        pl.coalesce("crossref_pages", "pubmed_pages").alias("page_count"),
        pl.when(pl.col("crossref_pages").is_not_null())
        .then(pl.lit("crossref"))
        .when(pl.col("pubmed_pages").is_not_null())
        .then(pl.lit("pubmed"))
        .alias("page_count_source"),
    )
