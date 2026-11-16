"""Build the staging tables, one per source, from the raw files (decision agenda 4.8, layers).

A DVC stage: the tables follow from the raw files, which are registered snapshots, so a rebuild
gives the same tables. Polars reads each source's files in one go with the fields its table
needs; each work appears once per table, however many cells downloaded it; the DOI corrections
are applied to OpenAlex's metadata here, so every later step reads corrected values. Before the
Crossref and PubMed tables are built, every work they could be asked about must have a record.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.curated import load_doi_corrections
from interpretant.catalog.linked_records import LinkedSource, load_linked_records_table
from interpretant.catalog.metadata import load_work_metadata_table
from interpretant.catalog.staging import save_staging_table
from interpretant.corpus.crossref import CROSSREF_RECORD_SCHEMA, build_crossref_works
from interpretant.corpus.pubmed import PUBMED_RECORD_SCHEMA, build_pubmed_works
from interpretant.corpus.work_metadata import OPENALEX_METADATA_SCHEMA, build_openalex_works
from interpretant.pipelines.progress import SILENT, Progress


@dataclass(frozen=True)
class LinkedTablesResult:
    """How many rows each staging table holds."""

    openalex: int
    crossref: int
    pubmed: int


def build_linked_tables(
    catalog: CatalogConfig, *, progress: Progress = SILENT
) -> LinkedTablesResult:
    """Build and save the OpenAlex, Crossref and PubMed staging tables.

    Raises:
        ValueError: if a work the source could be asked about has no record from it (the lookup
            did not finish: run ``interpretant fetch-linked-sources``).
    """
    progress.stage("reading the raw files")
    metadata = _one_row_per_work(load_work_metadata_table(catalog, OPENALEX_METADATA_SCHEMA), "id")
    crossref_records = _one_row_per_work(
        load_linked_records_table(catalog, "crossref", CROSSREF_RECORD_SCHEMA), "work_id"
    )
    pubmed_records = _one_row_per_work(
        load_linked_records_table(catalog, "pubmed", PUBMED_RECORD_SCHEMA), "work_id"
    )
    progress.stage("flattening OpenAlex's metadata")
    openalex = build_openalex_works(metadata, load_doi_corrections(catalog))
    found = openalex.filter(~pl.col("is_missing"))
    _check_every_work_has_a_record(
        found.filter(pl.col("doi").is_not_null()), crossref_records, "crossref"
    )
    _check_every_work_has_a_record(
        found.filter(pl.col("pmid").is_not_null()), pubmed_records, "pubmed"
    )
    progress.stage("flattening the Crossref records")
    crossref = build_crossref_works(crossref_records)
    progress.stage("flattening the PubMed records")
    pubmed = build_pubmed_works(pubmed_records)
    progress.stage("saving the staging tables")
    save_staging_table(catalog, "openalex", openalex)
    save_staging_table(catalog, "crossref", crossref)
    save_staging_table(catalog, "pubmed", pubmed)
    return LinkedTablesResult(
        openalex=openalex.height, crossref=crossref.height, pubmed=pubmed.height
    )


def _one_row_per_work(records: pl.DataFrame, key: str) -> pl.DataFrame:
    """The first row of each work in cell order, the rows sorted by work id."""
    return records.unique(subset=key, keep="first", maintain_order=True).sort(key)


def _check_every_work_has_a_record(
    expected: pl.DataFrame, records: pl.DataFrame, source: LinkedSource
) -> None:
    unrecorded = expected.join(records, on="work_id", how="anti").height
    if unrecorded:
        raise ValueError(
            f"{unrecorded} works have no {source} record: run `interpretant fetch-linked-sources`"
        )
