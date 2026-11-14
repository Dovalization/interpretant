"""The record-type signals joined from the staging tables."""

from __future__ import annotations

import polars as pl

from interpretant.catalog import (
    CROSSREF_WORKS_COLUMNS,
    OPENALEX_WORKS_COLUMNS,
    PUBMED_WORKS_COLUMNS,
)
from interpretant.corpus.record_types import RECORD_SIGNAL_COLUMNS, join_record_signals


def staging(columns: pl.Schema, rows: list[dict[str, object]]) -> pl.DataFrame:
    """A staging table with only the given values set, the rest null."""
    return pl.DataFrame([{name: row.get(name) for name in columns} for row in rows], schema=columns)


def test_join_record_signals_keeps_each_source_and_prefers_crossrefs_page_count() -> None:
    works = pl.DataFrame(
        {
            "work_id": [
                "https://openalex.org/W1",
                "https://openalex.org/W2",
                "https://openalex.org/W3",
            ]
        }
    )
    openalex = staging(
        OPENALEX_WORKS_COLUMNS,
        [
            {"work_id": "https://openalex.org/W1", "title": "Errata", "openalex_type": "article"},
            {"work_id": "https://openalex.org/W2", "title": "Letter", "openalex_type": "article"},
            {"work_id": "https://openalex.org/W3", "title": "Lost", "openalex_type": "article"},
        ],
    )  # fmt: skip
    crossref = staging(
        CROSSREF_WORKS_COLUMNS,
        [
            {"work_id": "https://openalex.org/W1", "is_found": True,
             "crossref_type": "journal-article", "update_types": ["correction"], "page_count": 1},
            {"work_id": "https://openalex.org/W2", "is_found": False, "page_count": None},
        ],
    )  # fmt: skip
    pubmed = staging(
        PUBMED_WORKS_COLUMNS,
        [
            {"work_id": "https://openalex.org/W1", "is_found": True,
             "publication_types": ["Published Erratum"], "page_count": 3},
            {"work_id": "https://openalex.org/W2", "is_found": True,
             "publication_types": ["Letter"], "page_count": 2},
        ],
    )  # fmt: skip
    signals = join_record_signals(works, openalex, crossref, pubmed)
    assert signals.schema == RECORD_SIGNAL_COLUMNS
    first, second, third = signals.to_dicts()
    assert (first["update_types"], first["pubmed_types"]) == (["correction"], ["Published Erratum"])
    assert (first["page_count"], first["page_count_source"]) == (1, "crossref")
    assert (second["crossref_type"], second["pubmed_types"]) == (None, ["Letter"])
    assert (second["page_count"], second["page_count_source"]) == (2, "pubmed")
    assert (third["title"], third["page_count"], third["page_count_source"]) == ("Lost", None, None)
