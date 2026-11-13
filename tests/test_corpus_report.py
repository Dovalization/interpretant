"""The corpus report: its sections from a small abstracts table, its Markdown, and its stage."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from interpretant.catalog import read_catalog
from interpretant.corpus.work_metadata import REPOSITORY_ONLY
from interpretant.pipelines.annotate_corpus import annotate_corpus
from interpretant.pipelines.build_corpus import BuildCorpusSettings, build_corpus
from interpretant.pipelines.report_corpus import report_corpus
from interpretant.review.corpus_report import build_corpus_report, corpus_metrics, render_markdown
from tests.abstracts_tables import abstract_row, abstracts_table
from tests.corpus_catalog import ANNOTATE_SETTINGS, SELECTION

TEMPLATE = {"start": 0, "end": 8, "reason": "template_sentence"}


def report_table() -> pl.DataFrame:
    """Five works: two kept, one dropped as too short, one near-duplicate across fields."""
    return abstracts_table(
        [
            abstract_row(1, ["RESULTS:", "Models learn senses."], removals=[TEMPLATE]),
            abstract_row(2, ["Models learn senses again."], flags=[REPOSITORY_ONLY]),
            abstract_row(3, ["Short."], dropped_reason="too_short", year=1962),
            abstract_row(
                4,
                ["Models learn senses again and again."],
                fields=["sociology"],
                dropped_reason="near_duplicate",
                duplicate_of="https://openalex.org/W1",
            ),
            abstract_row(
                5, ["A photograph of the team."], flags=[REPOSITORY_ONLY], dropped_reason="template"
            ),
        ]
    )


def section_tables(table: pl.DataFrame) -> dict[str, pl.DataFrame]:
    sources = {"https://openalex.org/W2": "Papers", "https://openalex.org/W5": "Archive"}
    return {section.title: section.table for section in build_corpus_report(table, sources)}


def test_build_corpus_report_counts_what_each_rule_did() -> None:
    sections = section_tables(report_table())
    assert sections["Totals"]["count"].to_list() == [5, 2, 3]
    by_decade = sections["Dropped abstracts by decade"]
    assert by_decade.filter(pl.col("reason") == "too_short")["1960"].to_list() == [1]
    assert sections["Template sentences"].row(0) == ("RESULTS:", 1)
    removals = sections["Removed ranges"]
    assert removals.row(0) == ("template_sentence", 1, 1)


def test_build_corpus_report_splits_repository_works_by_outcome_and_repository() -> None:
    sections = section_tables(report_table())
    outcomes = dict(sections["Repository-only works by outcome"].select("outcome", "works").rows())
    assert outcomes == {"kept": 1, "template": 1}
    repositories = sections["Repository-only works by repository"]
    assert repositories.select("repository", "kept", "dropped").rows() == [
        ("Papers", 1, 0),
        ("Archive", 0, 1),
    ]


def test_build_corpus_report_places_near_duplicates_across_fields() -> None:
    near = section_tables(report_table())["Near-duplicates across fields and decades"]
    assert near.rows() == [("no field in common", "same decade", 1)]


def test_render_markdown_writes_unwrapped_tables() -> None:
    text = render_markdown(build_corpus_report(report_table(), {}), title="Corpus report")
    assert text.startswith("# Corpus report\n")
    assert "## Totals" in text
    rows = [line for line in text.splitlines() if line.startswith("| reason ")]
    assert rows
    table_lines = [line for line in text.splitlines() if line.startswith("|")]
    assert all(line.endswith("|") for line in table_lines)


def test_report_corpus_writes_the_report_after_the_build(corpus_catalog_file: Path) -> None:
    catalog = read_catalog(corpus_catalog_file)
    annotate_corpus(catalog, SELECTION, ANNOTATE_SETTINGS)
    build_corpus(catalog, SELECTION, BuildCorpusSettings())
    written = report_corpus(catalog, SELECTION)
    assert written == catalog.corpus_report.path
    assert "## Repository-only works by repository" in written.read_text()
    metrics = json.loads(catalog.corpus_metrics.path.read_text())
    assert metrics["abstracts"] == 4  # works 1 to 4, the 2010s cells


def test_corpus_metrics_counts_every_drop_removal_and_flag() -> None:
    assert corpus_metrics(report_table()) == {
        "abstracts": 5,
        "kept": 2,
        "dropped": 3,
        "dropped_by_reason": {"near_duplicate": 1, "template": 1, "too_short": 1},
        "removed_ranges_by_reason": {"template_sentence": 1},
        "flagged_by_flag": {REPOSITORY_ONLY: 2},
    }
