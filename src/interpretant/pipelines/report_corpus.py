"""The corpus report (decision agenda 4b, layer 2), written after the corpus is built.

Reads the abstracts table and each work's primary source from OpenAlex's staging table, and
writes the report the author reads before the corpus review, with its metrics for DVC. The
report opens with every count that changed since the metrics were last written.
"""

from __future__ import annotations

from pathlib import Path

from interpretant.catalog.abstracts import load_abstracts
from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.reports import (
    load_corpus_metrics,
    save_corpus_metrics,
    save_corpus_report,
)
from interpretant.catalog.staging import load_staging_table
from interpretant.pipelines.progress import SILENT, Progress
from interpretant.review.corpus_report import (
    build_corpus_report,
    compare_metrics,
    corpus_metrics,
    render_markdown,
)


def report_corpus(catalog: CatalogConfig, *, progress: Progress = SILENT) -> Path:
    """Write the corpus report for the built abstracts table and return its file.

    Raises:
        FileNotFoundError: if the abstracts table or the staging tables have not been built.
    """
    progress.stage("loading the abstracts table")
    abstracts = load_abstracts(catalog).collect()
    progress.stage("reading the works' sources")
    sources = load_staging_table(catalog, "openalex").select("work_id", "source_name").collect()
    progress.stage("writing the report and its metrics")
    previous = load_corpus_metrics(catalog)
    metrics = corpus_metrics(abstracts)
    sections = build_corpus_report(abstracts, sources)
    if previous:
        sections.insert(0, compare_metrics(previous, metrics))
    save_corpus_metrics(catalog, metrics)
    return save_corpus_report(catalog, render_markdown(sections, title="Corpus report"))
