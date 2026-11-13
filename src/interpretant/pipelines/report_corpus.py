"""The corpus report (decision agenda 4b, layer 2), written after the corpus is built.

Reads the abstracts table and each work's primary source from the work metadata, and writes the
report the author reads before the corpus review, with its metrics for DVC. The report opens
with every count that changed since the metrics were last written.
"""

from __future__ import annotations

from pathlib import Path

from interpretant.catalog import (
    CatalogConfig,
    WorkMetadata,
    load_abstracts,
    load_corpus_metrics,
    save_corpus_metrics,
    save_corpus_report,
)
from interpretant.pipelines.corpus import corpus_cells, load_corpus_metadata
from interpretant.pipelines.params import CorpusSelection
from interpretant.pipelines.progress import SILENT, Progress
from interpretant.review.corpus_report import (
    build_corpus_report,
    compare_metrics,
    corpus_metrics,
    render_markdown,
)


def report_corpus(
    catalog: CatalogConfig, selection: CorpusSelection, *, progress: Progress = SILENT
) -> Path:
    """Write the corpus report for the corpus in ``selection`` and return its file.

    Raises:
        FileNotFoundError: if the abstracts table has not been built.
        ValueError: if no raw cell matches the selection, or a cell's metadata is missing.
    """
    cells = corpus_cells(catalog, selection)
    progress.stage("loading the abstracts table")
    abstracts = load_abstracts(catalog).collect()
    progress.stage("reading the works' sources")
    sources = {
        record.work_id: _source_name(record) for record in load_corpus_metadata(catalog, cells)
    }
    progress.stage("writing the report and its metrics")
    previous = load_corpus_metrics(catalog)
    metrics = corpus_metrics(abstracts)
    sections = build_corpus_report(abstracts, sources)
    if previous:
        sections.insert(0, compare_metrics(previous, metrics))
    save_corpus_metrics(catalog, metrics)
    return save_corpus_report(catalog, render_markdown(sections, title="Corpus report"))


def _source_name(record: WorkMetadata) -> str | None:
    location = record.primary_location
    return location.source.display_name if location and location.source else None
