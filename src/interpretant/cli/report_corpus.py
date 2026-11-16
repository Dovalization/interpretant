"""The ``report-corpus`` command: the corpus report the author reads before the review."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog import CATALOG_FILE, read_catalog
from interpretant.cli.progress import LiveProgress
from interpretant.pipelines.progress import StageTimer
from interpretant.pipelines.report_corpus import report_corpus
from interpretant.pipelines.runs import record_run


@click.command("report-corpus")
@click.option(
    "--catalog",
    "catalog_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=CATALOG_FILE,
    show_default=True,
    help="The catalog file naming the inputs and outputs.",
)
def report_corpus_command(catalog_file: Path) -> None:
    """Write the corpus report: what each rule dropped, removed and flagged, by decade and field."""
    console = Console()
    catalog = read_catalog(catalog_file)
    with LiveProgress(console) as live:
        timer = StageTimer(live)
        written = report_corpus(catalog, progress=timer)
    record_run(catalog, command="report-corpus", stages=timer.finish(), counts={})
    console.print(
        f"Corpus report written to {written}, its metrics to {catalog.corpus_metrics.path}"
    )
