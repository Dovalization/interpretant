"""The ``report-corpus`` command: the corpus report the author reads before the review."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog.config import CATALOG_FILE, CatalogConfig
from interpretant.cli.jobs import run_on_command_line
from interpretant.pipelines.progress import Progress
from interpretant.pipelines.report_corpus import report_corpus


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
    run_on_command_line(
        "report-corpus",
        catalog_file,
        lambda catalog, progress, console: run_report_corpus(
            catalog, progress=progress, console=console
        ),
    )


def run_report_corpus(
    catalog: CatalogConfig, *, progress: Progress, console: Console
) -> dict[str, int]:
    """Write the corpus report and its metrics; no counts for the run log."""
    written = report_corpus(catalog, progress=progress)
    console.print(
        f"Corpus report written to {written}, its metrics to {catalog.corpus_metrics.path}"
    )
    return {}
