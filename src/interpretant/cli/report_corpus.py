"""The ``report-corpus`` command: the corpus report the author reads before the review."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog import CATALOG_FILE, read_catalog
from interpretant.cli.progress import LiveProgress
from interpretant.pipelines.params import CorpusSelection
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
    console = Console()
    with LiveProgress(console) as progress:
        written = report_corpus(read_catalog(catalog_file), CorpusSelection(), progress=progress)
    console.print(f"Corpus report written to {written}")
