"""The ``compare-corpus`` command: what changed since an earlier build, with samples."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog import CATALOG_FILE, read_catalog
from interpretant.cli.progress import LiveProgress
from interpretant.pipelines.compare_corpus import compare_corpus


@click.command("compare-corpus")
@click.argument("previous", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--catalog",
    "catalog_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=CATALOG_FILE,
    show_default=True,
    help="The catalog file naming the inputs and outputs.",
)
def compare_corpus_command(previous: Path, catalog_file: Path) -> None:
    """Compare the abstracts table with an earlier build kept in the folder PREVIOUS.

    Writes the outcome changes, the removals and sentence languages that changed in works kept in
    both, and samples of each.
    """
    console = Console()
    with LiveProgress(console) as live:
        written = compare_corpus(read_catalog(catalog_file), previous, progress=live)
    console.print(f"Comparison written to {written}")
