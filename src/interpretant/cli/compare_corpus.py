"""The ``compare-corpus`` command: what changed since an earlier build, with samples."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog import CATALOG_FILE, read_catalog
from interpretant.cli.progress import LiveProgress
from interpretant.pipelines.compare_corpus import compare_corpus


@click.command("compare-corpus")
@click.option(
    "--revision",
    default="HEAD",
    show_default=True,
    help="The git revision whose recorded build to compare with (HEAD: the last recorded run).",
)
@click.option(
    "--catalog",
    "catalog_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=CATALOG_FILE,
    show_default=True,
    help="The catalog file naming the inputs and outputs.",
)
def compare_corpus_command(revision: str, catalog_file: Path) -> None:
    """Compare this build with the one recorded at a git revision, read from the DVC cache.

    Writes the outcome changes; the removals, sentence languages and final text that changed in
    works kept in both; the word-repair decisions that changed; and samples of each.
    """
    console = Console()
    with LiveProgress(console) as live:
        written = compare_corpus(
            read_catalog(catalog_file), catalog_file.resolve().parent, revision, progress=live
        )
    console.print(f"Comparison written to {written}")
