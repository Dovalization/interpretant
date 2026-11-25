"""The ``clean-corpus`` command: the cleaning patterns and keeping rules, and what each took."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog.config import CATALOG_FILE, CatalogConfig
from interpretant.cli.jobs import run_on_command_line
from interpretant.cli.options import default_process_count, process_count_option
from interpretant.cli.results import render_result, result_counts
from interpretant.pipelines.clean_corpus import CleanCorpusSettings, clean_corpus
from interpretant.pipelines.params import CorpusSelection
from interpretant.pipelines.progress import Progress


@click.command("clean-corpus")
@click.option(
    "--catalog",
    "catalog_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=CATALOG_FILE,
    show_default=True,
    help="The catalog file naming the inputs and outputs.",
)
@process_count_option
def clean_corpus_command(catalog_file: Path, process_count: int | None) -> None:
    """Apply the cleaning patterns and keeping rules to the cache, writing the annotated table.

    Progress is shown live: a bar in a terminal, a line every half minute in a log file.
    """
    run_on_command_line(
        "clean-corpus",
        catalog_file,
        lambda catalog, progress, console: run_clean_corpus(
            catalog, process_count, progress=progress, console=console
        ),
    )


def run_clean_corpus(
    catalog: CatalogConfig, process_count: int | None, *, progress: Progress, console: Console
) -> dict[str, int]:
    """Clean the annotated abstracts; their headline counts for the run log."""
    result = clean_corpus(
        catalog,
        CorpusSelection(),
        CleanCorpusSettings(),
        process_count=process_count or default_process_count(),
        progress=progress,
    )
    console.print(render_result(result))
    return result_counts(result)
