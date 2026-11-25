"""The ``build-corpus`` command: the corpus-level rules and review flags, and what each took."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog.config import CATALOG_FILE, CatalogConfig
from interpretant.cli.jobs import run_on_command_line
from interpretant.cli.options import default_process_count, process_count_option
from interpretant.cli.results import render_result, result_counts
from interpretant.pipelines.build_corpus import BuildCorpusSettings, build_corpus
from interpretant.pipelines.params import CorpusSelection
from interpretant.pipelines.progress import Progress


@click.command("build-corpus")
@click.option(
    "--catalog",
    "catalog_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=CATALOG_FILE,
    show_default=True,
    help="The catalog file naming the inputs and outputs.",
)
@process_count_option
def build_corpus_command(catalog_file: Path, process_count: int | None) -> None:
    """Apply the corpus-level rules to every annotated part and write the abstracts table.

    Progress is shown live: a bar in a terminal, a line every half minute in a log file.
    """
    run_on_command_line(
        "build-corpus",
        catalog_file,
        lambda catalog, progress, console: run_build_corpus(
            catalog, process_count, progress=progress, console=console
        ),
    )


def run_build_corpus(
    catalog: CatalogConfig, process_count: int | None, *, progress: Progress, console: Console
) -> dict[str, int]:
    """Build the corpus from the annotated parts; its headline counts for the run log."""
    result = build_corpus(
        catalog,
        CorpusSelection(),
        BuildCorpusSettings(),
        process_count=process_count or default_process_count(),
        progress=progress,
    )
    console.print(render_result(result))
    return result_counts(result)
