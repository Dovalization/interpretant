"""The ``count-word-repairs`` command: the corpus's word repairs, counted and decided."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog.config import CATALOG_FILE, CatalogConfig
from interpretant.cli.jobs import run_on_command_line
from interpretant.cli.options import default_process_count, process_count_option
from interpretant.cli.results import render_word_repairs_result
from interpretant.pipelines.count_word_repairs import CountWordRepairsSettings, count_word_repairs
from interpretant.pipelines.params import CorpusSelection
from interpretant.pipelines.progress import Progress


@click.command("count-word-repairs")
@click.option(
    "--catalog",
    "catalog_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=CATALOG_FILE,
    show_default=True,
    help="The catalog file naming the inputs and outputs.",
)
@process_count_option
def count_word_repairs_command(catalog_file: Path, process_count: int | None) -> None:
    """Count the word repairs of the repaired texts and decide which the spaCy pass makes.

    The settings come from params.yaml and the kept-joined words from the curated file. Progress
    is shown live: a bar in a terminal, a line every half minute in a log file.
    """
    run_on_command_line(
        "count-word-repairs",
        catalog_file,
        lambda catalog, progress, console: run_count_word_repairs(
            catalog, process_count, progress=progress, console=console
        ),
    )


def run_count_word_repairs(
    catalog: CatalogConfig, process_count: int | None, *, progress: Progress, console: Console
) -> dict[str, int]:
    """Count the word repairs and write the repaired texts; the counts for the run log."""
    result = count_word_repairs(
        catalog,
        CorpusSelection(),
        CountWordRepairsSettings(),
        process_count=process_count or default_process_count(),
        progress=progress,
    )
    console.print(render_word_repairs_result(result))
    return {
        "abstracts": result.abstract_count,
        "glued_words": result.glued_word_count,
        "glued_words_split": sum(result.glued_split_counts.values()),
        "glued_split_occurrences": result.glued_split_occurrences,
        "split_words": result.split_word_count,
        "split_words_joined": result.split_joined_count,
        "split_join_occurrences": result.split_join_occurrences,
    }
