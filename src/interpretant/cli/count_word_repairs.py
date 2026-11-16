"""The ``count-word-repairs`` command: the corpus's word repairs, counted and decided."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog import CATALOG_FILE, read_catalog
from interpretant.cli.options import default_process_count, process_count_option
from interpretant.cli.progress import LiveProgress
from interpretant.cli.results import render_word_repairs_result
from interpretant.pipelines.count_word_repairs import CountWordRepairsSettings, count_word_repairs
from interpretant.pipelines.params import CorpusSelection
from interpretant.pipelines.progress import StageTimer
from interpretant.pipelines.runs import record_run


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
    console = Console()
    catalog = read_catalog(catalog_file)
    with LiveProgress(console) as live:
        timer = StageTimer(live)
        result = count_word_repairs(
            catalog,
            CorpusSelection(),
            CountWordRepairsSettings(),
            process_count=process_count or default_process_count(),
            progress=timer,
        )
    counts = {
        "abstracts": result.abstract_count,
        "glued_words": result.glued_word_count,
        "glued_words_split": sum(result.glued_split_counts.values()),
        "glued_split_occurrences": result.glued_split_occurrences,
    }
    record_run(catalog, command="count-word-repairs", stages=timer.finish(), counts=counts)
    console.print(render_word_repairs_result(result))
