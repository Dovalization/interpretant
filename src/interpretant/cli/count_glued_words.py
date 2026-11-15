"""The ``count-glued-words`` command: the corpus's glued words, counted and decided."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog import CATALOG_FILE, read_catalog
from interpretant.cli.options import default_process_count, process_count_option
from interpretant.cli.progress import LiveProgress
from interpretant.cli.results import render_glued_words_result
from interpretant.pipelines.count_glued_words import CountGluedWordsSettings, count_glued_words
from interpretant.pipelines.params import CorpusSelection
from interpretant.pipelines.progress import StageTimer
from interpretant.pipelines.runs import record_run


@click.command("count-glued-words")
@click.option(
    "--catalog",
    "catalog_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=CATALOG_FILE,
    show_default=True,
    help="The catalog file naming the inputs and outputs.",
)
@process_count_option
def count_glued_words_command(catalog_file: Path, process_count: int | None) -> None:
    """Count the glued words of the repaired texts and decide which the spaCy pass splits.

    The settings come from params.yaml and the kept-joined words from the curated file. Progress
    is shown live: a bar in a terminal, a line every half minute in a log file.
    """
    console = Console()
    catalog = read_catalog(catalog_file)
    with LiveProgress(console) as live:
        timer = StageTimer(live)
        result = count_glued_words(
            catalog,
            CorpusSelection(),
            CountGluedWordsSettings(),
            process_count=process_count or default_process_count(),
            progress=timer,
        )
    counts = {
        "abstracts": result.abstract_count,
        "glued_words": result.word_count,
        "split_words": sum(result.split_counts.values()),
        "split_occurrences": result.split_occurrences,
    }
    record_run(catalog, command="count-glued-words", stages=timer.finish(), counts=counts)
    console.print(render_glued_words_result(result))
