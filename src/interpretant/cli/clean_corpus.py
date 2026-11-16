"""The ``clean-corpus`` command: the cleaning patterns and keeping rules, and what each took."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog.config import CATALOG_FILE, read_catalog
from interpretant.cli.options import default_process_count, process_count_option
from interpretant.cli.progress import LiveProgress
from interpretant.cli.results import render_result, result_counts
from interpretant.pipelines.clean_corpus import CleanCorpusSettings, clean_corpus
from interpretant.pipelines.params import CorpusSelection
from interpretant.pipelines.progress import StageTimer
from interpretant.pipelines.runs import record_run


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
    console = Console()
    catalog = read_catalog(catalog_file)
    with LiveProgress(console) as live:
        timer = StageTimer(live)
        result = clean_corpus(
            catalog,
            CorpusSelection(),
            CleanCorpusSettings(),
            process_count=process_count or default_process_count(),
            progress=timer,
        )
    record_run(catalog, command="clean-corpus", stages=timer.finish(), counts=result_counts(result))
    console.print(render_result(result))
