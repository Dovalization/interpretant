"""The ``detect-base-languages`` command: lingua on every sentence of the base texts."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog import CATALOG_FILE, read_catalog
from interpretant.cli.options import default_process_count, process_count_option
from interpretant.cli.progress import LiveProgress
from interpretant.cli.results import render_language_result
from interpretant.pipelines.detect_base_languages import detect_base_languages
from interpretant.pipelines.detect_languages import DetectLanguagesSettings
from interpretant.pipelines.params import CorpusSelection
from interpretant.pipelines.progress import StageTimer
from interpretant.pipelines.runs import record_run


@click.command("detect-base-languages")
@click.option(
    "--catalog",
    "catalog_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=CATALOG_FILE,
    show_default=True,
    help="The catalog file naming the inputs and outputs.",
)
@process_count_option
def detect_base_languages_command(catalog_file: Path, process_count: int | None) -> None:
    """Detect the language of each sentence of the base texts, before the word repairs.

    Each worker process takes whole parts. Progress is shown live: a bar in a terminal, a line
    every half minute in a log file.
    """
    console = Console()
    catalog = read_catalog(catalog_file)
    with LiveProgress(console) as live:
        timer = StageTimer(live)
        result = detect_base_languages(
            catalog,
            CorpusSelection(),
            DetectLanguagesSettings(),
            process_count=process_count or default_process_count(),
            progress=timer,
        )
    counts = {"abstracts": result.abstract_count, "sentences": sum(result.sentence_counts.values())}
    record_run(catalog, command="detect-base-languages", stages=timer.finish(), counts=counts)
    console.print(render_language_result(result))
