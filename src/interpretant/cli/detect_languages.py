"""The ``detect-languages`` command: lingua on every cached sentence, part by part."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog import CATALOG_FILE, read_catalog
from interpretant.cli.options import default_process_count, process_count_option
from interpretant.cli.progress import LiveProgress
from interpretant.cli.results import render_language_result
from interpretant.pipelines.detect_languages import DetectLanguagesSettings, detect_languages
from interpretant.pipelines.params import CorpusSelection
from interpretant.pipelines.progress import StageTimer
from interpretant.pipelines.runs import record_run


@click.command("detect-languages")
@click.option(
    "--catalog",
    "catalog_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=CATALOG_FILE,
    show_default=True,
    help="The catalog file naming the inputs and outputs.",
)
@process_count_option
def detect_languages_command(catalog_file: Path, process_count: int | None) -> None:
    """Detect each cached sentence's language with the settings in params.yaml.

    Progress is shown live: a bar in a terminal, a line every half minute in a log file.
    """
    console = Console()
    catalog = read_catalog(catalog_file)
    with LiveProgress(console) as live:
        timer = StageTimer(live)
        result = detect_languages(
            catalog,
            CorpusSelection(),
            DetectLanguagesSettings(),
            process_count=process_count or default_process_count(),
            progress=timer,
        )
    counts = {"abstracts": result.abstract_count, "sentences": sum(result.sentence_counts.values())}
    record_run(catalog, command="detect-languages", stages=timer.finish(), counts=counts)
    console.print(render_language_result(result))
