"""The ``detect-languages`` command: lingua on every cached sentence, part by part."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog.config import CATALOG_FILE, CatalogConfig
from interpretant.cli.jobs import run_on_command_line
from interpretant.cli.options import default_process_count, process_count_option
from interpretant.cli.results import render_language_result
from interpretant.pipelines.detect_languages import DetectLanguagesSettings, detect_languages
from interpretant.pipelines.params import CorpusSelection
from interpretant.pipelines.progress import Progress


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

    Each worker process takes whole parts. Progress is shown live: a bar in a terminal, a line
    every half minute in a log file.
    """
    run_on_command_line(
        "detect-languages",
        catalog_file,
        lambda catalog, progress, console: run_detect_languages(
            catalog, process_count, progress=progress, console=console
        ),
    )


def run_detect_languages(
    catalog: CatalogConfig, process_count: int | None, *, progress: Progress, console: Console
) -> dict[str, int]:
    """Detect each sentence's language; the abstracts and sentences for the run log."""
    result = detect_languages(
        catalog,
        CorpusSelection(),
        DetectLanguagesSettings(),
        process_count=process_count or default_process_count(),
        progress=progress,
    )
    console.print(render_language_result(result))
    return {"abstracts": result.abstract_count, "sentences": sum(result.sentence_counts.values())}
