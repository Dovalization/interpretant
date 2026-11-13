"""The ``annotate-corpus`` command: the spaCy pass, part by part, and what each rule took."""

from __future__ import annotations

import os
from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog import CATALOG_FILE, read_catalog
from interpretant.cli.progress import LiveProgress
from interpretant.cli.results import render_result, result_counts
from interpretant.pipelines.annotate_corpus import AnnotateCorpusSettings, annotate_corpus
from interpretant.pipelines.params import CorpusSelection
from interpretant.pipelines.progress import StageTimer
from interpretant.pipelines.runs import record_run


@click.command("annotate-corpus")
@click.option(
    "--field",
    "fields",
    multiple=True,
    help="Annotate only this field's parts; repeat it. Default: every part of the corpus.",
)
@click.option(
    "--decade",
    "decades",
    multiple=True,
    type=int,
    help="Annotate only this decade's parts; repeat it. Default: every part of the corpus.",
)
@click.option(
    "--catalog",
    "catalog_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=CATALOG_FILE,
    show_default=True,
    help="The catalog file naming the inputs and outputs.",
)
@click.option("--batch-size", default=256, show_default=True, help="Abstracts per spaCy batch.")
@click.option(
    "--process-count",
    type=click.IntRange(min=1),
    default=None,
    help="spaCy worker processes. Default: the machine's CPU count minus 2.",
)
def annotate_corpus_command(
    fields: tuple[str, ...],
    decades: tuple[int, ...],
    catalog_file: Path,
    batch_size: int,
    process_count: int | None,
) -> None:
    """Run the spaCy pass over the corpus in params.yaml, writing the cache and annotated table.

    Progress is shown live: a bar in a terminal, a line every half minute in a log file.
    """
    console = Console()
    catalog = read_catalog(catalog_file)
    with LiveProgress(console) as live:
        timer = StageTimer(live)
        result = annotate_corpus(
            catalog,
            CorpusSelection(),
            AnnotateCorpusSettings(),
            fields=fields,
            decades=decades,
            batch_size=batch_size,
            process_count=process_count or default_process_count(),
            progress=timer,
        )
    record_run(
        catalog, command="annotate-corpus", stages=timer.finish(), counts=result_counts(result)
    )
    console.print(render_result(result))


def default_process_count() -> int:
    """The machine's CPU count minus two, leaving room for the main process and the system."""
    return max(1, (os.cpu_count() or 2) - 2)
