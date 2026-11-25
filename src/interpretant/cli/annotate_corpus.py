"""The ``annotate-corpus`` command: the spaCy pass, part by part, and what each rule took."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog.config import CATALOG_FILE, CatalogConfig
from interpretant.cli.jobs import run_on_command_line
from interpretant.cli.options import default_process_count, process_count_option
from interpretant.cli.results import render_result, result_counts
from interpretant.pipelines.annotate_corpus import AnnotateCorpusSettings, annotate_corpus
from interpretant.pipelines.params import CorpusSelection
from interpretant.pipelines.progress import Progress


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
@process_count_option
def annotate_corpus_command(
    fields: tuple[str, ...],
    decades: tuple[int, ...],
    catalog_file: Path,
    batch_size: int,
    process_count: int | None,
) -> None:
    """Run the spaCy pass over the corpus in params.yaml, writing the annotation cache.

    Progress is shown live: a bar in a terminal, a line every half minute in a log file.
    """
    run_on_command_line(
        "annotate-corpus",
        catalog_file,
        lambda catalog, progress, console: run_annotate_corpus(
            catalog, fields, decades, batch_size, process_count, progress=progress, console=console
        ),
    )


def run_annotate_corpus(
    catalog: CatalogConfig,
    fields: tuple[str, ...],
    decades: tuple[int, ...],
    batch_size: int,
    process_count: int | None,
    *,
    progress: Progress,
    console: Console,
) -> dict[str, int]:
    """Annotate the selected cells; their headline counts for the run log."""
    result = annotate_corpus(
        catalog,
        CorpusSelection(),
        AnnotateCorpusSettings(),
        fields=fields,
        decades=decades,
        batch_size=batch_size,
        process_count=process_count or default_process_count(),
        progress=progress,
    )
    console.print(render_result(result))
    return result_counts(result)
