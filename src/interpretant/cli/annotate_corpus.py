"""The ``annotate-corpus`` command: the spaCy pass, part by part, and what each rule took."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog import CATALOG_FILE, read_catalog
from interpretant.cli.results import render_result
from interpretant.pipelines.annotate_corpus import AnnotateCorpusSettings, annotate_corpus
from interpretant.pipelines.params import CorpusSelection


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
@click.option("--process-count", default=1, show_default=True, help="spaCy worker processes.")
def annotate_corpus_command(
    fields: tuple[str, ...],
    decades: tuple[int, ...],
    catalog_file: Path,
    batch_size: int,
    process_count: int,
) -> None:
    """Run the spaCy pass over the corpus in params.yaml, writing the cache and annotated table."""
    result = annotate_corpus(
        read_catalog(catalog_file),
        CorpusSelection(),
        AnnotateCorpusSettings(),
        fields=fields,
        decades=decades,
        batch_size=batch_size,
        process_count=process_count,
    )
    Console().print(render_result(result))
