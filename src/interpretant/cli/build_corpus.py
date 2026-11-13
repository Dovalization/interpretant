"""The ``build-corpus`` command: the corpus-level rules and review flags, and what each took."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog import CATALOG_FILE, read_catalog
from interpretant.cli.progress import LiveProgress
from interpretant.cli.results import render_result
from interpretant.pipelines.build_corpus import BuildCorpusSettings, build_corpus
from interpretant.pipelines.params import CorpusSelection


@click.command("build-corpus")
@click.option(
    "--catalog",
    "catalog_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=CATALOG_FILE,
    show_default=True,
    help="The catalog file naming the inputs and outputs.",
)
def build_corpus_command(catalog_file: Path) -> None:
    """Apply the corpus-level rules to every annotated part and write the abstracts table.

    Progress is shown live: a bar in a terminal, a line every half minute in a log file.
    """
    console = Console()
    with LiveProgress(console) as progress:
        result = build_corpus(
            read_catalog(catalog_file), CorpusSelection(), BuildCorpusSettings(), progress=progress
        )
    console.print(render_result(result))
