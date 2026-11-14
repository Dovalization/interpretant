"""The ``build-linked-tables`` command: the staging table of each source, from the raw files."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog import CATALOG_FILE, read_catalog
from interpretant.cli.progress import LiveProgress
from interpretant.pipelines.build_linked_tables import build_linked_tables
from interpretant.pipelines.progress import StageTimer
from interpretant.pipelines.runs import record_run


@click.command("build-linked-tables")
@click.option(
    "--catalog",
    "catalog_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=CATALOG_FILE,
    show_default=True,
    help="The catalog file naming the inputs and outputs.",
)
def build_linked_tables_command(catalog_file: Path) -> None:
    """Build one checked table per source (OpenAlex, Crossref, PubMed), keyed by work id."""
    console = Console()
    catalog = read_catalog(catalog_file)
    with LiveProgress(console) as live:
        timer = StageTimer(live)
        result = build_linked_tables(catalog, progress=timer)
    counts = {"openalex_works": result.openalex, "crossref_works": result.crossref}
    counts["pubmed_works"] = result.pubmed
    record_run(catalog, command="build-linked-tables", stages=timer.finish(), counts=counts)
    console.print(
        f"Staging tables written: {result.openalex:,} OpenAlex, {result.crossref:,} Crossref and "
        f"{result.pubmed:,} PubMed works"
    )
