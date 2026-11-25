"""The ``build-linked-tables`` command: the staging table of each source, from the raw files."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog.config import CATALOG_FILE, CatalogConfig
from interpretant.cli.jobs import run_on_command_line
from interpretant.pipelines.build_linked_tables import build_linked_tables
from interpretant.pipelines.progress import Progress


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
    run_on_command_line(
        "build-linked-tables",
        catalog_file,
        lambda catalog, progress, console: run_build_linked_tables(
            catalog, progress=progress, console=console
        ),
    )


def run_build_linked_tables(
    catalog: CatalogConfig, *, progress: Progress, console: Console
) -> dict[str, int]:
    """Write the three staging tables; their works for the run log."""
    result = build_linked_tables(catalog, progress=progress)
    console.print(
        f"Staging tables written: {result.openalex:,} OpenAlex, {result.crossref:,} Crossref and "
        f"{result.pubmed:,} PubMed works"
    )
    return {
        "openalex_works": result.openalex,
        "crossref_works": result.crossref,
        "pubmed_works": result.pubmed,
    }
