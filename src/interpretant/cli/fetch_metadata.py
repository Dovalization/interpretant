"""The ``fetch-metadata`` command: OpenAlex's record of each raw work, cell by cell."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog import CATALOG_FILE, read_catalog
from interpretant.pipelines.fetch_metadata import (
    OpenAlexSettings,
    fetch_cell_metadata,
    openalex_fetcher,
    select_raw_cells,
)


@click.command("fetch-metadata")
@click.option(
    "--field", "fields", multiple=True, help="Only this field; repeat it. Default: every raw cell."
)
@click.option(
    "--decade",
    "decades",
    multiple=True,
    type=int,
    help="Only this decade; repeat it. Default: every raw cell.",
)
@click.option(
    "--catalog",
    "catalog_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=CATALOG_FILE,
    show_default=True,
    help="The catalog file naming the inputs and outputs.",
)
def fetch_metadata_command(
    fields: tuple[str, ...], decades: tuple[int, ...], catalog_file: Path
) -> None:
    """Fetch OpenAlex's record of every raw work beyond its abstract, resuming where it stopped.

    About one request per 100 works, inside the API key's free daily budget for the regular
    samples. One line is printed per cell as it finishes.
    """
    catalog = read_catalog(catalog_file)
    cells = select_raw_cells(catalog, fields=fields, decades=decades)
    fetch = openalex_fetcher(OpenAlexSettings(), today=date.today())
    console = Console()
    for number, cell in enumerate(cells, start=1):
        result = fetch_cell_metadata(catalog, cell, fetch)
        console.print(
            f"[{number}/{len(cells)}] {cell.field} {cell.decade}: {result.fetched:,} fetched, "
            f"{result.missing:,} missing, {result.already_fetched:,} already there"
        )
