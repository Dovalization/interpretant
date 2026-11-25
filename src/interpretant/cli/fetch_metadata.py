"""The ``fetch-metadata`` command: OpenAlex's record of each raw work, cell by cell."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog.config import CATALOG_FILE, CatalogConfig
from interpretant.cli.jobs import run_on_command_line
from interpretant.pipelines.fetch_metadata import (
    OpenAlexSettings,
    count_works,
    fetch_cell_metadata,
    openalex_fetcher,
    select_raw_cells,
)
from interpretant.pipelines.progress import Progress


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
@click.option(
    "--workers",
    type=click.IntRange(1, 20),
    default=8,
    show_default=True,
    help="Requests run at once (OpenAlex allows a free key 30 a second).",
)
def fetch_metadata_command(
    fields: tuple[str, ...], decades: tuple[int, ...], catalog_file: Path, workers: int
) -> None:
    """Fetch OpenAlex's record of every raw work beyond its abstract, resuming where it stopped.

    About one request per 100 works, inside the API key's free daily budget for the regular
    samples. Progress is shown live, and one line is printed per cell as it finishes.
    """
    run_on_command_line(
        "fetch-metadata",
        catalog_file,
        lambda catalog, progress, console: run_fetch_metadata(
            catalog, fields, decades, workers, progress=progress, console=console
        ),
    )


def run_fetch_metadata(
    catalog: CatalogConfig,
    fields: tuple[str, ...],
    decades: tuple[int, ...],
    workers: int,
    *,
    progress: Progress,
    console: Console,
) -> dict[str, int]:
    """Fetch the selected cells' metadata, printing a line per cell; the works for the run log."""
    cells = select_raw_cells(catalog, fields=fields, decades=decades)
    fetch = openalex_fetcher(OpenAlexSettings(), today=date.today())
    progress.stage("counting the raw works")
    total = count_works(catalog, cells)
    progress.stage("fetching metadata", total=total)
    for number, cell in enumerate(cells, start=1):
        result = fetch_cell_metadata(catalog, cell, fetch, workers=workers, progress=progress)
        console.print(
            f"[{number}/{len(cells)}] {cell.field} {cell.decade}: {result.fetched:,} "
            f"fetched, {result.missing:,} missing, {result.already_fetched:,} already there"
        )
    return {"works": total}
