"""The ``fetch-linked-sources`` command: each raw work's record in Crossref and PubMed."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog.config import CATALOG_FILE, CatalogConfig
from interpretant.catalog.curated import load_doi_corrections
from interpretant.catalog.linked_records import LINKED_SOURCES, LinkedSource
from interpretant.cli.jobs import run_on_command_line
from interpretant.pipelines.fetch_linked_sources import (
    LinkedSourcesSettings,
    build_source_plan,
    count_linked_works,
    fetch_cell_linked_records,
)
from interpretant.pipelines.fetch_metadata import select_raw_cells
from interpretant.pipelines.progress import Progress


@click.command("fetch-linked-sources")
@click.option(
    "--source",
    "sources",
    type=click.Choice(LINKED_SOURCES),
    multiple=True,
    help="Only this source; repeat it. Default: Crossref, then PubMed.",
)
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
def fetch_linked_sources_command(
    sources: tuple[LinkedSource, ...],
    fields: tuple[str, ...],
    decades: tuple[int, ...],
    catalog_file: Path,
) -> None:
    """Look up each raw work in Crossref (by DOI) and PubMed (by PubMed id), resuming if stopped.

    Free. A contact address in CONTACT_EMAIL (environment or .env) moves Crossref's requests to
    its faster polite pool. Progress is shown live, and one line is printed per cell.
    """
    run_on_command_line(
        "fetch-linked-sources",
        catalog_file,
        lambda catalog, progress, console: run_fetch_linked_sources(
            catalog, sources, fields, decades, progress=progress, console=console
        ),
    )


def run_fetch_linked_sources(
    catalog: CatalogConfig,
    sources: tuple[LinkedSource, ...],
    fields: tuple[str, ...],
    decades: tuple[int, ...],
    *,
    progress: Progress,
    console: Console,
) -> dict[str, int]:
    """Look the selected cells' works up in each source; the works asked about for the run log."""
    cells = select_raw_cells(catalog, fields=fields, decades=decades)
    corrections = load_doi_corrections(catalog)
    settings = LinkedSourcesSettings()
    counts: dict[str, int] = {}
    for source in sources or LINKED_SOURCES:
        plan = build_source_plan(source, settings, today=date.today())
        progress.stage(f"counting the works {source} can be asked about")
        total = count_linked_works(catalog, cells, plan, corrections)
        counts[f"{source}_works"] = total
        progress.stage(f"looking the works up in {source}", total=total)
        for number, cell in enumerate(cells, start=1):
            result = fetch_cell_linked_records(
                catalog, cell, source, plan, corrections, progress=progress
            )
            console.print(
                f"{source} [{number}/{len(cells)}] {cell.field} {cell.decade}: "
                f"{result.found:,} found, {result.missing:,} missing, "
                f"{result.already_fetched:,} already there"
            )
    return counts
