"""The ``fetch-linked-sources`` command: each raw work's record in Crossref and PubMed."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog import CATALOG_FILE, LINKED_SOURCES, LinkedSource, read_catalog
from interpretant.cli.progress import LiveProgress
from interpretant.pipelines.fetch_linked_sources import (
    LinkedSourcesSettings,
    build_source_plan,
    count_linked_works,
    fetch_cell_linked_records,
)
from interpretant.pipelines.fetch_metadata import select_raw_cells
from interpretant.pipelines.progress import StageTimer
from interpretant.pipelines.runs import record_run


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
    catalog = read_catalog(catalog_file)
    cells = select_raw_cells(catalog, fields=fields, decades=decades)
    settings = LinkedSourcesSettings()
    console = Console()
    counts: dict[str, int] = {}
    with LiveProgress(console) as live:
        timer = StageTimer(live)
        for source in sources or LINKED_SOURCES:
            plan = build_source_plan(source, settings, today=date.today())
            timer.stage(f"counting the works {source} can be asked about")
            total = count_linked_works(catalog, cells, plan)
            counts[f"{source}_works"] = total
            timer.stage(f"looking the works up in {source}", total=total)
            for number, cell in enumerate(cells, start=1):
                result = fetch_cell_linked_records(catalog, cell, source, plan, progress=timer)
                console.print(
                    f"{source} [{number}/{len(cells)}] {cell.field} {cell.decade}: "
                    f"{result.found:,} found, {result.missing:,} missing, "
                    f"{result.already_fetched:,} already there"
                )
    record_run(catalog, command="fetch-linked-sources", stages=timer.finish(), counts=counts)
