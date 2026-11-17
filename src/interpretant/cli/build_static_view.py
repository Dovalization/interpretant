"""The ``build-static-view`` command: every cached document's static tokens, part by part."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from interpretant.catalog.config import CATALOG_FILE, read_catalog
from interpretant.cli.options import default_process_count, process_count_option
from interpretant.cli.progress import LiveProgress
from interpretant.pipelines.build_static_view import StaticViewResult, build_static_view
from interpretant.pipelines.params import CorpusSelection
from interpretant.pipelines.progress import StageTimer
from interpretant.pipelines.runs import record_run


@click.command("build-static-view")
@click.option(
    "--catalog",
    "catalog_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=CATALOG_FILE,
    show_default=True,
    help="The catalog file naming the inputs and outputs.",
)
@process_count_option
def build_static_view_command(catalog_file: Path, process_count: int | None) -> None:
    """Write the static tokens of every cached document, with offsets and finite-verb flags.

    Each worker process takes whole parts. Progress is shown live: a bar in a terminal, a line
    every half minute in a log file.
    """
    console = Console()
    catalog = read_catalog(catalog_file)
    with LiveProgress(console) as live:
        timer = StageTimer(live)
        result = build_static_view(
            catalog,
            CorpusSelection(),
            process_count=process_count or default_process_count(),
            progress=timer,
        )
    counts = {"documents": result.document_count, "tokens": result.token_count}
    record_run(catalog, command="build-static-view", stages=timer.finish(), counts=counts)
    console.print(render_static_view_result(result))


def render_static_view_result(result: StaticViewResult) -> Table:
    """One table: the documents and static tokens written."""
    table = Table(title=f"The static view of {len(result.parts)} parts")
    table.add_column("Documents", justify="right")
    table.add_column("Static tokens", justify="right")
    table.add_row(f"{result.document_count:,}", f"{result.token_count:,}")
    return table
