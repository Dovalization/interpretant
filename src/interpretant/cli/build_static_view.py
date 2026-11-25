"""The ``build-static-view`` command: every cached document's static tokens, part by part."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from interpretant.catalog.config import CATALOG_FILE, CatalogConfig
from interpretant.cli.jobs import run_on_command_line
from interpretant.cli.options import default_process_count, process_count_option
from interpretant.pipelines.build_static_view import StaticViewResult, build_static_view
from interpretant.pipelines.params import CorpusSelection
from interpretant.pipelines.progress import Progress


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

    The lemma exceptions of the catalog's exceptions file are applied first (agenda 3.6), then the
    spelling map and the compound map (3.4).

    Each worker process takes whole parts. Progress is shown live: a bar in a terminal, a line
    every half minute in a log file.
    """
    run_on_command_line(
        "build-static-view",
        catalog_file,
        lambda catalog, progress, console: run_build_static_view(
            catalog, process_count, progress=progress, console=console
        ),
    )


def run_build_static_view(
    catalog: CatalogConfig, process_count: int | None, *, progress: Progress, console: Console
) -> dict[str, int]:
    """Write the static view; its documents, tokens and maps applied for the run log."""
    result = build_static_view(
        catalog,
        CorpusSelection(),
        process_count=process_count or default_process_count(),
        progress=progress,
    )
    console.print(render_static_view_result(result))
    return {
        "documents": result.document_count,
        "tokens": result.token_count,
        "lemma_exceptions": result.exception_count,
        "spellings": result.spelling_count,
        "compounds": result.compound_count,
    }


def render_static_view_result(result: StaticViewResult) -> Table:
    """One table: the documents and static tokens written."""
    table = Table(title=f"The static view of {len(result.parts)} parts")
    table.add_column("Documents", justify="right")
    table.add_column("Static tokens", justify="right")
    table.add_row(f"{result.document_count:,}", f"{result.token_count:,}")
    return table
