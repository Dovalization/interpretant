"""The ``score-records`` command: the junk gate's two decision models over every record."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog.config import CATALOG_FILE, CatalogConfig
from interpretant.catalog.record_decisions import default_root
from interpretant.cli.jobs import run_on_command_line
from interpretant.pipelines.judge_ollama import build_http_transport
from interpretant.pipelines.progress import Progress
from interpretant.pipelines.score_records import DecisionSettings, score_records

DEFAULT_HOST = "http://localhost:11434"


@click.command("score-records")
@click.option(
    "--catalog",
    "catalog_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=CATALOG_FILE,
    show_default=True,
    help="The catalog file naming the inputs and outputs.",
)
@click.option(
    "--host",
    default=DEFAULT_HOST,
    show_default=True,
    help="The Ollama server; its address is never written to a file.",
)
@click.option(
    "--limit",
    type=click.IntRange(min=1),
    default=None,
    help="A trial: score only the first records, by both models, with no cut-off.",
)
@click.option(
    "--out",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="A trial's folder in place of the catalog's record_decisions.",
)
def score_records_command(
    catalog_file: Path, host: str, limit: int | None, out: Path | None
) -> None:
    """Score every record behind a drawn usage with the junk gate's two decision models.

    The screening model (params.yaml's junk_gate.decision) scores every record; the confirming
    model scores L's and E's records and those the screening model could call junk
    (docs/junk_gate.md, the deviation of 2026-11-28). A stopped run resumes. Progress is shown
    live: a bar in a terminal, a line every half minute in a log file.
    """
    run_on_command_line(
        "score-records",
        catalog_file,
        lambda catalog, progress, console: run_score_records(
            catalog, host, limit, out, progress=progress, console=console
        ),
    )


def run_score_records(
    catalog: CatalogConfig,
    host: str,
    limit: int | None,
    out: Path | None,
    *,
    progress: Progress,
    console: Console,
) -> dict[str, int]:
    """Score the records; each pass's records needed and scored now, for the run log."""
    settings = DecisionSettings()
    root = out if out is not None else default_root(catalog)
    result = score_records(
        catalog, settings, build_http_transport(host), root=root, limit=limit, progress=progress
    )
    console.print(
        f"{settings.screen_model}: {result.screen.scored:,} of {result.screen.needed:,} records "
        f"scored now; {settings.confirm_model}: {result.confirm.scored:,} of "
        f"{result.confirm.needed:,}"
    )
    if result.lowest_screen_cut_off is not None:
        console.print(f"lowest screening cut-off on L: {result.lowest_screen_cut_off:.4f}")
    return {
        "screen_needed": result.screen.needed,
        "screen_scored": result.screen.scored,
        "confirm_needed": result.confirm.needed,
        "confirm_scored": result.confirm.scored,
    }
