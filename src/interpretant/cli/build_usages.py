"""The ``build-usages`` command: every target mention in a kept work, and the one drawn per work."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from interpretant.catalog.config import CATALOG_FILE, read_catalog
from interpretant.cli.options import default_process_count, process_count_option
from interpretant.cli.progress import LiveProgress
from interpretant.pipelines.build_usages import UsagesResult, UsagesSettings, build_usages
from interpretant.pipelines.params import CorpusSelection
from interpretant.pipelines.progress import StageTimer
from interpretant.pipelines.runs import record_run

COUNT_LABELS = {
    "mentions": "Mentions",
    "staged_sentences": "Target sentences annotated",
    "skipped_parts": "Parts kept from an earlier run",
    "cache_hits": "Sentences read from the annotation cache",
    "cache_misses": "Sentences annotated by the transformer",
    "inside_removed_ranges": "In removed ranges",
    "outside_sentences": "Outside sentences",
    "in_removed_sentences": "In removed sentences",
    "usages": "Usages",
    "in_scope": "In scope",
    "drawn": "Drawn",
}


@click.command("build-usages")
@click.option(
    "--catalog",
    "catalog_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=CATALOG_FILE,
    show_default=True,
    help="The catalog file naming the inputs and outputs.",
)
@click.option(
    "--field", "fields", multiple=True, help="Only this field (repeatable). Default: params.yaml."
)
@click.option(
    "--decade",
    "decades",
    multiple=True,
    type=int,
    help="Only this decade (repeatable). Default: params.yaml.",
)
@click.option(
    "--gpu-process-count",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="Processes sharing the GPU when the transformer annotates the usages.",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Keep the parts the transformer already wrote with the current columns, and skip them.",
)
@process_count_option
def build_usages_command(
    catalog_file: Path,
    fields: tuple[str, ...],
    decades: tuple[int, ...],
    gpu_process_count: int,
    process_count: int | None,
    *,
    resume: bool,
) -> None:
    """Write one row per target mention in a kept work, and draw one usage per work and word.

    A mention inside a removed range is left out. Each usage records its sentence, its
    annotation, whether it sits inside a name, whether it is in scope, and whether it is the one
    drawn: the eligible usage with the smallest draw key (decision agenda 4.1-4.6).

    Each worker process takes whole parts. Progress is shown live: a bar in a terminal, a line
    every half minute in a log file.
    """
    console = Console()
    catalog = read_catalog(catalog_file)
    overrides: dict[str, Any] = {}  # only the options given override params.yaml
    if fields:
        overrides["fields"] = fields
    if decades:
        overrides["decades"] = decades
    with LiveProgress(console) as live:
        timer = StageTimer(live)
        result = build_usages(
            catalog,
            CorpusSelection(**overrides),
            UsagesSettings(),
            process_count=process_count or default_process_count(),
            gpu_process_count=gpu_process_count,
            resume=resume,
            progress=timer,
        )
    record_run(catalog, command="build-usages", stages=timer.finish(), counts=result.counts)
    console.print(render_usages_result(result))


def render_usages_result(result: UsagesResult) -> Table:
    """One table: the mentions found, those left out and why, the usages, in scope and drawn."""
    table = Table(title=f"The usages of {len(result.parts)} parts")
    table.add_column("Mentions")
    table.add_column("Count", justify="right")
    for key, label in COUNT_LABELS.items():
        table.add_row(label, f"{result.counts.get(key, 0):,}")
    return table
