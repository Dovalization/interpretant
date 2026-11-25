"""The ``build-usages`` command: every target mention in a kept work, and the one drawn per work."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from interpretant.catalog.config import CATALOG_FILE, CatalogConfig
from interpretant.cli.jobs import run_on_command_line
from interpretant.cli.options import default_process_count, process_count_option
from interpretant.pipelines.build_usages import (
    BATCH_TOKENS,
    UsagesResult,
    UsagesSettings,
    build_usages,
)
from interpretant.pipelines.params import CorpusSelection
from interpretant.pipelines.progress import Progress

COUNT_LABELS = {
    "mentions": "Mentions",
    "staged_sentences": "Target sentences annotated",
    "skipped_parts": "Parts kept from an earlier run",
    "cache_hits": "Sentences read from the annotation cache",
    "cache_misses": "Sentences annotated by the transformer",
    "shared_sentences": "Repeated sentences given one annotation",
    "inside_removed_ranges": "In removed ranges",
    "outside_sentences": "Outside sentences",
    "in_removed_sentences": "In removed sentences",
    "usages": "Usages",
    "in_scope": "In scope",
    "drawn": "Drawn",
    "copy_works": "Works copying an earlier one",
    "copies_undrawn": "Drawn usages given up as copies",
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
    default=7,
    show_default=True,
    help="Processes sharing the GPU when the transformer annotates the usages.",
)
@click.option(
    "--batch-tokens",
    type=click.IntRange(min=100),
    default=BATCH_TOKENS,
    show_default=True,
    help="Most tokens in one transformer batch: smaller batches let more processes share the GPU.",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Keep the parts the transformer already wrote with the current columns, and skip them.",
)
@click.option(
    "--reuse-cache",
    is_flag=True,
    help="Read sentences already annotated from the annotation cache instead of annotating them.",
)
@process_count_option
def build_usages_command(
    catalog_file: Path,
    fields: tuple[str, ...],
    decades: tuple[int, ...],
    gpu_process_count: int,
    batch_tokens: int,
    process_count: int | None,
    *,
    resume: bool,
    reuse_cache: bool,
) -> None:
    """Write one row per target mention in a kept work, and draw one usage per work and word.

    A mention inside a removed range is left out. Each usage records its sentence, its
    annotation, whether it sits inside a name, whether it is in scope, and whether it is the one
    drawn: the eligible usage with the smallest draw key (decision agenda 4.1-4.6).

    Each worker process takes whole parts. Progress is shown live: a bar in a terminal, a line
    every half minute in a log file.
    """
    run_on_command_line(
        "build-usages",
        catalog_file,
        lambda catalog, progress, console: run_build_usages(
            catalog,
            fields,
            decades,
            gpu_process_count,
            batch_tokens,
            process_count,
            resume=resume,
            reuse_cache=reuse_cache,
            progress=progress,
            console=console,
        ),
    )


def run_build_usages(
    catalog: CatalogConfig,
    fields: tuple[str, ...],
    decades: tuple[int, ...],
    gpu_process_count: int,
    batch_tokens: int,
    process_count: int | None,
    *,
    resume: bool,
    reuse_cache: bool,
    progress: Progress,
    console: Console,
) -> dict[str, int]:
    """Write the usages of the selected cells; the mentions and usages for the run log."""
    overrides: dict[str, Any] = {}  # only the options given override params.yaml
    if fields:
        overrides["fields"] = fields
    if decades:
        overrides["decades"] = decades
    result = build_usages(
        catalog,
        CorpusSelection(**overrides),
        UsagesSettings(),
        process_count=process_count or default_process_count(),
        gpu_process_count=gpu_process_count,
        batch_tokens=batch_tokens,
        resume=resume,
        reuse_cache=reuse_cache,
        progress=progress,
    )
    console.print(render_usages_result(result))
    return dict(result.counts)


def render_usages_result(result: UsagesResult) -> Table:
    """One table: the mentions found, those left out and why, the usages, in scope and drawn."""
    table = Table(title=f"The usages of {len(result.parts)} parts")
    table.add_column("Mentions")
    table.add_column("Count", justify="right")
    for key, label in COUNT_LABELS.items():
        table.add_row(label, f"{result.counts.get(key, 0):,}")
    return table
