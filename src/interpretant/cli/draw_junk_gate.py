"""The ``draw-junk-gate`` command: the junk gate's records as review items, with key and counts."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog.config import CATALOG_FILE, CatalogConfig
from interpretant.cli.jobs import run_on_command_line
from interpretant.cli.options import default_process_count, process_count_option
from interpretant.pipelines.draw_junk_gate import draw_junk_gate
from interpretant.pipelines.progress import Progress
from interpretant.review.junk_gate import DRAW_SETS, DrawSet


@click.command("draw-junk-gate")
@click.argument("out", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--set",
    "draw_set",
    type=click.Choice(DRAW_SETS),
    default="sample",
    show_default=True,
    help="sample: L and E; run3: run 3's records; confirmation: the flagged cells' samples.",
)
@click.option(
    "--catalog",
    "catalog_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=CATALOG_FILE,
    show_default=True,
    help="The catalog file naming the inputs and outputs.",
)
@process_count_option
def draw_junk_gate_command(
    out: Path, draw_set: DrawSet, catalog_file: Path, process_count: int | None
) -> None:
    """Draw one of the junk gate's sets of records as review items into OUT (docs/junk_gate.md).

    sample: the labelled sample (each decade's 500 drawn usages with the smallest keys) and the
    enrichment set (200 records of each stratum likely to be junk), together; run3: run 3's 200
    judged records with the text they had then; confirmation: 40 drawn usages of each cell the
    screen flags, reusing the labels already made. Writes the items in batches (batch_NN.jsonl:
    they quote the corpus and stay out of git), key.json and counts.json.
    """
    run_on_command_line(
        f"draw-junk-gate {draw_set}",
        catalog_file,
        lambda catalog, progress, console: run_draw_junk_gate(
            catalog, out, draw_set, process_count, progress=progress, console=console
        ),
    )


def run_draw_junk_gate(
    catalog: CatalogConfig,
    out: Path,
    draw_set: DrawSet,
    process_count: int | None,
    *,
    progress: Progress,
    console: Console,
) -> dict[str, int]:
    """Draw the set's items into ``out``; the items by set."""
    draw = draw_junk_gate(
        catalog,
        out,
        draw_set=draw_set,
        process_count=process_count or default_process_count(),
        progress=progress,
    )
    console.print(f"{draw.item_count} items {draw.items_by_set} in {draw.batch_count} batches")
    return {
        "items": draw.item_count,
        **{f"{name}_items": count for name, count in draw.items_by_set.items()},
    }
