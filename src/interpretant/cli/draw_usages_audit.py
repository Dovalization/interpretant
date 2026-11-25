"""The ``draw-usages-audit`` command: the matching audit's items, key, calibration and counts."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog.config import CATALOG_FILE, CatalogConfig
from interpretant.cli.jobs import run_on_command_line
from interpretant.cli.options import default_process_count, process_count_option
from interpretant.pipelines.draw_usages_audit import draw_usages_audit
from interpretant.pipelines.progress import Progress
from interpretant.review.usages_audit import AUDIT_SEED


@click.command("draw-usages-audit")
@click.argument("out", type=click.Path(file_okay=False, path_type=Path))
@click.option("--seed", default=AUDIT_SEED, show_default=True, help="The draws' seed.")
@click.option(
    "--skip-calibration",
    is_flag=True,
    help="Write no calibration set: a redraw keeps the calibration of record.",
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
def draw_usages_audit_command(
    out: Path,
    seed: int,
    catalog_file: Path,
    process_count: int | None,
    *,
    skip_calibration: bool,
) -> None:
    """Draw the matching audit's kept, left-out and missed usages as review items into OUT.

    20 drawn in-scope usages per word spread over decades, 10 left-out usages per word, and 20
    candidate misses per word from an independent search of the kept texts. Writes the items in
    shuffled batches (batch_NN.jsonl: they quote corpus sentences and stay out of git), key.json,
    calibration.json and counts.json. A redraw after a failed bar takes its own seed and
    --skip-calibration.
    """
    run_on_command_line(
        f"draw-usages-audit {seed}",
        catalog_file,
        lambda catalog, progress, console: run_draw_usages_audit(
            catalog,
            out,
            seed,
            process_count,
            skip_calibration=skip_calibration,
            progress=progress,
            console=console,
        ),
    )


def run_draw_usages_audit(
    catalog: CatalogConfig,
    out: Path,
    seed: int,
    process_count: int | None,
    *,
    skip_calibration: bool,
    progress: Progress,
    console: Console,
) -> dict[str, int]:
    """Draw the audit's items into ``out``; the items by kind and the candidate misses."""
    draw = draw_usages_audit(
        catalog,
        out,
        seed=seed,
        skip_calibration=skip_calibration,
        process_count=process_count or default_process_count(),
        progress=progress,
    )
    console.print(
        f"{draw.item_count} items {draw.items_by_kind} in {draw.batch_count} batches; "
        f"{draw.candidate_count} candidate misses over {draw.candidate_word_count} words"
    )
    return {
        "items": draw.item_count,
        **{f"{kind}_items": count for kind, count in draw.items_by_kind.items()},
        "candidate_misses": draw.candidate_count,
    }
