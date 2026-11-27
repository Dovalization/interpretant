"""The ``build-junk-gate`` command: the classifier's scores, the OCR scores and the cell screen."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog.config import CATALOG_FILE, CatalogConfig
from interpretant.cli.jobs import run_on_command_line
from interpretant.cli.options import default_process_count, process_count_option
from interpretant.pipelines.build_junk_gate import (
    ClassifierSettings,
    JunkGateResult,
    OcrSettings,
    ScreenSettings,
    build_junk_gate,
)
from interpretant.pipelines.progress import Progress


@click.command("build-junk-gate")
@click.option(
    "--catalog",
    "catalog_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=CATALOG_FILE,
    show_default=True,
    help="The catalog file naming the inputs and outputs.",
)
@process_count_option
def build_junk_gate_command(catalog_file: Path, process_count: int | None) -> None:
    """Score every record behind a drawn usage for junk and screen the cells (docs/junk_gate.md).

    The classifier (params.yaml's junk_gate.classifier) learns from the judges' labels on L and E:
    it is cross-fitted on L by source, and the final model scores every other record. Each drawn
    usage gets its sentence's OCR score (junk_gate.ocr), and the screen (junk_gate.screen) flags the
    cells with an excess of usages from records called junk. Writes the records' scores, the
    usages' OCR scores, the cells, the classifier and metrics/junk_gate.json (words by their
    reported names). Each worker process takes whole parts. Progress is shown live: a bar in a
    terminal, a line every half minute in a log file.
    """
    run_on_command_line(
        "build-junk-gate",
        catalog_file,
        lambda catalog, progress, console: run_build_junk_gate(
            catalog, process_count, progress=progress, console=console
        ),
    )


def run_build_junk_gate(
    catalog: CatalogConfig, process_count: int | None, *, progress: Progress, console: Console
) -> dict[str, int]:
    """Write the gate's scores, cells and metrics; their counts for the run log."""
    result = build_junk_gate(
        catalog,
        ClassifierSettings(),
        ScreenSettings(),
        OcrSettings(),
        process_count=process_count or default_process_count(),
        progress=progress,
    )
    console.print(format_junk_gate_result(result))
    return {
        "records": result.record_count,
        "usages": result.usage_count,
        "labelled_records": result.labelled_count,
        "meets_bar": int(result.quality.meets_bar),
        "gated_cells": result.gated_count,
        "flagged_cells": result.flagged_count,
    }


def format_junk_gate_result(result: JunkGateResult) -> str:
    """The result in two lines: the classifier against its bar, then the screen."""
    sensitivity, specificity = result.quality.sensitivity, result.quality.specificity
    verdict = "meets" if result.quality.meets_bar else "fails"
    return (
        f"{result.record_count:,} records and {result.usage_count:,} drawn usages scored; "
        f"cross-fitted on {result.labelled_count:,} records of L: sensitivity "
        f"{sensitivity.share:.3f} ({sensitivity.count} of {sensitivity.total}), specificity "
        f"{specificity.share:.3f} ({specificity.count} of {specificity.total}), {verdict} the bar\n"
        f"{result.flagged_count} of {result.gated_count} gated cells flagged by the screen"
    )
