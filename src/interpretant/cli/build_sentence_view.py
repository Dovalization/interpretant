"""The ``build-sentence-view`` command: every kept work's sentences, with their exclusions."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from interpretant.catalog.config import CATALOG_FILE, read_catalog
from interpretant.cli.options import default_process_count, process_count_option
from interpretant.cli.progress import LiveProgress
from interpretant.pipelines.build_sentence_view import SentenceViewResult, build_sentence_view
from interpretant.pipelines.progress import StageTimer
from interpretant.pipelines.runs import record_run


@click.command("build-sentence-view")
@click.option(
    "--catalog",
    "catalog_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=CATALOG_FILE,
    show_default=True,
    help="The catalog file naming the inputs and outputs.",
)
@process_count_option
def build_sentence_view_command(catalog_file: Path, process_count: int | None) -> None:
    """Write one row per sentence of every kept work, marking those no usage is drawn from.

    Cuts the merge rules find wrong are joined first, and breaks the split rule finds are cut. A
    sentence whose every non-space character is removed is marked removed; the unfinished last
    sentence of a truncated work is marked cut off (docs/sentence_view.md).

    Each worker process takes whole parts. Progress is shown live: a bar in a terminal, a line
    every half minute in a log file.
    """
    console = Console()
    catalog = read_catalog(catalog_file)
    with LiveProgress(console) as live:
        timer = StageTimer(live)
        result = build_sentence_view(
            catalog, process_count=process_count or default_process_count(), progress=timer
        )
    counts = {
        "works": result.work_count,
        "sentences": result.sentence_count,
        "joined": result.joined_count,
        "split": result.split_count,
        "removed": result.removed_count,
        "cut_off": result.cut_off_count,
    }
    record_run(catalog, command="build-sentence-view", stages=timer.finish(), counts=counts)
    console.print(render_sentence_view_result(result))


def render_sentence_view_result(result: SentenceViewResult) -> Table:
    """One table: the works and sentences written, the joins, and the sentences not drawn from."""
    table = Table(title=f"The sentence view of {len(result.parts)} parts")
    for column in ("Works", "Sentences", "Joined", "Split", "Removed", "Cut off"):
        table.add_column(column, justify="right")
    table.add_row(
        f"{result.work_count:,}",
        f"{result.sentence_count:,}",
        f"{result.joined_count:,}",
        f"{result.split_count:,}",
        f"{result.removed_count:,}",
        f"{result.cut_off_count:,}",
    )
    return table
