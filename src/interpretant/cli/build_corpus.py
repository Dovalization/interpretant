"""The ``build-corpus`` command: runs the build_corpus pipeline and prints what each rule took."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from interpretant.catalog import CATALOG_FILE, read_catalog
from interpretant.pipelines.build_corpus import BuildCorpusResult, BuildCorpusSettings, build_corpus


@click.command("build-corpus")
@click.option(
    "--field",
    "fields",
    multiple=True,
    help="Only this field; repeat it. Default: the fields in params.yaml.",
)
@click.option(
    "--decade",
    "decades",
    multiple=True,
    type=int,
    help="Only this decade; repeat it. Default: the decades in params.yaml.",
)
@click.option(
    "--catalog",
    "catalog_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=CATALOG_FILE,
    show_default=True,
    help="The catalog file naming the inputs and outputs.",
)
@click.option("--batch-size", default=256, show_default=True, help="Abstracts per spaCy batch.")
@click.option("--process-count", default=1, show_default=True, help="spaCy worker processes.")
def build_corpus_command(
    fields: tuple[str, ...],
    decades: tuple[int, ...],
    catalog_file: Path,
    batch_size: int,
    process_count: int,
) -> None:
    """Annotate the raw OpenAlex cells and build the abstracts table."""
    overrides = {name: value for name, value in (("fields", fields), ("decades", decades)) if value}
    settings = BuildCorpusSettings.model_validate(BuildCorpusSettings().model_dump() | overrides)
    result = build_corpus(
        read_catalog(catalog_file), settings, batch_size=batch_size, process_count=process_count
    )
    Console().print(render_result(result))


def render_result(result: BuildCorpusResult) -> Table:
    """One table: the run's totals, then the count for each drop and removal reason."""
    table = Table(title=f"{result.abstract_count:,} abstracts from {result.cell_count} cells")
    table.add_column("Rule")
    table.add_column("Reason")
    table.add_column("Count", justify="right")
    for reason, count in result.dropped_counts.items():
        table.add_row("dropped", reason, f"{count:,}")
    for reason, count in result.removal_counts.items():
        table.add_row("removed range", reason, f"{count:,}")
    return table
