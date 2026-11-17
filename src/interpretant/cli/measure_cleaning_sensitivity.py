"""The ``measure-cleaning-sensitivity`` command: round S1 of docs/predictions_cleaning.md."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from interpretant.catalog.config import CATALOG_FILE, read_catalog
from interpretant.catalog.revisions import load_abstracts_at
from interpretant.cli.options import default_process_count, process_count_option
from interpretant.cli.progress import LiveProgress
from interpretant.corpus.audit import load_target_words
from interpretant.pipelines.cleaning_sensitivity import (
    SensitivityResult,
    SensitivitySettings,
    measure_cleaning_sensitivity,
)
from interpretant.pipelines.progress import StageTimer
from interpretant.pipelines.runs import record_run

# The builds of round S1, fixed in its pre-registration: before and after the cleaning batch.
BASE_REVISION = "2e20238c"
TREATMENT_REVISION = "75f36a82"


@click.command("measure-cleaning-sensitivity")
@click.option(
    "--catalog",
    "catalog_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=CATALOG_FILE,
    show_default=True,
    help="The catalog file naming the inputs and outputs.",
)
@click.option("--base-revision", default=BASE_REVISION, show_default=True, help="P's build.")
@click.option(
    "--treatment-revision",
    default=TREATMENT_REVISION,
    show_default=True,
    help="The build whose drops and removals C applies.",
)
@click.option(
    "--targets",
    "targets_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("data/words_of_interest.txt"),
    show_default=True,
    help="The targets; only single words are used, and none is named in the movements.",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/processed/cleaning_sensitivity"),
    show_default=True,
    help="Where the movements and dose tables are written.",
)
@click.option("--field", "fields", multiple=True, help="Only these fields (a timing run).")
@click.option("--placebos", default=19, show_default=True, help="Placebo versions per cell.")
@click.option("--bootstraps", default=9, show_default=True, help="Bootstrap samples per cell.")
@click.option(
    "--averaged",
    is_flag=True,
    help="Round S1b: drift averaged over bootstrap models of P, of C and of P again.",
)
@process_count_option
def measure_cleaning_sensitivity_command(
    catalog_file: Path,
    base_revision: str,
    treatment_revision: str,
    targets_file: Path,
    out_dir: Path,
    fields: tuple[str, ...],
    placebos: int,
    bootstraps: int,
    process_count: int | None,
    *,
    averaged: bool,
) -> None:
    """Measure whether removing the identifiable junk moves static drift (round S1).

    Each worker process takes one field. Progress is shown live: a bar in a terminal, a line
    every half minute in a log file.
    """
    console = Console()
    catalog = read_catalog(catalog_file)
    repository = catalog_file.resolve().parent
    targets = frozenset(word for word in load_target_words(targets_file) if "_" not in word)
    with LiveProgress(console) as live:
        timer = StageTimer(live)
        timer.stage("reading the two builds")
        base = load_abstracts_at(catalog, repository, base_revision)
        later = load_abstracts_at(catalog, repository, treatment_revision)
        result = measure_cleaning_sensitivity(
            catalog,
            base,
            later,
            targets=targets,
            settings=SensitivitySettings(
                placebos=placebos, bootstraps=bootstraps, fields=fields, averaged=averaged
            ),
            process_count=process_count or default_process_count(),
            progress=timer,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    result.movements.write_parquet(out_dir / "movements.parquet")
    result.doses.write_parquet(out_dir / "doses.parquet")
    result.target_doses.write_parquet(out_dir / "target_doses.parquet")
    counts = {"comparisons": result.movements.height, "models": result.models}
    record_run(
        catalog, command="measure-cleaning-sensitivity", stages=timer.finish(), counts=counts
    )
    console.print(render_sensitivity_result(result))


def render_sensitivity_result(result: SensitivityResult) -> Table:
    """One table: the cells, their dose, and the models trained."""
    doses = result.doses
    table = Table(title=f"{result.models:,} models, {result.movements.height:,} movements")
    table.add_column("Cells", justify="right")
    table.add_column("Works", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Removed by C", justify="right")
    tokens = int(doses["tokens"].sum())
    removed = int(doses["removed_tokens"].sum())
    table.add_row(
        f"{doses.height}",
        f"{int(doses['works'].sum()):,}",
        f"{tokens:,}",
        f"{removed:,} ({removed / max(tokens, 1):.1%})",
    )
    return table
