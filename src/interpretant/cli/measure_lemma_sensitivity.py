"""The ``measure-lemma-sensitivity`` command: round L1 of docs/predictions_lemmas.md."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from interpretant.catalog.abstracts import load_abstracts
from interpretant.catalog.config import CATALOG_FILE, read_catalog
from interpretant.cli.options import default_process_count, process_count_option
from interpretant.cli.progress import LiveProgress
from interpretant.corpus.audit import load_target_words
from interpretant.embedding.ppmi import SvdBackend
from interpretant.pipelines.lemma_sensitivity import (
    LemmaSensitivityResult,
    measure_lemma_sensitivity,
)
from interpretant.pipelines.progress import StageTimer
from interpretant.pipelines.runs import record_run

# The abstracts columns the cells are made from.
_ABSTRACT_COLUMNS = ("work_id", "fields", "decade", "text", "removals", "dropped_reason", "source")


@click.command("measure-lemma-sensitivity")
@click.option(
    "--catalog",
    "catalog_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=CATALOG_FILE,
    show_default=True,
    help="The catalog file; its static view is view B, with the lemma exceptions.",
)
@click.option(
    "--control-view",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/interim/static_view_without_exceptions"),
    show_default=True,
    help="View A: the static view built without the lemma exceptions (round L1's R_A).",
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
    default=Path("data/processed/lemma_sensitivity"),
    show_default=True,
    help="Where the movements, dose and target-count tables are written.",
)
@click.option("--field", "fields", multiple=True, help="Only these fields (a timing run).")
@click.option("--bootstraps", default=9, show_default=True, help="Bootstrap samples per set.")
@click.option(
    "--svd-backend",
    type=click.Choice(["arpack", "cuda"]),
    default="arpack",
    show_default=True,
    help="ARPACK on the CPU, or cuSOLVER on the GPU (only once checked against ARPACK).",
)
@process_count_option
def measure_lemma_sensitivity_command(
    catalog_file: Path,
    control_view: Path,
    targets_file: Path,
    out_dir: Path,
    fields: tuple[str, ...],
    bootstraps: int,
    svd_backend: SvdBackend,
    process_count: int | None,
) -> None:
    """Measure whether the lemma exceptions move bootstrap-averaged static drift (round L1).

    Each worker process takes one field. Progress is shown live: a bar in a terminal, a line
    every half minute in a log file.
    """
    console = Console()
    treatment = read_catalog(catalog_file)
    entry = treatment.static_view.model_copy(update={"path": control_view.resolve()})
    control = treatment.model_copy(update={"static_view": entry})
    targets = frozenset(word for word in load_target_words(targets_file) if "_" not in word)
    with LiveProgress(console) as live:
        timer = StageTimer(live)
        timer.stage("reading the build of record")
        abstracts = load_abstracts(treatment).select(_ABSTRACT_COLUMNS).collect()
        result = measure_lemma_sensitivity(
            control,
            treatment,
            abstracts,
            targets=targets,
            bootstraps=bootstraps,
            fields=fields,
            svd_backend=svd_backend,
            process_count=process_count or default_process_count(),
            progress=timer,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    result.movements.write_parquet(out_dir / "movements.parquet")
    result.doses.write_parquet(out_dir / "doses.parquet")
    result.target_counts.write_parquet(out_dir / "target_counts.parquet")
    counts = {
        "comparisons": result.movements.height,
        "models": result.models,
        "changed_tokens": int(result.doses["changed_tokens"].sum()),
    }
    record_run(treatment, command="measure-lemma-sensitivity", stages=timer.finish(), counts=counts)
    console.print(render_lemma_sensitivity_result(result))


def render_lemma_sensitivity_result(result: LemmaSensitivityResult) -> Table:
    """One table: the cells, their tokens, the tokens the exceptions change, the models."""
    doses = result.doses
    table = Table(title=f"{result.models:,} models, {result.movements.height:,} movements")
    table.add_column("Cells", justify="right")
    table.add_column("Works", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Changed by the exceptions", justify="right")
    tokens = int(doses["tokens"].sum())
    changed = int(doses["changed_tokens"].sum())
    table.add_row(
        f"{doses.height}",
        f"{int(doses['works'].sum()):,}",
        f"{tokens:,}",
        f"{changed:,} ({changed / max(tokens, 1):.2%})",
    )
    return table
