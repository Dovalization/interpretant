"""CLI entry point for interpretant."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from interpretant.cli.annotate_corpus import annotate_corpus_command
from interpretant.cli.build_corpus import build_corpus_command
from interpretant.cli.build_linked_tables import build_linked_tables_command
from interpretant.cli.build_static_view import build_static_view_command
from interpretant.cli.clean_corpus import clean_corpus_command
from interpretant.cli.compare_corpus import compare_corpus_command
from interpretant.cli.count_word_repairs import count_word_repairs_command
from interpretant.cli.detect_languages import detect_languages_command
from interpretant.cli.fetch_linked_sources import fetch_linked_sources_command
from interpretant.cli.fetch_metadata import fetch_metadata_command
from interpretant.cli.measure_cleaning_sensitivity import measure_cleaning_sensitivity_command
from interpretant.cli.measure_lemma_sensitivity import measure_lemma_sensitivity_command
from interpretant.cli.report_corpus import report_corpus_command

console = Console()


@click.group()
@click.version_option(package_name="interpretant")
def cli() -> None:
    """interpretant — diachronic semantic drift tracker for academic corpora."""


cli.add_command(fetch_metadata_command)
cli.add_command(fetch_linked_sources_command)
cli.add_command(build_linked_tables_command)
cli.add_command(count_word_repairs_command)
cli.add_command(annotate_corpus_command)
cli.add_command(detect_languages_command)
cli.add_command(build_static_view_command)
cli.add_command(clean_corpus_command)
cli.add_command(build_corpus_command)
cli.add_command(report_corpus_command)
cli.add_command(compare_corpus_command)
cli.add_command(measure_cleaning_sensitivity_command)
cli.add_command(measure_lemma_sensitivity_command)


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------


@cli.group()
def corpus() -> None:
    """Corpus loading and preprocessing commands."""


@corpus.group("openalex")
def corpus_openalex() -> None:
    """OpenAlex abstract samples per field and decade (the field × decade grid)."""


@corpus_openalex.command("download")
@click.option(
    "--raw-dir",
    type=click.Path(path_type=Path),
    default=Path("data/raw/openalex"),
    show_default=True,
)
@click.option("--field", "fields", multiple=True, help="Field slug; repeat it. Default: all.")
@click.option("--start", default=1950, show_default=True, help="First decade (inclusive).")
@click.option("--end", default=2030, show_default=True, help="Last decade (exclusive).")
@click.option(
    "--sample-size", default=10_000, show_default=True, help="Works per cell (max 10000)."
)
@click.option("--seed", default=42, show_default=True, help="Sampling seed; keep it fixed.")
@click.option(
    "--full",
    is_flag=True,
    help="Download every matching work instead of a sample (use a separate --raw-dir).",
)
@click.option(
    "--word",
    default=None,
    help="Only works whose abstract contains this word (use a separate --raw-dir).",
)
def corpus_openalex_download(
    raw_dir: Path,
    fields: tuple[str, ...],
    start: int,
    end: int,
    sample_size: int,
    seed: int,
    full: bool,
    word: str | None,
) -> None:
    """Download English research-prose abstracts for each field and decade.

    Finished cells are skipped, so a run stopped by the daily credit budget resumes where it
    left off.
    """
    from interpretant.corpus.openalex import (
        FIELDS,
        BudgetExhaustedError,
        OpenAlexClient,
        OpenAlexSettings,
        download_cell,
    )

    unknown = set(fields) - FIELDS.keys()
    if unknown:
        raise click.BadParameter(f"unknown fields {sorted(unknown)}; choose from {sorted(FIELDS)}")
    client = OpenAlexClient(OpenAlexSettings().openalex_api_key)
    for field in fields or FIELDS:
        for decade in range(start, end, 10):
            try:
                works = download_cell(client, raw_dir, field, decade, sample_size, seed, full, word)
            except BudgetExhaustedError as error:
                console.print(f"[yellow]Daily budget spent ({error}); rerun to resume.[/yellow]")
                return
            console.print(f"  {field:14} {decade}s  {works:>6,} works")
    console.print("[green]Done.[/green]")


@corpus_openalex.command("sources")
@click.option(
    "--raw-dir",
    type=click.Path(path_type=Path),
    default=Path("data/raw/openalex"),
    show_default=True,
)
@click.option("--field", "fields", multiple=True, default=("hci",), show_default=True)
@click.option(
    "--decade", "decades", multiple=True, type=int, default=(1990, 2020), show_default=True
)
def corpus_openalex_sources(
    raw_dir: Path, fields: tuple[str, ...], decades: tuple[int, ...]
) -> None:
    """Look up the primary source (venue) of every work in the given cells (needs the API key)."""
    from interpretant.corpus.openalex import (
        OpenAlexClient,
        OpenAlexSettings,
        fetch_sources,
    )

    client = OpenAlexClient(OpenAlexSettings().openalex_api_key)
    for field in fields:
        for decade in decades:
            output = fetch_sources(client, raw_dir, field, decade)
            console.print(f"  {field} {decade}s → {output}")


@corpus_openalex.command("ingest")
@click.option(
    "--raw-dir",
    type=click.Path(path_type=Path),
    default=Path("data/raw/openalex"),
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/processed/openalex/texts"),
    show_default=True,
)
@click.option("--start", default=1950, show_default=True, help="First decade (inclusive).")
@click.option("--end", default=2030, show_default=True, help="Last decade (exclusive).")
def corpus_openalex_ingest(raw_dir: Path, output_dir: Path, start: int, end: int) -> None:
    """Clean and preprocess downloaded cells into ``<field>_<decade>.txt``, one per line.

    The table shows documents kept per cell and, after the minus sign, how many were dropped
    as duplicates, archive catalogue or citation records, scraped web pages, or by the 1% author
    cap.
    """
    from interpretant.corpus.openalex import ingest_fields

    decades = list(range(start, end, 10))
    cleaned = ingest_fields(raw_dir, output_dir, decades)
    table = Table(title="OpenAlex ingest (kept −dropped per cell)", show_lines=False)
    table.add_column("Field", style="cyan")
    for decade in decades:
        table.add_column(str(decade), justify="right")
    for field, cells in cleaned.items():
        table.add_row(
            field,
            *(
                f"{len(cell.documents):,} −{cell.dropped:,}"
                if cell.dropped
                else f"{len(cell.documents):,}"
                for cell in cells.values()
            ),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# twec
# ---------------------------------------------------------------------------


def _configure_gensim_logging() -> None:
    """Silence gensim's harmless Apple Silicon finalizer noise and show training progress."""
    import logging
    import sys

    # Suppress the harmless gensim Apple Silicon finalizer spam.
    # our_dot_float is a Cython symbol missing from the pre-built wheel;
    # Python emits "Exception ignored" for it but it doesn't affect training.
    _prev_unraisable_hook = sys.unraisablehook

    def _unraisable_hook(args: sys.UnraisableHookArgs) -> None:
        # Check object, exc_type name, and exc_value string to be thorough.
        haystack = " ".join(
            [
                str(getattr(args, "object", "") or ""),
                getattr(getattr(args, "exc_type", None), "__name__", ""),
                str(getattr(args, "exc_value", "") or ""),
            ]
        )
        if "our_dot_float" in haystack:
            return
        _prev_unraisable_hook(args)

    sys.unraisablehook = _unraisable_hook

    # Enable gensim progress logging so mid-epoch word/sec updates appear in the log.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger("gensim").setLevel(logging.INFO)


@cli.group()
def twec() -> None:
    """Temporal Word Embeddings with a Compass (TWEC) commands."""


@twec.command("grid")
@click.option(
    "--input-dir",
    type=click.Path(path_type=Path),
    default=Path("data/phrased"),
    show_default=True,
)
@click.option(
    "--layer",
    "layers",
    multiple=True,
    default=("openalex",),
    show_default=True,
    help="Main layers: they define the compass and the comparable grid.",
)
@click.option(
    "--extra-layer",
    "extra_layers",
    multiple=True,
    default=("openalex_full",),
    show_default=True,
    help="Layers trained against the compass but left out of it (full cells).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("models/twec_grid"),
    show_default=True,
)
@click.option("--compass-tokens-per-cell", default=200_000, show_default=True)
@click.option("--vector-size", default=100, show_default=True)
@click.option("--window", default=5, show_default=True)
@click.option("--min-count", default=5, show_default=True, help="Compass vocabulary threshold.")
@click.option("--epochs", default=10, show_default=True)
@click.option("--negative", default=5, show_default=True, help="Negative samples per word.")
@click.option("--workers", default=8, show_default=True)
@click.option("--seed", default=42, show_default=True)
def twec_grid(
    input_dir: Path,
    layers: tuple[str, ...],
    extra_layers: tuple[str, ...],
    output_dir: Path,
    compass_tokens_per_cell: int,
    vector_size: int,
    window: int,
    min_count: int,
    epochs: int,
    negative: int,
    workers: int,
    seed: int,
) -> None:
    """Train one TWEC model per field × decade cell on a balanced compass.

    The compass is trained on the same token budget from every main cell; each cell's model
    is then trained with the compass's context vectors frozen, so every cell shares one
    space. Finished cells are skipped, so an interrupted run resumes.
    """
    from interpretant.embedding.grid import discover_cells, train_twec_grid
    from interpretant.embedding.twec import TWECTrainer

    _configure_gensim_logging()
    main_cells = discover_cells(input_dir, layers)
    extra_cells = discover_cells(input_dir, extra_layers)
    trainer = TWECTrainer(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        negative=negative,
        seed=seed,
    )
    console.print(f"[bold]TWEC grid[/bold]: {len(main_cells)} main cells, {len(extra_cells)} extra")
    console.print(
        f"  dims {vector_size}  window {window}  epochs {epochs}  negative {negative}"
        f"  min_count {min_count}"
    )
    written = train_twec_grid(
        trainer, main_cells, extra_cells, output_dir, compass_tokens_per_cell, console.print
    )
    console.print(f"[green]Done.[/green] {len(written)} new models in {output_dir}")


@cli.group()
def ppmi() -> None:
    """Count-based PPMI-SVD embeddings per cell (robustness check on TWEC)."""


@ppmi.command("grid")
@click.option(
    "--input-dir",
    type=click.Path(path_type=Path),
    default=Path("data/phrased"),
    show_default=True,
)
@click.option(
    "--layer",
    "layers",
    multiple=True,
    default=("openalex", "openalex_full"),
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("models/ppmi_grid"),
    show_default=True,
)
@click.option("--window", default=5, show_default=True)
@click.option("--dimensions", default=100, show_default=True)
@click.option("--min-count", default=20, show_default=True, help="Vocabulary threshold per cell.")
def ppmi_grid(
    input_dir: Path,
    layers: tuple[str, ...],
    output_dir: Path,
    window: int,
    dimensions: int,
    min_count: int,
) -> None:
    """Build PPMI-SVD vectors for every cell; finished cells are skipped.

    Each cell is factorised on its own, so compare PPMI cells by neighbourhood shift only.
    """
    from interpretant.embedding.grid import build_ppmi_grid, discover_cells

    cells = discover_cells(input_dir, layers)
    console.print(
        f"[bold]PPMI-SVD grid[/bold]: {len(cells)} cells, window {window}, {dimensions} dims"
    )
    written = build_ppmi_grid(cells, output_dir, window, dimensions, min_count)
    console.print(f"[green]Done.[/green] {len(written)} new models in {output_dir}")


# ---------------------------------------------------------------------------
# drift
# ---------------------------------------------------------------------------


@cli.group()
def drift() -> None:
    """Drift computation commands."""


@drift.command("grid")
@click.option(
    "--model-dir",
    type=click.Path(path_type=Path),
    default=Path("models/twec_grid"),
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/drift"),
    show_default=True,
)
@click.option(
    "--words-file",
    type=click.Path(path_type=Path),
    default=Path("data/words_of_interest.txt"),
    show_default=True,
)
@click.option(
    "--controls-file",
    type=click.Path(path_type=Path),
    default=Path("data/control_words.txt"),
    show_default=True,
)
@click.option("--measure-min", default=150, show_default=True, help="Occurrences in both cells.")
@click.option("--neighbour-min", default=50, show_default=True, help="Occurrences for a neighbour.")
@click.option("--neighbours", default=25, show_default=True)
def drift_grid(
    model_dir: Path,
    output_dir: Path,
    words_file: Path,
    controls_file: Path,
    measure_min: int,
    neighbour_min: int,
    neighbours: int,
) -> None:
    """Measure drift on the grid: within-field trajectories and cross-field divergence.

    Writes within_field.parquet (each field's consecutive decades, including the full
    cells) and cross_field.parquet (every two OpenAlex fields in the same decade),
    one row per word measured in a pair, with percentile ranks against the pair's
    whole-vocabulary noise floor and a candidate flag.
    """
    import polars as pl

    from interpretant.corpus.audit import load_target_words
    from interpretant.drift.grid import measure_grid

    try:
        tables = measure_grid(
            model_dir,
            load_target_words(words_file),
            load_target_words(controls_file),
            measure_min,
            neighbour_min,
            neighbours,
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.write_parquet(output_dir / f"{name}.parquet")
        pairs = table.select("layer", "field_a", "decade_a", "field_b", "decade_b").n_unique()
        control_rate = table.filter(pl.col("is_control")).select(pl.col("candidate").mean()).item()
        target_candidates = (
            table.filter(pl.col("is_target")).select(pl.col("candidate").sum()).item()
        )
        console.print(
            f"  {name}: {pairs} pairs, {len(table):,} rows, "
            f"{target_candidates} target candidates, "
            f"controls flagged {control_rate:.1%} (expected under ~5%)"
        )


if __name__ == "__main__":
    cli()
