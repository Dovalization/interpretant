"""CLI entry point for interpretant."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from interpretant.cli.build_corpus import build_corpus_command
from interpretant.cli.fetch_metadata import fetch_metadata_command

console = Console()


@click.group()
@click.version_option(package_name="interpretant")
def cli() -> None:
    """interpretant — diachronic semantic drift tracker for academic corpora."""


cli.add_command(build_corpus_command)
cli.add_command(fetch_metadata_command)


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


@corpus.group("phrases")
def corpus_phrases() -> None:
    """Phrase detection: merge multi-word terms into single tokens before training."""


@corpus_phrases.command("learn")
@click.option(
    "--processed-dir",
    type=click.Path(path_type=Path),
    default=Path("data/lemmatised"),
    show_default=True,
)
@click.option(
    "--words-file",
    type=click.Path(path_type=Path),
    default=Path("data/words_of_interest.txt"),
    show_default=True,
)
@click.option(
    "--source",
    "sources",
    multiple=True,
    default=("openalex",),
    show_default=True,
    help="Processed sources to include (the main corpus is the OpenAlex grid).",
)
@click.option(
    "--tokens-per-file",
    default=300_000,
    show_default=True,
    help="Token budget sampled from each processed file, so every cell weighs the same.",
)
@click.option("--min-count", default=20, show_default=True)
@click.option("--threshold", default=0.5, show_default=True, help="Normalised PMI threshold.")
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("data/lemmatised/phrases.model"),
    show_default=True,
)
def corpus_phrases_learn(
    processed_dir: Path,
    sources: tuple[str, ...],
    words_file: Path,
    tokens_per_file: int,
    min_count: int,
    threshold: float,
    output: Path,
) -> None:
    """Learn one phrase model from a balanced sample of every processed file.

    Also writes ``<output>.tsv``, the phrases sorted by score, for review.
    """
    from interpretant.corpus.audit import load_target_words
    from interpretant.corpus.files import source_texts
    from interpretant.corpus.phrases import (
        learn_phrases,
        sample_chunks,
        split_targets,
    )

    paths = source_texts(processed_dir, sources)
    protected, forced = split_targets(load_target_words(words_file))
    console.print(f"  files     : {len(paths)} (≤{tokens_per_file:,} tokens each)")
    console.print(f"  protected : {len(protected)} targets   forced : {sorted(forced)}")
    chunks = [chunk for path in paths for chunk in sample_chunks(path, tokens_per_file)]
    model = learn_phrases(chunks, protected, forced, min_count, threshold)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    ranked = sorted(model.phrases.phrasegrams.items(), key=lambda item: -item[1])
    review = output.with_suffix(".tsv")
    rows = [f"{phrase}\t{score:.3f}\t{model.joins.get(phrase, '')}\n" for phrase, score in ranked]
    review.write_text("phrase\tscore\trepaired_as\n" + "".join(rows), "utf-8")
    console.print(
        f"[green]{len(ranked) - len(model.joins):,} phrases, {len(model.joins):,} repairs[/green]"
        f" → {output} (review: {review})"
    )


@corpus_phrases.command("apply")
@click.option(
    "--processed-dir",
    type=click.Path(path_type=Path),
    default=Path("data/lemmatised"),
    show_default=True,
)
@click.option(
    "--source",
    "sources",
    multiple=True,
    default=("openalex",),
    show_default=True,
    help="Processed sources to include (the main corpus is the OpenAlex grid).",
)
@click.option(
    "--model",
    "model_path",
    type=click.Path(path_type=Path),
    default=Path("data/lemmatised/phrases.model"),
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/phrased"),
    show_default=True,
)
def corpus_phrases_apply(
    processed_dir: Path, sources: tuple[str, ...], model_path: Path, output_dir: Path
) -> None:
    """Rewrite every processed file with phrases merged, mirroring the directory layout."""
    from interpretant.corpus.files import rewrite_texts, source_texts
    from interpretant.corpus.phrases import PhraseModel

    model = PhraseModel.load(model_path)
    paths = source_texts(processed_dir, sources)
    written = rewrite_texts(paths, processed_dir, output_dir, model.apply)
    console.print(f"[green]{len(written)} files[/green] → {output_dir}")


@corpus.group("lemmas")
def corpus_lemmas() -> None:
    """Lemma map: collapse plurals and British spellings before phrase detection."""


@corpus_lemmas.command("votes")
@click.option(
    "--raw-dir",
    "raw_dirs",
    type=click.Path(path_type=Path),
    multiple=True,
    default=(Path("data/raw/openalex"), Path("data/raw/openalex_full")),
    show_default=True,
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("data/processed/lemma_votes.parquet"),
    show_default=True,
)
@click.option("--workers", default=1, show_default=True, help="Processes running spaCy.")
@click.option("--model", default="en_core_web_sm", show_default=True, help="spaCy pipeline.")
@click.option("--gpu", is_flag=True, help="Run the pipeline on the GPU, in one process.")
def corpus_lemmas_votes(
    raw_dirs: tuple[Path, ...], output: Path, workers: int, model: str, gpu: bool
) -> None:
    """Tag every cleaned abstract once with spaCy and count each word type's readings and lemmas.

    An abstract in several cells (a full cell holds its sample cell; a work can sit in two
    fields) is tagged once, the first time its work id is seen.
    """
    from collections.abc import Iterator

    from interpretant.corpus.lemmas import lemma_votes, save_votes
    from interpretant.drift.senses import cased_abstracts

    cells = [
        (raw_dir, path.parent.name, int(path.name.split(".")[0]))
        for raw_dir in raw_dirs
        for path in sorted(raw_dir.glob("*/*.jsonl.gz"))
    ]
    device = "the GPU" if gpu else f"{workers} CPU processes"
    console.print(f"  {len(cells)} raw cells, {model} on {device}")
    seen: set[str] = set()
    repeats = 0

    def texts() -> Iterator[str]:
        nonlocal repeats
        for raw_dir, field, decade in tqdm(cells, unit="cell"):
            for work_id, abstract in cased_abstracts(raw_dir, field, decade):
                if work_id in seen:
                    repeats += 1
                    continue
                seen.add(work_id)
                yield abstract

    votes = lemma_votes(texts(), workers, model, gpu)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_votes(votes, output)
    console.print(f"  {len(seen):,} abstracts tagged, {repeats:,} repeats skipped")
    console.print(
        f"[green]{sum(votes.values()):,} tokens, {len({key[0] for key in votes}):,} types[/green]"
        f" → {output}"
    )


@corpus_lemmas.command("learn")
@click.option(
    "--processed-dir",
    type=click.Path(path_type=Path),
    default=Path("data/processed"),
    show_default=True,
)
@click.option(
    "--source",
    "sources",
    multiple=True,
    default=("openalex", "openalex_full"),
    show_default=True,
)
@click.option(
    "--words-file",
    type=click.Path(path_type=Path),
    default=Path("data/words_of_interest.txt"),
    show_default=True,
)
@click.option(
    "--overrides",
    "overrides_path",
    type=click.Path(path_type=Path),
    default=Path("data/lemma_overrides.tsv"),
    show_default=True,
)
@click.option(
    "--votes",
    "votes_path",
    type=click.Path(path_type=Path),
    default=Path("data/processed/lemma_votes.parquet"),
    show_default=True,
    help="Output of `corpus lemmas votes`.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("data/processed/lemmas_voted.tsv"),
    show_default=True,
    help="Kept apart from data/processed/lemmas.tsv, the map the frozen grid was built with.",
)
@click.option("--review", default=150, show_default=True, help="Most frequent mappings to print.")
def corpus_lemmas_learn(
    processed_dir: Path,
    sources: tuple[str, ...],
    words_file: Path,
    overrides_path: Path,
    votes_path: Path,
    output: Path,
    review: int,
) -> None:
    """Learn the type-level lemma map from the processed files and spaCy's votes."""
    from collections import Counter

    from interpretant.corpus.audit import load_target_words
    from interpretant.corpus.files import source_texts
    from interpretant.corpus.lemmas import (
        count_types,
        learn_lemma_map,
        load_votes,
        read_overrides,
    )

    paths = source_texts(processed_dir, sources)
    console.print(f"  counting types in {len(paths)} files")
    counts = count_types(tqdm(paths, unit="file"))
    protected = set(load_target_words(words_file))
    votes = load_votes(votes_path)
    lemma_map = learn_lemma_map(counts, votes, protected, read_overrides(overrides_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    lemma_map.save(output, counts)
    by_rule: Counter[str] = Counter()
    tokens_by_rule: Counter[str] = Counter()
    for token, rule in lemma_map.rules.items():
        by_rule[rule] += 1
        tokens_by_rule[rule] += counts[token]
    total_tokens = sum(counts.values())
    console.print(f"  types: {len(counts):,}   tokens: {total_tokens:,}")
    for rule in sorted(by_rule):
        console.print(
            f"  {rule:9} {by_rule[rule]:>8,} types   {tokens_by_rule[rule]:>12,} tokens"
            f" ({tokens_by_rule[rule] / total_tokens:.2%})"
        )
    table = Table(title=f"{review} most frequent mappings", show_lines=False)
    for column in ("token", "lemma", "count", "rule"):
        table.add_column(column)
    ranked = sorted(lemma_map.mapping, key=lambda token: -counts[token])[:review]
    for token in ranked:
        table.add_row(token, lemma_map.mapping[token], f"{counts[token]:,}", lemma_map.rules[token])
    console.print(table)
    console.print(f"[green]{len(lemma_map.mapping):,} types mapped[/green] → {output}")


@corpus_lemmas.command("apply")
@click.option(
    "--processed-dir",
    type=click.Path(path_type=Path),
    default=Path("data/processed"),
    show_default=True,
)
@click.option(
    "--source",
    "sources",
    multiple=True,
    default=("openalex", "openalex_full"),
    show_default=True,
)
@click.option(
    "--map",
    "map_path",
    type=click.Path(path_type=Path),
    default=Path("data/processed/lemmas.tsv"),
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/lemmatised"),
    show_default=True,
)
def corpus_lemmas_apply(
    processed_dir: Path, sources: tuple[str, ...], map_path: Path, output_dir: Path
) -> None:
    """Rewrite every processed file with the lemma map applied, mirroring the layout."""
    from interpretant.corpus.files import rewrite_texts, source_texts
    from interpretant.corpus.lemmas import LemmaMap

    lemma_map = LemmaMap.load(map_path)
    paths = source_texts(processed_dir, sources)
    written = rewrite_texts(paths, processed_dir, output_dir, lemma_map.apply)
    console.print(f"[green]{len(written)} files[/green] → {output_dir}")


@corpus.command("audit")
@click.option(
    "--processed-dir",
    type=click.Path(path_type=Path),
    default=Path("data/processed"),
    show_default=True,
)
@click.option(
    "--words-file",
    type=click.Path(path_type=Path),
    default=Path("data/words_of_interest.txt"),
    show_default=True,
)
@click.option(
    "--min-occurrences",
    default=150,
    show_default=True,
    help="Occurrences a word needs in both decades for a transition to count as reliable.",
)
@click.option("--output", type=click.Path(path_type=Path), help="Also write the audit as JSON.")
def corpus_audit(
    processed_dir: Path,
    words_file: Path,
    min_occurrences: int,
    output: Path | None,
) -> None:
    """Report cell volume and target-word coverage."""
    import json

    from interpretant.corpus.audit import (
        audit_corpus,
        format_decade_ranges,
        load_target_words,
    )

    audit = audit_corpus(processed_dir, load_target_words(words_file), min_occurrences)
    columns = sorted({(cell.layer, cell.source) for cell in audit.cells})
    tokens_by_cell = {(cell.layer, cell.source, cell.decade): cell.tokens for cell in audit.cells}

    volume_table = Table(title="Tokens per cell (millions)", show_lines=False)
    volume_table.add_column("Decade", style="cyan")
    for layer, source in columns:
        volume_table.add_column(
            f"{source} (full)" if layer == "openalex_full" else source, justify="right"
        )
    volume_table.add_column("merged", justify="right", style="bold")
    for decade, total in audit.merged.items():
        cells_row = [
            f"{tokens_by_cell[(layer, source, decade)] / 1e6:.2f}"
            if (layer, source, decade) in tokens_by_cell
            else "—"
            for layer, source in columns
        ]
        volume_table.add_row(str(decade), *cells_row, f"{total.tokens / 1e6:.2f}")
    console.print(volume_table)

    transitions_total = max(len(audit.merged) - 1, 0)
    coverage_table = Table(
        title=f"Target coverage (merged, ≥{min_occurrences} occurrences)", show_lines=False
    )
    coverage_table.add_column("Word", style="magenta")
    coverage_table.add_column("Reliable transitions", justify="right")
    coverage_table.add_column("Decades at threshold")
    coverage_table.add_column("Min–max count", justify="right")
    for word in audit.targets:
        counts = [cell.target_counts[word] for cell in audit.merged.values()]
        covered = [
            decade
            for decade, cell in audit.merged.items()
            if cell.target_counts[word] >= min_occurrences
        ]
        transitions = audit.reliable_transitions[word]
        reliable = f"{len(transitions)}/{transitions_total}"
        coverage_table.add_row(
            word,
            reliable if transitions else f"[red]{reliable}[/red]",
            format_decade_ranges(covered) or "—",
            f"{min(counts, default=0):,}–{max(counts, default=0):,}",
        )
    console.print(coverage_table)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(audit.to_json(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        console.print(f"[green]Wrote {output}[/green]")


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
