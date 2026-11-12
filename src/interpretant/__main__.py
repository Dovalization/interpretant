"""CLI entry point for interpretant."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

console = Console()


@click.group()
@click.version_option(package_name="interpretant")
def cli() -> None:
    """interpretant — diachronic semantic drift tracker for academic corpora."""


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------


@cli.group()
def corpus() -> None:
    """Corpus loading and preprocessing commands."""


@corpus.group("books")
def corpus_books() -> None:
    """Books corpus commands (PDF-extracted texts)."""


@corpus_books.command("ingest")
@click.option(
    "--books-dir",
    type=click.Path(path_type=Path),
    default=Path("data/external/books"),
    show_default=True,
)  # noqa: E501
@click.option(
    "--output-dir", type=click.Path(path_type=Path), help="Override processed output dir."
)  # noqa: E501
@click.option("--start", default=1820, show_default=True, help="First decade (inclusive).")
@click.option("--end", default=2030, show_default=True, help="Last decade (exclusive).")
@click.option(
    "--author-cap",
    default=250_000,
    show_default=True,
    help="Tokens per author per decade; larger authors contribute a seeded sample of chunks.",
)
@click.option(
    "--leave-out-largest-author",
    is_flag=True,
    help="Drop each decade's largest author (leave-one-author-out check; use --output-dir).",
)
def corpus_books_ingest(  # noqa: PLR0913
    books_dir: Path,
    output_dir: Path | None,
    start: int,
    end: int,
    author_cap: int,
    leave_out_largest_author: bool,
) -> None:
    """Preprocess extracted book texts into decade-sliced token files, capping authors."""
    from interpretant.corpus.books import BooksSource, ingest_books  # noqa: PLC0415

    texts_dir = output_dir or Path("data/processed/books/texts")
    console.print("[bold]Ingesting books corpus[/bold]")
    console.print(f"  books dir  : {books_dir}")
    console.print(f"  output dir : {texts_dir}")
    console.print(f"  decades    : {start}–{end - 1}")
    console.print(f"  author cap : {author_cap:,} tokens")
    ingested = ingest_books(
        BooksSource(books_dir),
        texts_dir,
        range(start, end, 10),
        author_cap,
        leave_out_largest_author,
    )
    table = Table(title="Books ingest results", show_lines=True)
    table.add_column("Decade", style="cyan")
    table.add_column("Books", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Left out" if leave_out_largest_author else "Largest author")
    for decade in ingested:
        table.add_row(
            str(decade.decade), str(decade.books), f"{decade.tokens:,}", decade.largest_author
        )
    console.print(table)
    console.print("[green]Done.[/green]")


@corpus_books.command("validate")
@click.option(
    "--books-dir",
    type=click.Path(path_type=Path),
    default=Path("data/external/books"),
    show_default=True,
)  # noqa: E501
def corpus_books_validate(books_dir: Path) -> None:
    """Check all manifest entries are present on disk."""
    from interpretant.corpus.books import BooksSource

    source = BooksSource(books_dir=books_dir)
    present, missing = source.validate()

    table = Table(title="Books manifest validation", show_lines=True)
    table.add_column("Status", style="green")
    table.add_column("Author")
    table.add_column("Title")
    table.add_column("Decade", style="cyan")
    table.add_column("File")
    for entry in present:
        table.add_row(
            "✓",
            str(entry["author"]),
            str(entry["title"]),
            str(entry["decade"]),
            str(entry["filename"]),
        )  # noqa: E501
    for entry in missing:
        table.add_row(
            "[red]✗[/red]",
            str(entry["author"]),
            str(entry["title"]),
            str(entry["decade"]),
            str(entry["filename"]),
        )  # noqa: E501
    console.print(table)
    console.print(f"[green]{len(present)} present[/green]  [red]{len(missing)} missing[/red]")


@corpus_books.command("stats")
@click.option(
    "--books-dir",
    type=click.Path(path_type=Path),
    default=Path("data/external/books"),
    show_default=True,
)  # noqa: E501
def corpus_books_stats(books_dir: Path) -> None:
    """Show book counts per decade and manifest status."""
    from interpretant.corpus.books import BooksSource

    source = BooksSource(books_dir=books_dir)
    stats = source.stats()

    decade_table = Table(title="Books per decade (present files)", show_lines=True)
    decade_table.add_column("Decade", style="cyan")
    decade_table.add_column("Books (present/total)", justify="right")
    decade_table.add_column("Words", justify="right")
    manifest_entries = stats["manifest_entries"]
    total_present = stats["total_books"]
    per_decade: dict[str, dict[str, int]] = stats["per_decade"]  # type: ignore[assignment]
    for label, counts in sorted(per_decade.items()):
        decade_table.add_row(label, str(counts["books"]), f"{counts['words']:,}")
    decade_table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{total_present}/{manifest_entries}[/bold]",
        "",
    )
    console.print(decade_table)

    field_table = Table(title="Books per field (all manifest entries)", show_lines=True)
    field_table.add_column("Field", style="magenta")
    field_table.add_column("Books", justify="right")
    per_field: dict[str, int] = stats["per_field"]  # type: ignore[assignment]
    for field, count in sorted(per_field.items()):
        field_table.add_row(field, str(count))
    console.print(field_table)
    console.print(f"Manifest entries: {manifest_entries}  Missing: {stats['missing']}")


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
def corpus_openalex_download(  # noqa: PLR0913
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
    client = OpenAlexClient(OpenAlexSettings().openalex_api_key)  # type: ignore[call-arg]
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
)  # noqa: E501
def corpus_openalex_sources(
    raw_dir: Path, fields: tuple[str, ...], decades: tuple[int, ...]
) -> None:
    """Look up the primary source (venue) of every work in the given cells (needs the API key)."""
    from interpretant.corpus.openalex import (  # noqa: PLC0415
        OpenAlexClient,
        OpenAlexSettings,
        fetch_sources,
    )

    client = OpenAlexClient(OpenAlexSettings().openalex_api_key)  # type: ignore[call-arg]
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
    from interpretant.corpus.openalex import ingest_fields  # noqa: PLC0415

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
    default=("books", "openalex"),
    show_default=True,
    help="Processed sources to include (the main corpus is books and the OpenAlex grid).",
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
def corpus_phrases_learn(  # noqa: PLR0913
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
    from interpretant.corpus.audit import load_target_words  # noqa: PLC0415
    from interpretant.corpus.files import source_texts  # noqa: PLC0415
    from interpretant.corpus.phrases import (  # noqa: PLC0415
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
    default=("books", "openalex"),
    show_default=True,
    help="Processed sources to include (the main corpus is books and the OpenAlex grid).",
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
    from interpretant.corpus.files import rewrite_texts, source_texts  # noqa: PLC0415
    from interpretant.corpus.phrases import PhraseModel  # noqa: PLC0415

    model = PhraseModel.load(model_path)
    paths = source_texts(processed_dir, sources)
    written = rewrite_texts(paths, processed_dir, output_dir, model.apply)
    console.print(f"[green]{len(written)} files[/green] → {output_dir}")


@corpus.group("lemmas")
def corpus_lemmas() -> None:
    """Lemma map: collapse plurals and British spellings before phrase detection."""


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
    default=("books", "openalex", "openalex_full"),
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
    "--output",
    type=click.Path(path_type=Path),
    default=Path("data/processed/lemmas.tsv"),
    show_default=True,
)
@click.option("--review", default=150, show_default=True, help="Most frequent mappings to print.")
def corpus_lemmas_learn(  # noqa: PLR0913
    processed_dir: Path,
    sources: tuple[str, ...],
    words_file: Path,
    overrides_path: Path,
    output: Path,
    review: int,
) -> None:
    """Learn the type-level lemma map from every processed file and write it for review."""
    from collections import Counter  # noqa: PLC0415

    from interpretant.corpus.audit import load_target_words  # noqa: PLC0415
    from interpretant.corpus.files import source_texts  # noqa: PLC0415
    from interpretant.corpus.lemmas import (  # noqa: PLC0415
        count_types,
        learn_lemma_map,
        read_overrides,
    )

    paths = source_texts(processed_dir, sources)
    console.print(f"  counting types in {len(paths)} files")
    counts = count_types(tqdm(paths, unit="file"))
    protected = set(load_target_words(words_file))
    lemma_map = learn_lemma_map(counts, protected, read_overrides(overrides_path))
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
    default=("books", "openalex", "openalex_full"),
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
    from interpretant.corpus.files import rewrite_texts, source_texts  # noqa: PLC0415
    from interpretant.corpus.lemmas import LemmaMap  # noqa: PLC0415

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
)  # noqa: E501
@click.option(
    "--words-file",
    type=click.Path(path_type=Path),
    default=Path("data/words_of_interest.txt"),
    show_default=True,
)  # noqa: E501
@click.option(
    "--books-dir",
    type=click.Path(path_type=Path),
    default=Path("data/external/books"),
    show_default=True,
)  # noqa: E501
@click.option(
    "--min-occurrences",
    default=150,
    show_default=True,
    help="Occurrences a word needs in both decades for a transition to count as reliable.",
)  # noqa: E501
@click.option("--output", type=click.Path(path_type=Path), help="Also write the audit as JSON.")
def corpus_audit(
    processed_dir: Path,
    words_file: Path,
    books_dir: Path,
    min_occurrences: int,
    output: Path | None,
) -> None:
    """Report cell volume, target-word coverage and books-manifest risks."""
    import json  # noqa: PLC0415

    from interpretant.corpus.audit import (  # noqa: PLC0415
        audit_corpus,
        format_decade_ranges,
        load_target_words,
    )
    from interpretant.corpus.books import BooksSource  # noqa: PLC0415

    audit = audit_corpus(
        processed_dir, load_target_words(words_file), BooksSource(books_dir), min_occurrences
    )
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

    concentration_table = Table(title="Books: largest author per decade", show_lines=False)
    concentration_table.add_column("Decade", style="cyan")
    concentration_table.add_column("Books", justify="right")
    concentration_table.add_column("Words (M)", justify="right")
    concentration_table.add_column("Top author")
    concentration_table.add_column("Share", justify="right")
    for summary in audit.author_concentration:
        share = f"{summary.top_share:.0%}"
        concentration_table.add_row(
            summary.decade,
            str(summary.books),
            f"{summary.words / 1e6:.2f}",
            summary.top_author,
            f"[red]{share}[/red]" if summary.top_share >= 0.25 else share,
        )
    console.print(concentration_table)

    flag_table = Table(
        title=f"Books: dating risks ({len(audit.book_flags)} entries)", show_lines=False
    )
    flag_table.add_column("Decade", style="cyan")
    flag_table.add_column("Author")
    flag_table.add_column("Title")
    flag_table.add_column("Reasons")
    for flag in audit.book_flags:
        flag_table.add_row(flag.decade, flag.author, flag.title[:50], "; ".join(flag.reasons))
    console.print(flag_table)
    console.print(f"Manifest entries without a file on disk: {len(audit.missing_books)}")

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
    import logging  # noqa: PLC0415
    import sys  # noqa: PLC0415

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
    default=("books", "openalex"),
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
def twec_grid(  # noqa: PLR0913
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
    from interpretant.embedding.grid import discover_cells, train_twec_grid  # noqa: PLC0415
    from interpretant.embedding.twec import TWECTrainer  # noqa: PLC0415

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
    default=("books", "openalex", "openalex_full"),
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
    from interpretant.embedding.grid import build_ppmi_grid, discover_cells  # noqa: PLC0415

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


@drift.command("perplexity")
@click.option(
    "--texts-dir",
    type=click.Path(path_type=Path),
    default=Path("data/phrased/openalex/texts"),
    show_default=True,
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("data/drift/cross_perplexity.parquet"),
    show_default=True,
)
@click.option("--seed", "seeds", multiple=True, default=(42,), show_default=True)
@click.option(
    "--order",
    default=2,
    show_default=True,
    type=click.Choice([1, 2]),
    help="2: Kneser-Ney bigram; 1: add-one unigram, the lexical-composition control.",
)  # noqa: E501
@click.option(
    "--fixed-budget",
    default=0,
    show_default=True,
    help="Train every field in every decade on exactly this many tokens (0: per-decade rule).",
)
@click.option(
    "--drop-shared", is_flag=True, help="Remove abstracts that appear in more than one cell."
)
def drift_perplexity(  # noqa: PLR0913
    texts_dir: Path,
    output: Path,
    seeds: tuple[int, ...],
    order: int,
    fixed_budget: int,
    drop_shared: bool,
) -> None:
    """Cross-entropy of every field's held-out abstracts under every field's model, per decade.

    Method and predictions: docs/predictions_cross_perplexity.md.
    """
    from interpretant.drift.perplexity import cross_entropy_runs  # noqa: PLC0415

    result = cross_entropy_runs(
        texts_dir, seeds, fixed_budget or None, drop_shared, order, console.print
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    console.print(f"[green]{len(result):,} rows[/green] → {output}")


@drift.command("decompose")
@click.option(
    "--texts-dir",
    type=click.Path(path_type=Path),
    default=Path("data/phrased/openalex/texts"),
    show_default=True,
)
@click.option("--early", default=1990, show_default=True)
@click.option("--late", default=2020, show_default=True)
@click.option("--seed", "seeds", multiple=True, default=(1, 2, 3, 4, 5), show_default=True)
@click.option("--order", default=2, show_default=True, type=click.Choice([1, 2]))
@click.option("--fixed-budget", default=200_000, show_default=True)
@click.option("--text", "texts", multiple=True, help="Text fields to decompose; default all.")
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("data/drift/cross_perplexity_decomposition.parquet"),
    show_default=True,
)
def drift_decompose(  # noqa: PLR0913
    texts_dir: Path,
    early: int,
    late: int,
    seeds: tuple[int, ...],
    order: int,
    fixed_budget: int,
    texts: tuple[str, ...],
    output: Path,
) -> None:
    """Split every pair's fall in excess cross-entropy between two decades into a word-mix
    effect (the text's word frequencies changed) and a rate effect (the model's per-word
    skill changed). A fall that is all mix says the text moved; the model learned nothing."""
    from interpretant.drift.perplexity import decompose_runs  # noqa: PLC0415

    try:
        result = decompose_runs(texts_dir, early, late, seeds, fixed_budget, order, texts or None)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    summary = result.groupby(["model_field", "text_field"])[["fall", "mix", "rate"]].mean()
    console.print(summary.round(3).to_string())
    console.print(f"[green]{len(result):,} rows[/green] → {output}")


@drift.command("separability")
@click.option(
    "--texts-dir",
    type=click.Path(path_type=Path),
    default=Path("data/phrased/openalex/texts"),
    show_default=True,
)
@click.option(
    "--decade",
    "decades",
    multiple=True,
    type=int,
    default=(1980, 1990, 2000, 2010, 2020),
    show_default=True,
)
@click.option("--seed", "seeds", multiple=True, default=(1, 2, 3, 4, 5), show_default=True)
@click.option("--per-field", default=2_000, show_default=True, help="Abstracts per field.")
@click.option(
    "--remove-terms",
    "terms_file",
    type=click.Path(exists=True, path_type=Path),
    help="Strip these terms (one per line) from the samples first (round V).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/drift"),
    show_default=True,
)
def drift_separability(  # noqa: PLR0913
    texts_dir: Path,
    decades: tuple[int, ...],
    seeds: tuple[int, ...],
    per_field: int,
    terms_file: Path | None,
    output_dir: Path,
) -> None:
    """How well a classifier tells every two fields apart, per decade, and where a classifier
    over all fields sends each field's abstracts (round S)."""
    from interpretant.corpus.audit import load_target_words  # noqa: PLC0415
    from interpretant.drift.separability import separability_runs  # noqa: PLC0415

    terms = frozenset(load_target_words(terms_file)) if terms_file else frozenset()
    pairs, assignments = separability_runs(
        texts_dir, decades, seeds, per_field=per_field, removed_terms=terms, log=console.print
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(output_dir / "separability_pairs.parquet", index=False)
    assignments.to_parquet(output_dir / "separability_assignments.parquet", index=False)
    console.print(f"[green]{len(pairs):,} pair rows, {len(assignments):,} assignment rows[/green]")


@drift.command("heldout")
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
    default=("books", "openalex", "openalex_full"),
    show_default=True,
)
@click.option("--seed", "seeds", multiple=True, default=(1, 2, 3), show_default=True)
@click.option("--workers", default=4, show_default=True)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("data/drift/heldout.parquet"),
    show_default=True,
)
def drift_heldout(
    input_dir: Path, layers: tuple[str, ...], seeds: tuple[int, ...], workers: int, output: Path
) -> None:
    """Held-out minus training cross-entropy of every cell's bigram model (round M)."""
    from interpretant.drift.heldout import heldout_runs  # noqa: PLC0415

    table = heldout_runs(input_dir, layers, seeds, workers, console.print)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output, index=False)
    console.print(f"[green]{len(table):,} rows[/green] → {output}")


@drift.command("direction")
@click.option(
    "--texts-dir",
    type=click.Path(path_type=Path),
    default=Path("data/phrased/openalex/texts"),
    show_default=True,
)
@click.option(
    "--train-decade",
    "train_decades",
    multiple=True,
    type=int,
    default=(1980, 1990),
    show_default=True,
)
@click.option(
    "--test-decade",
    "test_decades",
    multiple=True,
    type=int,
    default=(1990, 2000, 2010, 2020),
    show_default=True,
)
@click.option("--seed", "seeds", multiple=True, default=(1, 2, 3, 4, 5), show_default=True)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("data/drift/direction_assignments.parquet"),
    show_default=True,
)
def drift_direction(  # noqa: PLR0913
    texts_dir: Path,
    train_decades: tuple[int, ...],
    test_decades: tuple[int, ...],
    seeds: tuple[int, ...],
    output: Path,
) -> None:
    """Where each decade's classifier sends later decades' abstracts (round N)."""
    from interpretant.drift.direction import assignment_runs  # noqa: PLC0415

    table = assignment_runs(texts_dir, train_decades, test_decades, seeds, log=console.print)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output, index=False)
    console.print(f"[green]{len(table):,} rows[/green] → {output}")


@drift.command("senses")
@click.option(
    "--raw-dir",
    type=click.Path(path_type=Path),
    default=Path("data/raw/openalex"),
    show_default=True,
)
@click.option("--word", "words", multiple=True, required=True)
@click.option("--field", "fields", multiple=True)
@click.option(
    "--decade", "decades", multiple=True, type=int, default=(1990, 2020), show_default=True
)
@click.option("--seed", "seeds", multiple=True, default=(1, 2, 3, 4, 5), show_default=True)
@click.option(
    "--encoder", "model_name", default="allenai/scibert_scivocab_uncased", show_default=True
)
@click.option("--cap", type=int, default=500, show_default=True)
@click.option(
    "--cell",
    "cells",
    multiple=True,
    help="field:decade; repeat it. Replaces the --field × --decade grid.",
)
@click.option("--senses", type=int, default=None, help="Fixed number of senses (round C′).")
@click.option("--replace", is_flag=True, help="Draw occurrences with replacement (round C′).")
@click.option("--plural", is_flag=True, help="Match the word's regular plural too (round C′).")
@click.option("--quality", is_flag=True, help="Round C′'s quality rules.")
@click.option("--output", type=click.Path(path_type=Path), required=True)
def drift_senses(  # noqa: PLR0913
    raw_dir: Path,
    words: tuple[str, ...],
    fields: tuple[str, ...],
    decades: tuple[int, ...],
    seeds: tuple[int, ...],
    model_name: str,
    cap: int,
    cells: tuple[str, ...],
    senses: int | None,
    replace: bool,
    plural: bool,
    quality: bool,
    output: Path,
) -> None:
    """Senses of words by field and decade, from one contextual encoder (round C)."""
    from interpretant.drift.senses import (  # noqa: PLC0415
        SENSE_RANGE,
        Sampling,
        TransformerEncoder,
        cased_abstracts,
        drop_shared,
        sense_runs,
    )

    chosen = (
        [(field, int(decade)) for field, decade in (cell.split(":") for cell in cells)]
        if cells
        else [(field, decade) for field in fields for decade in decades]
    )
    abstracts = drop_shared(
        {(field, decade): cased_abstracts(raw_dir, field, decade) for field, decade in chosen}
    )
    encoder = TransformerEncoder(model_name)
    sampling = Sampling(cap, replace, plural, quality)
    sense_range = (senses,) if senses else SENSE_RANGE
    table = sense_runs(abstracts, words, seeds, encoder, sampling, sense_range, log=console.print)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output, index=False)
    console.print(f"[green]{len(table):,} rows[/green] → {output}")


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
def drift_grid(  # noqa: PLR0913
    model_dir: Path,
    output_dir: Path,
    words_file: Path,
    controls_file: Path,
    measure_min: int,
    neighbour_min: int,
    neighbours: int,
) -> None:
    """Measure drift on the grid: within-field trajectories and cross-field divergence.

    Writes within_field.parquet (each field's consecutive decades, including the books and
    the full cells) and cross_field.parquet (every two OpenAlex fields in the same decade),
    one row per word measured in a pair, with percentile ranks against the pair's
    whole-vocabulary noise floor and a candidate flag.
    """
    from interpretant.corpus.audit import load_target_words  # noqa: PLC0415
    from interpretant.drift.grid import measure_grid  # noqa: PLC0415

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
        table.to_parquet(output_dir / f"{name}.parquet", index=False)
        pairs = table.groupby(["layer", "field_a", "decade_a", "field_b", "decade_b"]).ngroups
        control_rate = table.loc[table["is_control"], "candidate"].mean()
        console.print(
            f"  {name}: {pairs} pairs, {len(table):,} rows, "
            f"{int(table.loc[table['is_target'], 'candidate'].sum())} target candidates, "
            f"controls flagged {control_rate:.1%} (expected under ~5%)"
        )


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


@cli.group()
def export() -> None:
    """Exports of the models and tables for other surfaces."""


@export.command("web")
@click.option(
    "--model-dir",
    type=click.Path(path_type=Path),
    default=Path("models/twec_grid"),
    show_default=True,
)  # noqa: E501
@click.option(
    "--ppmi-dir",
    type=click.Path(path_type=Path),
    default=Path("models/ppmi_grid"),
    show_default=True,
)  # noqa: E501
@click.option(
    "--drift-dir", type=click.Path(path_type=Path), default=Path("data/drift"), show_default=True
)  # noqa: E501
@click.option(
    "--audit",
    "audit_path",
    type=click.Path(path_type=Path),
    default=Path("data/phrased/audit.json"),
    show_default=True,
)  # noqa: E501
@click.option(
    "--words-file",
    type=click.Path(path_type=Path),
    default=Path("data/words_of_interest.txt"),
    show_default=True,
)  # noqa: E501
@click.option(
    "--output-dir", type=click.Path(path_type=Path), default=Path("web/data"), show_default=True
)  # noqa: E501
def export_web_command(  # noqa: PLR0913
    model_dir: Path,
    ppmi_dir: Path,
    drift_dir: Path,
    audit_path: Path,
    words_file: Path,
    output_dir: Path,
) -> None:
    """Write the JSON the web explorer (web/) builds from: neighbours, drift rows, counts,
    cross-perplexity and the prediction verdicts."""
    from interpretant.export import PREDICTIONS_PATHS, export_web  # noqa: PLC0415

    export_web(
        model_dir=model_dir,
        ppmi_dir=ppmi_dir,
        drift_dir=drift_dir,
        audit_path=audit_path,
        predictions_paths=PREDICTIONS_PATHS,
        words_file=words_file,
        output_dir=output_dir,
        log=console.print,
    )


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


@cli.command("figures")
@click.option(
    "--model-dir",
    type=click.Path(path_type=Path),
    default=Path("models/twec_grid"),
    show_default=True,
)
@click.option(
    "--drift-dir", type=click.Path(path_type=Path), default=Path("data/drift"), show_default=True
)
@click.option(
    "--audit",
    "audit_path",
    type=click.Path(path_type=Path),
    default=Path("data/phrased/audit.json"),
    show_default=True,
    help="Output of `corpus audit --output`; the corpus grid figure reads its cell counts.",
)
@click.option(
    "--output-dir", type=click.Path(path_type=Path), default=Path("figures"), show_default=True
)
def figures(model_dir: Path, drift_dir: Path, audit_path: Path, output_dir: Path) -> None:
    """Draw the README figures from the trained grid models and the drift tables."""
    from interpretant.visualization.figures import readme_figures  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)
    drawn = readme_figures(model_dir, drift_dir, audit_path)
    for name, figure in drawn.items():
        path = output_dir / f"{name}.png"
        figure.savefig(path, dpi=150, facecolor=figure.get_facecolor())
        console.print(f"  wrote {path}")


if __name__ == "__main__":
    cli()
