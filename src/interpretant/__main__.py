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
    from interpretant.corpus.books import BooksSource
    from interpretant.corpus.caps import cap_author_tokens, largest_author

    source = BooksSource(books_dir=books_dir)
    texts_dir = output_dir or Path("data/processed/books/texts")
    texts_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Ingesting books corpus[/bold]")
    console.print(f"  books dir  : {books_dir}")
    console.print(f"  output dir : {texts_dir}")
    console.print(f"  decades    : {start}–{end - 1}")
    console.print(f"  author cap : {author_cap:,} tokens")

    table = Table(title="Books ingest results", show_lines=True)
    table.add_column("Decade", style="cyan")
    table.add_column("Books", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Left out" if leave_out_largest_author else "Largest author")

    for decade in range(start, end, 10):
        books = source.decade_documents(decade)
        if not books:
            continue
        top = largest_author(books)
        if leave_out_largest_author:
            books = [(author, text) for author, text in books if author != top]
        documents = cap_author_tokens(books, author_cap)
        (texts_dir / f"books_{decade}.txt").write_text("\n".join(documents), encoding="utf-8")
        tokens = sum(len(document.split()) for document in documents)
        table.add_row(str(decade), str(len(books)), f"{tokens:,}", top)

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
def corpus_openalex_download(  # noqa: PLR0913
    raw_dir: Path,
    fields: tuple[str, ...],
    start: int,
    end: int,
    sample_size: int,
    seed: int,
    full: bool,
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
                works = download_cell(client, raw_dir, field, decade, sample_size, seed, full)
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
    from interpretant.corpus.openalex import FIELDS, OpenAlexSource

    output_dir.mkdir(parents=True, exist_ok=True)
    table = Table(title="OpenAlex ingest (kept −dropped per cell)", show_lines=False)
    table.add_column("Field", style="cyan")
    decades = list(range(start, end, 10))
    for decade in decades:
        table.add_column(str(decade), justify="right")
    for field in FIELDS:
        source = OpenAlexSource(raw_dir, field)
        counts = []
        for decade in decades:
            cell = source.cell(decade)
            if cell.documents:
                (output_dir / f"{field}_{decade}.txt").write_text(
                    "\n".join(cell.documents), "utf-8"
                )
            dropped = (
                cell.duplicates
                + cell.catalogue_records
                + cell.web_pages
                + cell.non_english
                + cell.author_capped
            )
            counts.append(
                f"{len(cell.documents):,} −{dropped:,}" if dropped else f"{len(cell.documents):,}"
            )
        table.add_row(field, *counts)
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
    from interpretant.corpus.audit import load_target_words
    from interpretant.corpus.phrases import learn_phrases, sample_chunks, split_targets

    paths = sorted(
        path for source in sources for path in processed_dir.glob(f"{source}/texts/*.txt")
    )
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
    from interpretant.corpus.phrases import PhraseModel

    model = PhraseModel.load(model_path)
    paths = sorted(
        path for source in sources for path in processed_dir.glob(f"{source}/texts/*.txt")
    )
    for path in paths:
        target = output_dir / path.relative_to(processed_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        with path.open(encoding="utf-8") as source, target.open("w", encoding="utf-8") as out:
            for line in source:
                out.write(model.apply(line) + "\n")
        console.print(f"  {target}")


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
    from interpretant.corpus.lemmas import (  # noqa: PLC0415
        count_types,
        learn_lemma_map,
        read_overrides,
    )

    paths = sorted(
        path for source in sources for path in processed_dir.glob(f"{source}/texts/*.txt")
    )
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
    from interpretant.corpus.lemmas import LemmaMap  # noqa: PLC0415

    lemma_map = LemmaMap.load(map_path)
    paths = sorted(
        path for source in sources for path in processed_dir.glob(f"{source}/texts/*.txt")
    )
    for path in tqdm(paths, unit="file"):
        target = output_dir / path.relative_to(processed_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        with path.open(encoding="utf-8") as source, target.open("w", encoding="utf-8") as out:
            for line in source:
                out.write(lemma_map.apply(line) + "\n")
    console.print(f"[green]{len(paths)} files[/green] → {output_dir}")


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
    import json
    from dataclasses import asdict

    from interpretant.corpus.audit import (
        author_concentration,
        flag_manifest_entries,
        format_decade_ranges,
        load_target_words,
        merge_by_decade,
        reliable_transitions,
        scan_processed,
    )
    from interpretant.corpus.books import BooksSource

    targets = load_target_words(words_file)
    cells = scan_processed(processed_dir, set(targets))
    merged = merge_by_decade(cells)
    sources = sorted({cell.source for cell in cells})
    tokens_by_cell = {(cell.source, cell.decade): cell.tokens for cell in cells}

    volume_table = Table(title="Tokens per cell (millions)", show_lines=False)
    volume_table.add_column("Decade", style="cyan")
    for source in sources:
        volume_table.add_column(source, justify="right")
    volume_table.add_column("merged", justify="right", style="bold")
    for decade, total in merged.items():
        cells_row = [
            f"{tokens_by_cell[(source, decade)] / 1e6:.2f}"
            if (source, decade) in tokens_by_cell
            else "—"
            for source in sources
        ]
        volume_table.add_row(str(decade), *cells_row, f"{total.tokens / 1e6:.2f}")
    console.print(volume_table)

    transitions_total = max(len(merged) - 1, 0)
    transitions_by_word = {
        word: reliable_transitions(merged, word, min_occurrences) for word in targets
    }
    coverage_table = Table(
        title=f"Target coverage (merged, ≥{min_occurrences} occurrences)", show_lines=False
    )
    coverage_table.add_column("Word", style="magenta")
    coverage_table.add_column("Reliable transitions", justify="right")
    coverage_table.add_column("Decades at threshold")
    coverage_table.add_column("Min–max count", justify="right")
    for word in targets:
        counts = [cell.target_counts[word] for cell in merged.values()]
        covered = [d for d, cell in merged.items() if cell.target_counts[word] >= min_occurrences]
        reliable = f"{len(transitions_by_word[word])}/{transitions_total}"
        coverage_table.add_row(
            word,
            reliable if transitions_by_word[word] else f"[red]{reliable}[/red]",
            format_decade_ranges(covered) or "—",
            f"{min(counts, default=0):,}–{max(counts, default=0):,}",
        )
    console.print(coverage_table)

    books = BooksSource(books_dir=books_dir)
    present, missing = books.validate()
    book_words = [
        (
            str(entry.get("decade", "")),
            str(entry.get("author", "")),
            len(books.entry_path(entry).read_text(encoding="utf-8", errors="replace").split()),
        )
        for entry in present
        if str(entry.get("decade", "")).endswith("0s")
    ]
    concentration = author_concentration(book_words)
    concentration_table = Table(title="Books: largest author per decade", show_lines=False)
    concentration_table.add_column("Decade", style="cyan")
    concentration_table.add_column("Books", justify="right")
    concentration_table.add_column("Words (M)", justify="right")
    concentration_table.add_column("Top author")
    concentration_table.add_column("Share", justify="right")
    for summary in concentration:
        share = f"{summary.top_share:.0%}"
        concentration_table.add_row(
            summary.decade,
            str(summary.books),
            f"{summary.words / 1e6:.2f}",
            summary.top_author,
            f"[red]{share}[/red]" if summary.top_share >= 0.25 else share,
        )
    console.print(concentration_table)

    flags = sorted(flag_manifest_entries(books.manifest()), key=lambda flag: flag.decade)
    flag_table = Table(title=f"Books: dating risks ({len(flags)} entries)", show_lines=False)
    flag_table.add_column("Decade", style="cyan")
    flag_table.add_column("Author")
    flag_table.add_column("Title")
    flag_table.add_column("Reasons")
    for flag in flags:
        flag_table.add_row(flag.decade, flag.author, flag.title[:50], "; ".join(flag.reasons))
    console.print(flag_table)
    console.print(f"Manifest entries without a file on disk: {len(missing)}")

    if output is not None:
        report = {
            "min_occurrences": min_occurrences,
            "cells": [
                {**asdict(cell), "target_counts": dict(cell.target_counts)} for cell in cells
            ],
            "reliable_transitions": transitions_by_word,
            "author_concentration": [asdict(summary) for summary in concentration],
            "book_flags": [asdict(flag) for flag in flags],
            "missing_books": [str(books.entry_path(entry)) for entry in missing],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
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


@twec.command("train")
@click.option(
    "--corpus",
    "corpora",
    multiple=True,
    default=["books", "arxiv", "acl", "pubmed"],
    show_default=True,
    help="Corpus sources (repeatable).",
)  # noqa: E501
@click.option(
    "--input-dir",
    type=click.Path(path_type=Path),
    default=Path("data/processed"),
    show_default=True,
    help="Root of processed text dirs.",
)  # noqa: E501
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("models/twec"),
    show_default=True,
    help="Output dir for .kv files and compass.",
)  # noqa: E501
@click.option("--start", default=1820, show_default=True, help="First decade to train (inclusive).")
@click.option("--end", default=2030, show_default=True, help="Last decade (exclusive).")
@click.option("--vector-size", default=300, show_default=True, help="Embedding dimensionality.")
@click.option("--window", default=10, show_default=True, help="Context window size.")
@click.option(
    "--min-count", default=5, show_default=True, help="Minimum word frequency (applied to compass)."
)  # noqa: E501
@click.option("--workers", default=4, show_default=True, help="Parallel training workers.")
@click.option("--epochs", default=5, show_default=True, help="Training epochs.")
@click.option(
    "--max-docs-per-decade",
    default=20_000,
    show_default=True,
    help="Cap docs streamed per decade file (0 = no cap). Reduces PubMed training time.",
)  # noqa: E501
@click.option(
    "--load-compass",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to a saved compass.model to skip compass training (e.g. models/twec/compass.model).",
)  # noqa: E501
@click.option(
    "--source-cap",
    default=None,
    help="Per-source document cap as 'source:N,...' (e.g. 'pubmed:10000,arxiv:5000').",
)  # noqa: E501
def twec_train(  # noqa: PLR0913
    corpora: tuple[str, ...],
    input_dir: Path,
    output_dir: Path,
    start: int,
    end: int,
    vector_size: int,
    window: int,
    min_count: int,
    workers: int,
    epochs: int,
    max_docs_per_decade: int,
    load_compass: Path | None,
    source_cap: str | None,
) -> None:
    """Train TWEC embeddings: compass on all decades, then per-decade models.

    Outputs one .kv file per decade in the compass's space
    without a separate alignment step.
    """
    from interpretant.embedding.twec import TWECTrainer, _EpochProgressCallback  # noqa: PLC0415

    _configure_gensim_logging()

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Training TWEC embeddings[/bold]")
    console.print(f"  corpora  : {', '.join(corpora)}")
    console.print(f"  decades  : {start}–{end - 1} (step 10)")
    console.print(f"  output   : {output_dir}")
    console.print(
        f"  dims     : {vector_size}  window: {window}  min_count: {min_count}  epochs: {epochs}"
    )  # noqa: E501
    if max_docs_per_decade:
        console.print(f"  max docs : {max_docs_per_decade:,} per file")
    if source_cap:
        console.print(f"  source caps : {source_cap}")

    trainer = TWECTrainer(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
    )

    # --- Phase 1: collect file paths per decade (no data loaded into RAM) ---
    console.print("\n[bold]Phase 1:[/bold] Scanning corpus files …")
    all_paths: list[Path] = []
    decade_paths: dict[int, list[Path]] = {}

    for decade in range(start, end, 10):
        decade_paths[decade] = []
        for corpus_name in corpora:
            candidate = input_dir / corpus_name / "texts" / f"{corpus_name}_{decade}.txt"
            if not candidate.exists():
                continue
            decade_paths[decade].append(candidate)
            all_paths.append(candidate)

    # Drop empty decades
    decade_paths = {d: paths for d, paths in decade_paths.items() if paths}

    total_files = sum(len(p) for p in decade_paths.values())
    console.print(
        f"  {total_files} file(s) across {len(decade_paths)} decades "
        f"(streaming — no data loaded yet)"
    )

    path_caps: dict[Path, int] = {}
    if source_cap:
        cap_map: dict[str, int] = {}
        for entry in source_cap.split(","):
            source_name, n = entry.strip().split(":")
            cap_map[source_name.strip()] = int(n)
        for path in all_paths:
            source_prefix = "_".join(path.stem.split("_")[:-1])
            if source_prefix in cap_map:
                path_caps[path] = cap_map[source_prefix]

    _max_docs: int | dict[Path, int] | None = path_caps or (
        max_docs_per_decade if max_docs_per_decade else None
    )

    # --- Phase 2: train or load compass ---
    if load_compass is not None:
        console.print(f"\n[bold]Phase 2:[/bold] Loading compass from {load_compass} …")
        trainer.load_compass(load_compass)
        console.print(f"  compass vocab : [cyan]{trainer.compass_vocab_size:,}[/cyan] words")
    else:
        console.print("\n[bold]Phase 2:[/bold] Training compass …")
        with tqdm(total=epochs, desc="  compass", unit="epoch", leave=True) as bar:
            trainer.train_compass(
                all_paths,
                max_docs=_max_docs,
                extra_callbacks=[_EpochProgressCallback(bar)],
            )
        compass_path = trainer.save_compass(output_dir)
        console.print(f"  compass vocab : [cyan]{trainer.compass_vocab_size:,}[/cyan] words")
        console.print(f"  saved         : {compass_path}")

    # --- Phase 3: train each decade model (streams per-decade files) ---
    console.print("\n[bold]Phase 3:[/bold] Training decade models …")
    table = Table(title="TWEC decade results", show_lines=True)
    table.add_column("Decade", style="cyan")
    table.add_column("Files", justify="right")
    table.add_column("Vocab", justify="right")
    table.add_column("Shared w/ compass", justify="right")
    table.add_column("Output")

    decade_list = sorted(decade_paths)
    for decade in tqdm(decade_list, desc="  decades", unit="decade", leave=True):
        paths = decade_paths[decade]

        with tqdm(total=epochs, desc=f"    {decade}", unit="epoch", leave=False) as bar:
            kv = trainer.train_decade(
                paths,
                decade,
                max_docs=_max_docs,
                extra_callbacks=[_EpochProgressCallback(bar)],
            )
        path = trainer.save_decade_kv(kv, output_dir, decade)

        shared = sum(1 for w in kv.key_to_index if w in trainer.compass.wv.key_to_index)
        table.add_row(
            str(decade),
            str(len(paths)),
            f"{len(kv):,}",
            f"{shared:,}",
            str(path),
        )

    console.print(table)
    console.print(f"\n[green]Done.[/green] Models in [cyan]{output_dir}[/cyan]")


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
    from interpretant.embedding.grid import discover_cells, write_compass_sample  # noqa: PLC0415
    from interpretant.embedding.twec import TWECTrainer, _EpochProgressCallback  # noqa: PLC0415

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

    compass_path = output_dir / "compass.model"
    if compass_path.exists():
        trainer.load_compass(compass_path)
        console.print(f"  compass loaded from {compass_path}")
    else:
        sample_path = output_dir / "compass_sample.txt"
        tokens = write_compass_sample(main_cells, compass_tokens_per_cell, sample_path)
        console.print(f"  compass sample: {tokens:,} tokens → training compass …")
        with tqdm(total=epochs, desc="  compass", unit="epoch", leave=True) as bar:
            trainer.train_compass([sample_path], extra_callbacks=[_EpochProgressCallback(bar)])
        trainer.save_compass(output_dir)
        sample_path.unlink()
    console.print(f"  compass vocab: [cyan]{trainer.compass_vocab_size:,}[/cyan] words")

    for cell in tqdm(main_cells + extra_cells, desc="  cells", unit="cell"):
        target = output_dir / cell.layer / f"{cell.name}_{cell.decade}.kv"
        if target.exists():
            continue
        with tqdm(total=epochs, desc=f"    {cell.label}", unit="epoch", leave=False) as bar:
            kv = trainer.train_decade(
                [cell.path], cell.decade, extra_callbacks=[_EpochProgressCallback(bar)]
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        kv.save(str(target))
    console.print(f"[green]Done.[/green] Models in {output_dir}")


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
    from interpretant.embedding.grid import discover_cells  # noqa: PLC0415
    from interpretant.embedding.ppmi import train_cell  # noqa: PLC0415

    cells = discover_cells(input_dir, layers)
    console.print(
        f"[bold]PPMI-SVD grid[/bold]: {len(cells)} cells, window {window}, {dimensions} dims"
    )
    for cell in tqdm(cells, desc="  cells", unit="cell"):
        target = output_dir / cell.layer / f"{cell.name}_{cell.decade}.kv"
        if target.exists():
            continue
        vectors = train_cell(cell.path, window, dimensions, min_count)
        target.parent.mkdir(parents=True, exist_ok=True)
        vectors.save(str(target))
    console.print(f"[green]Done.[/green] Models in {output_dir}")


# ---------------------------------------------------------------------------
# drift
# ---------------------------------------------------------------------------


@cli.group()
def drift() -> None:
    """Drift computation commands."""


# ---------------------------------------------------------------------------
# pdf
# ---------------------------------------------------------------------------


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
    "--fixed-budget",
    default=0,
    show_default=True,
    help="Train every field in every decade on exactly this many tokens (0: per-decade rule).",
)
@click.option(
    "--drop-shared", is_flag=True, help="Remove abstracts that appear in more than one cell."
)
def drift_perplexity(
    texts_dir: Path, output: Path, seeds: tuple[int, ...], fixed_budget: int, drop_shared: bool
) -> None:
    """Cross-entropy of every field's held-out abstracts under every field's model, per decade.

    Method and predictions: docs/predictions_cross_perplexity.md.
    """
    import pandas as pd  # noqa: PLC0415

    from interpretant.drift.perplexity import add_excess, decade_cross_entropy  # noqa: PLC0415

    by_decade: dict[int, list[Path]] = {}
    for path in sorted(texts_dir.glob("*_*.txt")):
        by_decade.setdefault(int(path.stem.rsplit("_", 1)[1]), []).append(path)
    frames = []
    for seed in seeds:
        for decade, cells in tqdm(sorted(by_decade.items()), desc=f"  seed {seed}", unit="decade"):
            table = decade_cross_entropy(cells, seed, fixed_budget or None, drop_shared)
            table.insert(0, "decade", decade)
            table.insert(0, "seed", seed)
            frames.append(table)
            fields = table.text_field.nunique()
            if fields:
                console.print(
                    f"  seed {seed} {decade}s: {fields} fields, "
                    f"{table.training_tokens.max():,} tokens each"
                )
    result = add_excess(pd.concat(frames, ignore_index=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    console.print(f"[green]{len(result):,} rows[/green] → {output}")


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
    import pandas as pd  # noqa: PLC0415
    from gensim.models import KeyedVectors  # noqa: PLC0415

    from interpretant.corpus.audit import load_target_words  # noqa: PLC0415
    from interpretant.drift.grid import (  # noqa: PLC0415
        add_noise_floor,
        compare_cells,
        cross_field_pairs,
        discover_models,
        within_field_pairs,
    )

    targets = set(load_target_words(words_file))
    controls = set(load_target_words(controls_file))
    loaded: dict[Path, KeyedVectors] = {}

    def vectors(path: Path) -> KeyedVectors:
        if path not in loaded:
            loaded[path] = KeyedVectors.load(str(path))
        return loaded[path]

    tables = {
        "within_field": within_field_pairs(
            discover_models(model_dir, ("books", "openalex", "openalex_full"))
        ),
        "cross_field": cross_field_pairs(discover_models(model_dir, ("openalex",))),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, pairs in tables.items():
        frames = []
        for cell_a, cell_b in tqdm(pairs, desc=f"  {name}", unit="pair"):
            pair = add_noise_floor(
                compare_cells(
                    vectors(cell_a.path),
                    vectors(cell_b.path),
                    measure_min,
                    neighbour_min,
                    neighbours,
                )
            )
            pair.insert(0, "layer", cell_a.layer)
            pair.insert(1, "field_a", cell_a.name)
            pair.insert(2, "decade_a", cell_a.decade)
            pair.insert(3, "field_b", cell_b.name)
            pair.insert(4, "decade_b", cell_b.decade)
            frames.append(pair)
        table = pd.concat(frames, ignore_index=True)
        table["is_target"] = table["word"].isin(targets)
        table["is_control"] = table["word"].isin(controls)
        table.to_parquet(output_dir / f"{name}.parquet", index=False)
        control_rate = table.loc[table["is_control"], "candidate"].mean()
        console.print(
            f"  {name}: {len(pairs)} pairs, {len(table):,} rows, "
            f"{int(table.loc[table['is_target'], 'candidate'].sum())} target candidates, "
            f"controls flagged {control_rate:.1%} (expected under ~5%)"
        )


# ---------------------------------------------------------------------------
# pdf
# ---------------------------------------------------------------------------


@cli.group()
def pdf() -> None:
    """PDF text extraction commands (books corpus)."""


@pdf.command("extract")
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output", type=click.Path(path_type=Path), help="Output .txt path.")
@click.option(
    "--backend",
    default="docling",
    show_default=True,
    type=click.Choice(["docling", "pymupdf", "auto"]),
)  # noqa: E501
@click.option("--author", default="", help="Author metadata for manifest.")
@click.option("--title", default="", help="Title metadata for manifest.")
@click.option("--year", default=0, type=int, help="Publication year.")
@click.option("--decade", default="", help="Decade bucket (e.g. 1960s).")
@click.option("--fields", multiple=True, help="Subject fields (repeatable).")
@click.option(
    "--add-to-manifest", is_flag=True, default=False, help="Append entry to manifest.json."
)  # noqa: E501
def pdf_extract(  # noqa: PLR0913
    pdf_path: Path,
    output: Path | None,
    backend: str,
    author: str,
    title: str,
    year: int,
    decade: str,
    fields: tuple[str, ...],
    add_to_manifest: bool,
) -> None:
    """Extract text from a single PDF."""
    from interpretant.corpus.pdf_extractor import PDFExtractor

    if backend == "docling":
        console.print("[yellow]Note:[/yellow] First run downloads Docling ML models (~500MB).")

    extractor = PDFExtractor(backend=backend)
    console.print(
        f"Extracting [bold]{pdf_path.name}[/bold] using [cyan]{extractor.backend_name}[/cyan] …"
    )  # noqa: E501
    text = extractor.extract(pdf_path)

    dest = output or Path("data/inbox") / f"{pdf_path.stem}.txt"
    extractor.save(text, dest)
    console.print(f"[green]Saved to {dest}[/green]")

    qc = extractor.quality_check(text)
    table = Table(title="Quality check", show_lines=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Word count", str(qc["word_count"]))
    table.add_row("Avg line length", str(qc["avg_line_length"]))
    table.add_row("Short token ratio", str(qc["short_token_ratio"]))
    table.add_row("Suspected OCR errors", str(qc["suspected_ocr_errors"]))
    rec = str(qc["recommendation"])
    colour = "green" if rec == "ok" else "yellow" if rec == "try_docling" else "red"
    table.add_row("Recommendation", f"[{colour}]{rec}[/{colour}]")
    console.print(table)

    if qc["recommendation"] == "try_docling" and backend == "pymupdf":
        console.print(
            "[yellow]Warning:[/yellow] Output may have layout artifacts. Re-run with --backend docling."  # noqa: E501
        )

    if add_to_manifest:
        import json

        manifest_path = Path("data/external/books/manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, object]] = []
        if manifest_path.exists():
            entries = json.loads(manifest_path.read_text())
        entries.append(
            {
                "file": str(dest),
                "author": author,
                "title": title,
                "year": year,
                "decade": decade,
                "fields": list(fields),
                "status": "available",
                "source": "pdf_extraction",
            }
        )
        manifest_path.write_text(json.dumps(entries, indent=2))
        console.print(f"[green]Added to manifest:[/green] {manifest_path}")

    processed_dir = Path("data/inbox/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    pdf_path.rename(processed_dir / pdf_path.name)
    console.print(f"[dim]Moved {pdf_path.name} → {processed_dir}[/dim]")


@pdf.command("batch")
@click.option(
    "--backend",
    default="docling",
    show_default=True,
    type=click.Choice(["docling", "pymupdf", "auto"]),
)  # noqa: E501
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/external/books"),
    show_default=True,
)  # noqa: E501
@click.option("--decade", required=True, help="Decade bucket for all PDFs (e.g. 1960s).")
@click.option(
    "--inbox", type=click.Path(path_type=Path), default=Path("data/inbox/pdfs"), show_default=True
)  # noqa: E501
def pdf_batch(backend: str, output_dir: Path, decade: str, inbox: Path) -> None:
    """Extract all PDFs in the inbox folder."""
    from interpretant.corpus.pdf_extractor import PDFExtractor

    pdfs = sorted(inbox.glob("*.pdf"))
    if not pdfs:
        raise click.ClickException(f"No PDFs found in {inbox}")

    extractor = PDFExtractor(backend=backend)
    dest_dir = output_dir / decade
    dest_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path("data/inbox/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    table = Table(title=f"Batch extraction — {decade}", show_lines=True)
    table.add_column("File")
    table.add_column("Words", justify="right")
    table.add_column("Avg line len", justify="right")
    table.add_column("Recommendation")
    table.add_column("Status")

    for pdf_path in tqdm(pdfs, desc="Extracting PDFs", unit=" files"):
        try:
            text = extractor.extract(pdf_path)
            out_file = dest_dir / f"{pdf_path.stem}.txt"
            extractor.save(text, out_file)
            qc = extractor.quality_check(text)
            rec = str(qc["recommendation"])
            colour = "green" if rec == "ok" else "yellow" if rec == "try_docling" else "red"
            table.add_row(
                pdf_path.name,
                str(qc["word_count"]),
                str(qc["avg_line_length"]),
                f"[{colour}]{rec}[/{colour}]",
                "[green]ok[/green]",
            )
            pdf_path.rename(processed_dir / pdf_path.name)
        except Exception as exc:  # noqa: BLE001
            table.add_row(pdf_path.name, "—", "—", "—", f"[red]error: {exc}[/red]")

    console.print(table)


@pdf.command("status")
@click.option(
    "--inbox", type=click.Path(path_type=Path), default=Path("data/inbox/pdfs"), show_default=True
)  # noqa: E501
@click.option(
    "--processed",
    type=click.Path(path_type=Path),
    default=Path("data/inbox/processed"),
    show_default=True,
)  # noqa: E501
def pdf_status(inbox: Path, processed: Path) -> None:
    """Show what's in the inbox and processed folders."""
    for label, folder in [("Inbox (pending)", inbox), ("Processed", processed)]:
        table = Table(title=label, show_lines=True)
        table.add_column("File")
        table.add_column("Size", justify="right")
        if folder.exists():
            for f in sorted(folder.glob("*.pdf")):
                size_mb = f.stat().st_size / 1_048_576
                table.add_row(f.name, f"{size_mb:.1f} MB")
        else:
            table.add_row("[dim]folder not found[/dim]", "")
        console.print(table)


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
    import json  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415
    from gensim.models import KeyedVectors  # noqa: PLC0415
    from matplotlib.figure import Figure  # noqa: PLC0415

    from interpretant.drift.grid import cell_neighbours  # noqa: PLC0415
    from interpretant.visualization.figures import (  # noqa: PLC0415
        NeighbourColumn,
        corpus_grid,
        excess_by_decade,
        fall_by_model,
        field_label,
        neighbour_panel,
    )

    def column(layer: str, field: str, decade: int, word: str) -> NeighbourColumn:
        vectors = KeyedVectors.load(str(model_dir / layer / f"{field}_{decade}.kv"))
        count = int(vectors.get_vecattr(word, "count")) if word in vectors.key_to_index else 0
        return NeighbourColumn(field, decade, count, cell_neighbours(vectors, word))

    output_dir.mkdir(parents=True, exist_ok=True)
    drawn: dict[str, Figure] = {}

    drawn["embodiment"] = neighbour_panel(
        "embodiment",
        [
            column("openalex_full", "philosophy", decade, "embodiment")
            for decade in (1990, 2000, 2010, 2020)
        ]
        + [column("openalex_full", "hci", decade, "embodiment") for decade in (2000, 2010, 2020)],
        title="Embodiment means two different things",
        subtitle="Nearest neighbours of embodiment in every philosophy and HCI abstract, by decade",
        footnote="TWEC models on the full OpenAlex cells; 25 neighbours from words with at least 50 occurrences, the first 12 distinct words shown (inflections and spelling variants collapsed). Same result with a second seed and with PPMI-SVD.",  # noqa: E501
    )
    drawn["representation_ai"] = neighbour_panel(
        "representation",
        [column("openalex", "ai", decade, "representation") for decade in (1980, 2000, 2020)],
        title="Representation in AI, from symbolic to learned",
        subtitle="Nearest neighbours of representation in 10,000-abstract AI cells",
        footnote="TWEC models on the main OpenAlex grid, first 12 distinct neighbours (inflections and spelling variants collapsed); the symbolic 1980s holds in TWEC only, the learned 2020s in every run.",  # noqa: E501
    )
    table = pd.read_parquet(drift_dir / "cross_perplexity_corrected.parquet")
    drawn["philosophy_hci"] = excess_by_decade(
        table[table["decade"] >= 1990],
        [("philosophy", "hci"), ("hci", "philosophy")],
        title="Philosophy's model learned to predict HCI; HCI's did not learn philosophy",
        subtitle="Excess cross-entropy of one field's held-out abstracts under the other field's bigram model",  # noqa: E501
        footnote="Kneser-Ney bigrams, 200,000 training tokens per cell, shared abstracts removed, 1,000 held-out abstracts; points are 5 resampled runs, lines their mean.",  # noqa: E501
    )
    drawn["hci_gains"] = fall_by_model(
        table,
        "hci",
        1990,
        2020,
        title="HCI's language moved towards the behavioural and social sciences'",
        subtitle="How much better each field's model predicts HCI's abstracts in the 2020s than in the 1990s",  # noqa: E501
        footnote="Bars: mean over five resampled runs; points: the runs. Corrected design (200,000 training tokens, shared abstracts removed, 1,000 held-out abstracts). AI's and computer science's models are the two that did not gain.",  # noqa: E501
    )
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    cells = pd.DataFrame(report["cells"])[["source", "decade", "documents"]]
    fields = sorted(
        {str(source) for source in cells["source"] if source != "books"},
        key=lambda field: field_label(field).lower(),
    )
    drawn["corpus_grid"] = corpus_grid(
        cells[cells["source"] != "books"],
        title="The corpus is a grid",
        subtitle="Research abstracts per field × decade cell after cleaning (sampled to 10,000 per cell)",  # noqa: E501
        footnote="OpenAlex, seeded sample of English research prose; duplicates, catalogue records, scraped pages and non-English text removed; no author above 1% of a cell.",  # noqa: E501
        fields=fields,
    )
    for name, figure in drawn.items():
        path = output_dir / f"{name}.png"
        figure.savefig(path, dpi=150, facecolor=figure.get_facecolor())
        console.print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------


@cli.command("app")
@click.option("--demo", is_flag=True, default=False, help="Load synthetic demo data.")
@click.option("--port", default=8501, show_default=True, help="Streamlit server port.")
def app(demo: bool, port: int) -> None:
    """Launch the Streamlit dashboard."""
    import subprocess
    import sys

    app_path = Path(__file__).parent / "app" / "main.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), f"--server.port={port}"]
    if demo:
        cmd += ["--", "--demo"]
    mode_label = " (demo mode)" if demo else ""
    console.print(f"[bold]Launching dashboard[/bold] on port {port}{mode_label}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    cli()
