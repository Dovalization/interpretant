"""CLI entry point for interpretant."""

from __future__ import annotations

import random
from pathlib import Path

import click
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

console = Console()

_PUBMED_FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"


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


@corpus.command("download")
@click.argument("source", type=click.Choice(["acl", "pubmed"]))
@click.option("--output-dir", type=click.Path(path_type=Path), help="Override raw corpus dir.")
@click.option("--files", default=100, show_default=True, help="(pubmed) XML files to download.")  # noqa: E501
@click.option(
    "--skip", default=0, show_default=True, help="(pubmed) Skip first N files in listing."
)  # noqa: E501
def corpus_download(source: str, output_dir: Path | None, files: int, skip: int) -> None:
    """Download raw corpus data. SOURCE: acl|pubmed"""
    import urllib.request

    if source == "acl":
        url = "https://aclanthology.org/anthology+abstracts.bib.gz"
        dest_dir = output_dir or Path("data/raw/acl")
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / "anthology+abstracts.bib.gz"

        if dest_file.exists():
            console.print(f"[yellow]Already exists:[/yellow] {dest_file} — skipping download.")
            return

        console.print(f"Downloading ACL Anthology to {dest_file} …")
        with tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,  # noqa: E501
            desc="anthology+abstracts.bib.gz",
        ) as bar:

            def _reporthook(count: int, block_size: int, total_size: int) -> None:
                if bar.total is None and total_size > 0:
                    bar.total = total_size
                bar.update(block_size)

            urllib.request.urlretrieve(url, dest_file, reporthook=_reporthook)  # noqa: S310

        console.print(f"[green]Saved to {dest_file}[/green]")

    elif source == "pubmed":
        import re
        import urllib.error

        dest_dir = output_dir or Path("data/raw/pubmed")
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Discover the current baseline prefix by listing the FTP index
        console.print("Fetching PubMed baseline file listing …")
        with urllib.request.urlopen(_PUBMED_FTP_BASE) as resp:  # noqa: S310
            index_html = resp.read().decode("utf-8", errors="replace")

        # Extract .xml.gz filenames from the directory listing
        pattern = re.compile(r'href="(pubmed\d+n\d+\.xml\.gz)"')
        all_files = pattern.findall(index_html)
        if not all_files:
            raise click.ClickException("Could not parse PubMed FTP listing. Check the URL.")

        to_download = all_files[skip : skip + files]
        console.print(f"Downloading {len(to_download)} of {len(all_files)} files (skip={skip}) …")

        for filename in tqdm(to_download, desc="PubMed files", unit=" files"):
            dest_file = dest_dir / filename
            if dest_file.exists():
                continue
            url = f"{_PUBMED_FTP_BASE}{filename}"
            try:
                urllib.request.urlretrieve(url, dest_file)  # noqa: S310
            except urllib.error.URLError as exc:
                console.print(f"[yellow]Warning:[/yellow] failed to download {filename}: {exc}")

        console.print(f"[green]Done. Files saved to {dest_dir}[/green]")


@corpus.command("preprocess")
@click.option(
    "--config-name", default="arxiv", show_default=True, help="Corpus config (arxiv|s2orc)."
)  # noqa: E501
@click.option("--input-dir", type=click.Path(path_type=Path), help="Override raw corpus dir.")
@click.option(
    "--output-dir", type=click.Path(path_type=Path), help="Override processed output dir."
)  # noqa: E501
@click.option("--snapshot-file", default="arxiv-metadata-oai-snapshot.json", show_default=True)
@click.option("--start", default=1990, show_default=True, help="First decade (inclusive).")
@click.option("--end", default=2020, show_default=True, help="Last decade (exclusive).")
@click.option(
    "--min-tokens",
    default=0,
    show_default=True,
    help="Skip docs with fewer tokens (0 = no filter).",
)  # noqa: E501
@click.option(
    "--max-docs",
    default=0,
    show_default=True,
    help="Cap per-decade docs via random sample (0 = no cap).",
)  # noqa: E501
@click.option(
    "--seed", default=42, show_default=True, help="Random seed for --max-docs reproducibility."
)  # noqa: E501
@click.option(
    "--workers",
    default=1,
    show_default=True,
    help="Parallel worker processes (PubMed) or threads (ACL). 1 = sequential.",
)  # noqa: E501
def corpus_preprocess(  # noqa: PLR0913
    config_name: str,
    input_dir: Path | None,
    output_dir: Path | None,
    snapshot_file: str,
    start: int,
    end: int,
    min_tokens: int,
    max_docs: int,
    seed: int,
    workers: int,
) -> None:
    """Preprocess raw corpus text into decade-sliced token files."""
    conf_path = Path("conf") / "corpus" / f"{config_name}.yaml"
    base_path = Path("conf") / "config.yaml"
    if not conf_path.exists():
        raise click.ClickException(f"Config not found: {conf_path}")
    # Merge base config first so ${data_dir} interpolations resolve correctly
    base_cfg = OmegaConf.load(base_path) if base_path.exists() else OmegaConf.create({})
    cfg = OmegaConf.merge(base_cfg, OmegaConf.load(conf_path))

    raw_dir = input_dir or Path(str(cfg.get("raw_dir", f"data/raw/{config_name}")))
    default_processed = f"data/processed/{config_name}"
    processed_dir = output_dir or Path(str(cfg.get("processed_dir", default_processed)))
    texts_dir = processed_dir / "texts"
    texts_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Preprocessing corpus[/bold] (config: {config_name})")
    console.print(f"  raw dir      : {raw_dir}")
    console.print(f"  output dir   : {texts_dir}")
    console.print(f"  decades      : {start}–{end - 1}")
    if min_tokens:
        console.print(f"  min tokens   : {min_tokens}")
    if max_docs:
        console.print(f"  max docs/dec : {max_docs} (seed={seed})")
    if workers > 1:
        console.print(f"  workers      : {workers}")

    if config_name == "arxiv":
        from interpretant.corpus.arxiv import ArxivSource

        categories: list[str] | None = list(cfg.get("categories", [])) or None
        min_year: int = int(cfg.get("min_year", start))
        max_year: int = int(cfg.get("max_year", end - 1))
        source = ArxivSource(
            raw_dir=raw_dir,
            categories=categories,
            min_year=min_year,
            max_year=max_year,
            snapshot_filename=snapshot_file,
        )
    elif config_name == "acl":
        from interpretant.corpus.acl import ACLAnthologySource

        min_year_acl: int = int(cfg.get("min_year", start))
        max_year_acl: int = int(cfg.get("max_year", end - 1))
        bib_filename: str = str(cfg.get("bib_filename", "anthology+abstracts.bib.gz"))
        source = ACLAnthologySource(
            raw_dir=raw_dir,
            min_year=min_year_acl,
            max_year=max_year_acl,
            bib_filename=bib_filename,
        )
    elif config_name == "pubmed":
        from interpretant.corpus.pubmed import PubMedSource

        min_year_pm: int = int(cfg.get("min_year", start))
        max_year_pm: int = int(cfg.get("max_year", end - 1))
        lang_filter_raw: list[str] = list(cfg.get("language_filter", ["eng"]))
        source = PubMedSource(
            raw_dir=raw_dir,
            min_year=min_year_pm,
            max_year=max_year_pm,
            language_filter=lang_filter_raw,
        )
    else:
        raise click.ClickException(
            f"Preprocessing for '{config_name}' is not yet implemented. "
            "Use 'arxiv', 'acl', or 'pubmed'."
        )

    table = Table(title="Preprocessing results", show_lines=True)
    table.add_column("Decade", style="cyan")
    table.add_column("Documents", justify="right")
    table.add_column("Output file")

    rng = random.Random(seed)
    out_files = {
        decade: texts_dir / f"{config_name}_{decade}.txt" for decade in range(start, end, 10)
    }
    counts: dict[int, int] = dict.fromkeys(out_files, 0)

    if max_docs:
        # Accumulate first so we can sample uniformly across the full decade.
        all_buckets: dict[int, list[str]] = {decade: [] for decade in out_files}
        for decade, docs in source.iter_decade_slices(
            start=start, end=end, step=10, min_tokens=min_tokens, workers=workers
        ):
            if decade in all_buckets:
                all_buckets[decade].extend(docs)
        for decade, all_docs in sorted(all_buckets.items()):
            sampled = rng.sample(all_docs, min(max_docs, len(all_docs)))
            out_files[decade].write_text("\n".join(sampled), encoding="utf-8")
            counts[decade] = len(sampled)
    else:
        # Streaming: truncate output files upfront, then append as results arrive.
        # Keeps memory usage bounded to one file's worth of docs at a time.
        for out_file in out_files.values():
            out_file.write_text("", encoding="utf-8")
        for decade, docs in source.iter_decade_slices(
            start=start, end=end, step=10, min_tokens=min_tokens, workers=workers
        ):
            if decade not in out_files or not docs:
                continue
            with out_files[decade].open("a", encoding="utf-8") as fh:
                fh.write("\n".join(docs) + "\n")
            counts[decade] += len(docs)

    for decade in sorted(out_files):
        table.add_row(str(decade), str(counts[decade]), str(out_files[decade]))

    console.print(table)
    console.print("[green]Done.[/green]")


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
def corpus_books_ingest(
    books_dir: Path,
    output_dir: Path | None,
    start: int,
    end: int,
) -> None:
    """Preprocess extracted book texts into decade-sliced token files."""
    from interpretant.corpus.books import BooksSource

    source = BooksSource(books_dir=books_dir)
    texts_dir = output_dir or Path("data/processed/books/texts")
    texts_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Ingesting books corpus[/bold]")
    console.print(f"  books dir  : {books_dir}")
    console.print(f"  output dir : {texts_dir}")
    console.print(f"  decades    : {start}–{end - 1}")

    table = Table(title="Books ingest results", show_lines=True)
    table.add_column("Decade", style="cyan")
    table.add_column("Documents", justify="right")
    table.add_column("Output file")

    for decade, docs in source.iter_decade_slices(start=start, end=end, step=10):
        out_file = texts_dir / f"books_{decade}.txt"
        out_file.write_text("\n".join(docs), encoding="utf-8")
        table.add_row(str(decade), str(len(docs)), str(out_file))

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
    as duplicates or archive catalogue records.
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
            dropped = cell.duplicates + cell.catalogue_records
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
    default=Path("data/processed/phrases.model"),
    show_default=True,
)
def corpus_phrases_learn(  # noqa: PLR0913
    processed_dir: Path,
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

    paths = sorted(processed_dir.glob("*/texts/*.txt"))
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
    default=Path("data/processed"),
    show_default=True,
)
@click.option(
    "--model",
    "model_path",
    type=click.Path(path_type=Path),
    default=Path("data/processed/phrases.model"),
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/phrased"),
    show_default=True,
)
def corpus_phrases_apply(processed_dir: Path, model_path: Path, output_dir: Path) -> None:
    """Rewrite every processed file with phrases merged, mirroring the directory layout."""
    from interpretant.corpus.phrases import PhraseModel

    model = PhraseModel.load(model_path)
    for path in sorted(processed_dir.glob("*/texts/*.txt")):
        target = output_dir / path.relative_to(processed_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        with path.open(encoding="utf-8") as source, target.open("w", encoding="utf-8") as out:
            for line in source:
                out.write(model.apply(line) + "\n")
        console.print(f"  {target}")


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
# embed
# ---------------------------------------------------------------------------


@cli.group()
def embed() -> None:
    """Embedding training commands."""


@embed.command("train")
@click.option(
    "--config-name",
    default="word2vec",
    show_default=True,
    help="Embedding config (word2vec|fasttext).",
)  # noqa: E501
@click.option(
    "--corpus",
    "corpora",
    multiple=True,
    default=["arxiv", "acl", "pubmed"],
    show_default=True,
    help="Corpus sources to merge (repeatable). Also accepts 'books'.",
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
    help="Model output dir (default: models/raw/<config-name>).",
)  # noqa: E501
@click.option("--start", default=1970, show_default=True, help="First decade to train (inclusive).")
@click.option("--end", default=2020, show_default=True, help="Last decade (exclusive).")
@click.option("--vector-size", default=300, show_default=True, help="Embedding dimensionality.")
@click.option("--window", default=10, show_default=True, help="Context window size.")
@click.option("--min-count", default=10, show_default=True, help="Minimum word frequency.")
@click.option("--workers", default=4, show_default=True, help="Parallel training workers.")
@click.option("--epochs", default=5, show_default=True, help="Training epochs per decade.")
@click.option(
    "--separate",
    is_flag=True,
    default=False,
    help="Train one model per corpus per decade instead of merging.",
)  # noqa: E501
def embed_train(  # noqa: PLR0913
    config_name: str,
    corpora: tuple[str, ...],
    input_dir: Path,
    output_dir: Path | None,
    start: int,
    end: int,
    vector_size: int,
    window: int,
    min_count: int,
    workers: int,
    epochs: int,
    separate: bool,
) -> None:
    """Train decade-sliced word embedding models from preprocessed text."""
    if config_name == "word2vec":
        from interpretant.embedding.word2vec import Word2VecTrainer

        trainer_cls = Word2VecTrainer
    elif config_name == "fasttext":
        from interpretant.embedding.fasttext import FastTextTrainer

        trainer_cls = FastTextTrainer  # type: ignore[assignment]
    else:
        raise click.ClickException(
            f"Unknown config: {config_name!r}. Use 'word2vec' or 'fasttext'."
        )  # noqa: E501

    models_dir = output_dir or Path("models") / "raw" / config_name
    models_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Training embeddings[/bold] (config: {config_name})")
    console.print(f"  corpora  : {', '.join(corpora)}")
    console.print(f"  decades  : {start}–{end - 1} (step 10)")
    console.print(f"  output   : {models_dir}")
    console.print(
        f"  dims     : {vector_size}  window: {window}  min_count: {min_count}  epochs: {epochs}"
    )  # noqa: E501
    if separate:
        console.print("  mode     : separate (one model per corpus per decade)")

    if separate:
        table = Table(title="Training results (separate)", show_lines=True)
        table.add_column("Corpus", style="magenta")
        table.add_column("Decade", style="cyan")
        table.add_column("Documents", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Vocab", justify="right")
        table.add_column("Model path")

        for corpus_name in corpora:
            corpus_models_dir = models_dir / corpus_name
            corpus_models_dir.mkdir(parents=True, exist_ok=True)
            for decade in range(start, end, 10):
                candidate = input_dir / corpus_name / "texts" / f"{corpus_name}_{decade}.txt"
                if not candidate.exists():
                    console.print(
                        f"[yellow]Skipping {corpus_name}/{decade}[/yellow] — {candidate} not found"
                    )
                    continue

                sentences: list[list[str]] = []
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    tokens = line.split()
                    if tokens:
                        sentences.append(tokens)

                total_tokens = sum(len(s) for s in sentences)
                console.print(
                    f"  {corpus_name}/{decade}: {len(sentences):,} docs, {total_tokens:,} tokens — training …"  # noqa: E501
                )

                trainer = trainer_cls(
                    vector_size=vector_size,
                    window=window,
                    min_count=min_count,
                    workers=workers,
                    epochs=epochs,
                )
                trainer.train(sentences, decade)
                model_path = trainer.save(corpus_models_dir, decade)
                vocab_size = len(trainer.vocabulary())

                table.add_row(
                    corpus_name,
                    str(decade),
                    f"{len(sentences):,}",
                    f"{total_tokens:,}",
                    f"{vocab_size:,}",
                    str(model_path),
                )

        console.print(table)
        console.print("[green]Done.[/green]")
        return

    table = Table(title="Training results", show_lines=True)
    table.add_column("Decade", style="cyan")
    table.add_column("Documents", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Vocab", justify="right")
    table.add_column("Model path")

    for decade in range(start, end, 10):
        # Collect text files for this decade across all requested corpora
        decade_files: list[Path] = []
        for corpus_name in corpora:
            candidate = input_dir / corpus_name / "texts" / f"{corpus_name}_{decade}.txt"
            if candidate.exists():
                decade_files.append(candidate)

        if not decade_files:
            console.print(
                f"[yellow]Skipping {decade}[/yellow] — no text files found in {input_dir}"
            )  # noqa: E501
            continue

        # Load and tokenise all documents for this decade
        sentences_merged: list[list[str]] = []
        for text_file in decade_files:
            for line in text_file.read_text(encoding="utf-8").splitlines():
                tokens = line.split()
                if tokens:
                    sentences_merged.append(tokens)

        total_tokens = sum(len(s) for s in sentences_merged)
        console.print(
            f"  {decade}: {len(sentences_merged):,} docs, {total_tokens:,} tokens — training …"
        )  # noqa: E501

        trainer = trainer_cls(
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            epochs=epochs,
        )
        trainer.train(sentences_merged, decade)
        model_path = trainer.save(models_dir, decade)
        vocab_size = len(trainer.vocabulary())

        table.add_row(
            str(decade),
            f"{len(sentences_merged):,}",
            f"{total_tokens:,}",
            f"{vocab_size:,}",
            str(model_path),
        )

    console.print(table)
    console.print("[green]Done.[/green]")


# ---------------------------------------------------------------------------
# twec
# ---------------------------------------------------------------------------


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

    Outputs one .kv file per decade directly usable by 'drift compute'
    without a separate alignment step.
    """
    import logging  # noqa: PLC0415
    import sys  # noqa: PLC0415

    from interpretant.embedding.twec import TWECTrainer, _EpochProgressCallback  # noqa: PLC0415

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
    console.print(
        "\n[green]Done.[/green] Run drift compute with: "
        f"[cyan]uv run interpretant drift compute --model-dir {output_dir}[/cyan]"
    )


# ---------------------------------------------------------------------------
# align
# ---------------------------------------------------------------------------


@cli.group()
def align() -> None:
    """Alignment commands."""


@align.command("run")
@click.option(
    "--config-name",
    default="procrustes",
    show_default=True,
    type=click.Choice(["procrustes"]),
    help="Alignment method.",
)  # noqa: E501
@click.option(
    "--model-dir",
    type=click.Path(path_type=Path),
    default=Path("models/raw/word2vec"),
    show_default=True,
    help="Raw gensim models dir.",
)  # noqa: E501
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("models/aligned"),
    show_default=True,
    help="Aligned KeyedVectors output dir.",
)  # noqa: E501
@click.option(
    "--reference-decade",
    default=2010,
    show_default=True,
    help="Decade all others are rotated onto.",
)  # noqa: E501
def align_run(
    config_name: str,
    model_dir: Path,
    output_dir: Path,
    reference_decade: int,
) -> None:
    """Align decade-sliced embedding spaces via Procrustes rotation."""
    from interpretant.alignment.procrustes import ProcrustesAligner

    model_paths = {int(p.stem): p for p in sorted(model_dir.glob("*.model"))}
    if not model_paths:
        raise click.ClickException(f"No .model files found in {model_dir}")

    console.print(f"[bold]Aligning embeddings[/bold] (config: {config_name})")
    console.print(f"  model dir        : {model_dir}")
    console.print(f"  output dir       : {output_dir}")
    console.print(f"  reference decade : {reference_decade}")
    console.print(f"  decades          : {sorted(model_paths.keys())}")

    aligner = ProcrustesAligner(reference_decade=reference_decade)
    console.print("Fitting Procrustes rotation matrices …")
    aligner.fit(model_paths)

    shared = aligner.shared_vocabulary()
    console.print(f"Shared vocabulary: [cyan]{len(shared):,}[/cyan] words")

    saved = aligner.align(output_dir)

    table = Table(title="Alignment results", show_lines=True)
    table.add_column("Decade", style="cyan")
    table.add_column("Output file")
    for decade, path in sorted(saved.items()):
        table.add_row(str(decade), str(path))
    console.print(table)
    console.print("[green]Done.[/green]")


# ---------------------------------------------------------------------------
# drift
# ---------------------------------------------------------------------------


@cli.group()
def drift() -> None:
    """Drift computation commands."""


@drift.command("compute")
@click.option(
    "--model-dir",
    type=click.Path(path_type=Path),
    default=Path("models/aligned"),
    show_default=True,
    help="Aligned KeyedVectors dir.",
)  # noqa: E501
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("data/processed/drift.parquet"),
    show_default=True,
    help="Output parquet file path.",
)
@click.option("--words", "-w", multiple=True, help="Words to analyse (default: all shared vocab).")
@click.option(
    "--neighborhood-k",
    type=int,
    default=10,
    show_default=True,
    help="Nearest-neighbour count for neighbourhood shift Jaccard distance.",
)  # noqa: E501
def drift_compute(  # noqa: E501
    model_dir: Path, output: Path, words: tuple[str, ...], neighborhood_k: int
) -> None:
    """Compute drift metrics for aligned models and write to parquet."""
    import pandas as pd

    from interpretant.alignment.procrustes import ProcrustesAligner
    from interpretant.drift.metrics import (
        average_pairwise_distance,
        batch_neighborhood_shift,
        cosine_distance,
        frequency_corrected_drift,
    )

    console.print("[bold]Computing drift metrics[/bold]")
    console.print(f"  aligned models : {model_dir}")
    console.print(f"  output         : {output}")

    aligner = ProcrustesAligner()
    aligner.load_aligned(model_dir)
    decades = aligner.decades()

    if not decades:
        raise click.ClickException(f"No aligned models found in {model_dir}")

    words_in_any_decade = set.union(*[aligner.vocab_for_decade(d) for d in decades])

    if words:
        never_present = [w for w in words if w not in words_in_any_decade]
        if never_present:
            console.print(  # noqa: E501
                f"[yellow]Warning: {len(never_present)} word(s) not found in any decade: {', '.join(never_present)}[/yellow]"  # noqa: E501
            )
        target_words = [w for w in words if w in words_in_any_decade]
    else:
        target_words = sorted(words_in_any_decade)
    console.print(f"  words          : {len(target_words):,}")
    console.print(f"  decades        : {decades}")
    console.print(f"  neighborhood-k : {neighborhood_k}")

    freq_cache = {decade: aligner.word_freqs(decade) for decade in decades}
    has_freq = any(f is not None for f in freq_cache.values())
    if has_freq:
        console.print("  frequency data : available (cosine_drift_corrected will be computed)")
    else:
        console.print(
            "  [yellow]frequency data : not found in .kv files — cosine_drift_corrected will be NaN[/yellow]"  # noqa: E501
        )

    # avg_pairwise_distance spans all decades — compute only for global shared vocab.
    shared_vocab = aligner.shared_vocabulary()
    apd_map: dict[str, float] = {
        word: average_pairwise_distance([aligner.get_vector(word, d) for d in decades])
        for word in shared_vocab
    }

    rows = []
    for i in tqdm(range(1, len(decades)), desc="Computing drift", unit="transition"):
        t0, t1 = decades[i - 1], decades[i]
        pair_vocab = aligner.vocabulary_for_pair(t0, t1)
        pair_vocab_set = set(pair_vocab)
        pair_targets = [w for w in target_words if w in pair_vocab_set]

        # Build normalised matrices directly from KeyedVectors — no intermediate dict.
        words_t0, matrix_t0 = aligner.build_pair_index(t0, pair_vocab)
        words_t1, matrix_t1 = aligner.build_pair_index(t1, pair_vocab)

        ns_scores = batch_neighborhood_shift(
            pair_targets,
            words_t0,
            matrix_t0,  # type: ignore[arg-type]
            words_t1,
            matrix_t1,  # type: ignore[arg-type]
            k=neighborhood_k,
        )

        freqs_t0 = freq_cache.get(t0)
        freqs_t1 = freq_cache.get(t1)
        for word in pair_targets:
            v0 = aligner.get_vector(word, t0)
            v1 = aligner.get_vector(word, t1)
            cd = cosine_distance(v0, v1)
            ns = ns_scores.get(word, 0.0)
            if freqs_t0 and freqs_t1 and word in freqs_t0 and word in freqs_t1:
                cd_corrected: float | None = frequency_corrected_drift(
                    cd, freqs_t0[word], freqs_t1[word]
                )
            else:
                cd_corrected = None
            rows.append(
                {
                    "word": word,
                    "decade": t1,
                    "cosine_drift": cd,
                    "cosine_drift_corrected": cd_corrected,
                    "neighborhood_shift": ns,
                }
            )

    df = pd.DataFrame(rows)
    df["avg_pairwise_distance"] = df["word"].map(apd_map)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)

    console.print(f"\n[green]Wrote {len(df):,} rows → {output}[/green]")
    console.print(df.describe().to_string())


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
