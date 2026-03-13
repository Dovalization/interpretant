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
@click.option("--skip", default=0, show_default=True, help="(pubmed) Skip first N files in listing.")  # noqa: E501
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
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024,  # noqa: E501
                  desc="anthology+abstracts.bib.gz") as bar:
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
@click.option("--config-name", default="arxiv", show_default=True, help="Corpus config (arxiv|s2orc).")  # noqa: E501
@click.option("--input-dir", type=click.Path(path_type=Path), help="Override raw corpus dir.")
@click.option("--output-dir", type=click.Path(path_type=Path), help="Override processed output dir.")  # noqa: E501
@click.option("--snapshot-file", default="arxiv-metadata-oai-snapshot.json", show_default=True)
@click.option("--start", default=1990, show_default=True, help="First decade (inclusive).")
@click.option("--end", default=2020, show_default=True, help="Last decade (exclusive).")
@click.option("--min-tokens", default=0, show_default=True, help="Skip docs with fewer tokens (0 = no filter).")  # noqa: E501
@click.option("--max-docs", default=0, show_default=True, help="Cap per-decade docs via random sample (0 = no cap).")  # noqa: E501
@click.option("--seed", default=42, show_default=True, help="Random seed for --max-docs reproducibility.")  # noqa: E501
@click.option("--workers", default=1, show_default=True, help="Parallel worker processes (PubMed) or threads (ACL). 1 = sequential.")  # noqa: E501
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
    console.print(f"  decades      : {start}–{end-1}")
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
    for decade, docs in source.iter_decade_slices(
        start=start, end=end, step=10, min_tokens=min_tokens, workers=workers
    ):
        if max_docs and len(docs) > max_docs:
            docs = rng.sample(docs, max_docs)
        out_file = texts_dir / f"{config_name}_{decade}.txt"
        out_file.write_text("\n".join(docs), encoding="utf-8")
        table.add_row(str(decade), str(len(docs)), str(out_file))

    console.print(table)
    console.print("[green]Done.[/green]")


@corpus.group("books")
def corpus_books() -> None:
    """Books corpus commands (PDF-extracted texts)."""


@corpus_books.command("ingest")
@click.option("--books-dir", type=click.Path(path_type=Path), default=Path("data/external/books"), show_default=True)  # noqa: E501
@click.option("--output-dir", type=click.Path(path_type=Path), help="Override processed output dir.")  # noqa: E501
@click.option("--start", default=1900, show_default=True, help="First decade (inclusive).")
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


@corpus_books.command("stats")
@click.option("--books-dir", type=click.Path(path_type=Path), default=Path("data/external/books"), show_default=True)  # noqa: E501
def corpus_books_stats(books_dir: Path) -> None:
    """Show book counts per decade and manifest status."""
    from interpretant.corpus.books import BooksSource

    source = BooksSource(books_dir=books_dir)
    stats = source.stats()

    table = Table(title="Books corpus stats", show_lines=True)
    table.add_column("Decade", style="cyan")
    table.add_column("Books", justify="right")
    for decade_label, count in stats["per_decade"].items():  # type: ignore[union-attr]
        table.add_row(decade_label, str(count))
    table.add_row("[bold]Total[/bold]", f"[bold]{stats['total_books']}[/bold]")
    console.print(table)
    console.print(f"Manifest entries: {stats['manifest_entries']}")


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------


@cli.group()
def embed() -> None:
    """Embedding training commands."""


@embed.command("train")
@click.option("--config-name", default="word2vec", show_default=True, help="Embedding config (word2vec|fasttext).")  # noqa: E501
@click.option("--corpus", "corpora", multiple=True, default=["arxiv", "acl", "pubmed"], show_default=True, help="Corpus sources to merge (repeatable). Also accepts 'books'.")  # noqa: E501
@click.option("--input-dir", type=click.Path(path_type=Path), default=Path("data/processed"), show_default=True, help="Root of processed text dirs.")  # noqa: E501
@click.option("--output-dir", type=click.Path(path_type=Path), help="Model output dir (default: models/raw/<config-name>).")  # noqa: E501
@click.option("--start", default=1970, show_default=True, help="First decade to train (inclusive).")
@click.option("--end", default=2020, show_default=True, help="Last decade (exclusive).")
@click.option("--vector-size", default=300, show_default=True, help="Embedding dimensionality.")
@click.option("--window", default=10, show_default=True, help="Context window size.")
@click.option("--min-count", default=10, show_default=True, help="Minimum word frequency.")
@click.option("--workers", default=4, show_default=True, help="Parallel training workers.")
@click.option("--epochs", default=5, show_default=True, help="Training epochs per decade.")
@click.option("--separate", is_flag=True, default=False, help="Train one model per corpus per decade instead of merging.")  # noqa: E501
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
        raise click.ClickException(f"Unknown config: {config_name!r}. Use 'word2vec' or 'fasttext'.")  # noqa: E501

    models_dir = output_dir or Path("models") / "raw" / config_name
    models_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Training embeddings[/bold] (config: {config_name})")
    console.print(f"  corpora  : {', '.join(corpora)}")
    console.print(f"  decades  : {start}–{end - 1} (step 10)")
    console.print(f"  output   : {models_dir}")
    console.print(f"  dims     : {vector_size}  window: {window}  min_count: {min_count}  epochs: {epochs}")  # noqa: E501
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
            console.print(f"[yellow]Skipping {decade}[/yellow] — no text files found in {input_dir}")  # noqa: E501
            continue

        # Load and tokenise all documents for this decade
        sentences_merged: list[list[str]] = []
        for text_file in decade_files:
            for line in text_file.read_text(encoding="utf-8").splitlines():
                tokens = line.split()
                if tokens:
                    sentences_merged.append(tokens)

        total_tokens = sum(len(s) for s in sentences_merged)
        console.print(f"  {decade}: {len(sentences_merged):,} docs, {total_tokens:,} tokens — training …")  # noqa: E501

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
# align
# ---------------------------------------------------------------------------


@cli.group()
def align() -> None:
    """Alignment commands."""


@align.command("run")
@click.option("--config-name", default="procrustes", show_default=True, type=click.Choice(["procrustes"]), help="Alignment method.")  # noqa: E501
@click.option("--model-dir", type=click.Path(path_type=Path), default=Path("models/raw/word2vec"), show_default=True, help="Raw gensim models dir.")  # noqa: E501
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("models/aligned"), show_default=True, help="Aligned KeyedVectors output dir.")  # noqa: E501
@click.option("--reference-decade", default=2010, show_default=True, help="Decade all others are rotated onto.")  # noqa: E501
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
@click.option("--model-dir", type=click.Path(path_type=Path), default=Path("models/aligned"), show_default=True, help="Aligned KeyedVectors dir.")  # noqa: E501
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("data/processed/drift.parquet"),
    show_default=True,
    help="Output parquet file path.",
)
@click.option("--words", "-w", multiple=True, help="Words to analyse (default: all shared vocab).")
def drift_compute(model_dir: Path, output: Path, words: tuple[str, ...]) -> None:
    """Compute drift metrics for aligned models and write to parquet."""
    import pandas as pd

    from interpretant.alignment.procrustes import ProcrustesAligner
    from interpretant.drift.metrics import (
        average_pairwise_distance,
        cosine_distance,
        neighborhood_shift,
    )

    console.print("[bold]Computing drift metrics[/bold]")
    console.print(f"  aligned models : {model_dir}")
    console.print(f"  output         : {output}")

    aligner = ProcrustesAligner()
    aligner.load_aligned(model_dir)
    decades = aligner.decades()

    if not decades:
        raise click.ClickException(f"No aligned models found in {model_dir}")

    target_words = list(words) if words else aligner.shared_vocabulary()
    console.print(f"  words          : {len(target_words):,}")
    console.print(f"  decades        : {decades}")

    # Build {decade: {word: vector}} lookup
    vectors_by_decade: dict[int, dict[str, object]] = {
        decade: {word: aligner.get_vector(word, decade) for word in target_words}
        for decade in decades
    }
    # Full trajectory per word for avg_pairwise_distance
    trajectories: dict[str, list[object]] = {
        word: [vectors_by_decade[d][word] for d in decades]
        for word in target_words
    }

    rows = []
    for word in tqdm(target_words, desc="Computing drift", unit=" words"):
        for i in range(1, len(decades)):
            t0, t1 = decades[i - 1], decades[i]
            v0 = vectors_by_decade[t0][word]
            v1 = vectors_by_decade[t1][word]
            cd = cosine_distance(v0, v1)  # type: ignore[arg-type]
            ns = neighborhood_shift(word, vectors_by_decade[t0], vectors_by_decade[t1])  # type: ignore[arg-type]
            apd = average_pairwise_distance(trajectories[word])  # type: ignore[arg-type]
            rows.append({
                "word": word,
                "decade": t1,
                "cosine_drift": cd,
                "neighborhood_shift": ns,
                "avg_pairwise_distance": apd,
            })

    df = pd.DataFrame(rows)
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
@click.option("--backend", default="docling", show_default=True, type=click.Choice(["docling", "pymupdf", "auto"]))  # noqa: E501
@click.option("--author", default="", help="Author metadata for manifest.")
@click.option("--title", default="", help="Title metadata for manifest.")
@click.option("--year", default=0, type=int, help="Publication year.")
@click.option("--decade", default="", help="Decade bucket (e.g. 1960s).")
@click.option("--fields", multiple=True, help="Subject fields (repeatable).")
@click.option("--add-to-manifest", is_flag=True, default=False, help="Append entry to manifest.json.")  # noqa: E501
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
    console.print(f"Extracting [bold]{pdf_path.name}[/bold] using [cyan]{extractor.backend_name}[/cyan] …")  # noqa: E501
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
        console.print("[yellow]Warning:[/yellow] Output may have layout artifacts. Re-run with --backend docling.")  # noqa: E501

    if add_to_manifest:
        import json
        manifest_path = Path("data/external/books/manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, object]] = []
        if manifest_path.exists():
            entries = json.loads(manifest_path.read_text())
        entries.append({
            "file": str(dest),
            "author": author,
            "title": title,
            "year": year,
            "decade": decade,
            "fields": list(fields),
            "status": "available",
            "source": "pdf_extraction",
        })
        manifest_path.write_text(json.dumps(entries, indent=2))
        console.print(f"[green]Added to manifest:[/green] {manifest_path}")

    processed_dir = Path("data/inbox/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    pdf_path.rename(processed_dir / pdf_path.name)
    console.print(f"[dim]Moved {pdf_path.name} → {processed_dir}[/dim]")


@pdf.command("batch")
@click.option("--backend", default="docling", show_default=True, type=click.Choice(["docling", "pymupdf", "auto"]))  # noqa: E501
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("data/external/books"), show_default=True)  # noqa: E501
@click.option("--decade", required=True, help="Decade bucket for all PDFs (e.g. 1960s).")
@click.option("--inbox", type=click.Path(path_type=Path), default=Path("data/inbox/pdfs"), show_default=True)  # noqa: E501
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
@click.option("--inbox", type=click.Path(path_type=Path), default=Path("data/inbox/pdfs"), show_default=True)  # noqa: E501
@click.option("--processed", type=click.Path(path_type=Path), default=Path("data/inbox/processed"), show_default=True)  # noqa: E501
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
