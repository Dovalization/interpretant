# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install all dependencies (including dev extras)
uv sync --all-extras

# Lint and format check
make lint                    # ruff check + ruff format --check
make fmt                     # ruff format + ruff check --fix (auto-fixes)

# Type checking
make type-check              # mypy src/ (strict mode)

# Tests
make test                    # pytest with coverage
uv run pytest tests/test_drift.py -v          # single test file
uv run pytest tests/ -k "test_cosine" -v      # single test by name

# Dashboard
make app-demo                # synthetic data, no corpus required
make app                     # real data (requires models/aligned/*.kv)

# Full pipeline (DVC)
make pipeline                # dvc repro
```

## Pipeline

The pipeline has five sequential stages. Commands must be run in this order; the CLI is the authoritative entry point for each stage.

```
corpus preprocess → embed train → align run → drift compute → app
```

1. **`corpus preprocess --config-name <acl|pubmed|arxiv>`** — reads raw files from `data/raw/<source>/`, writes decade-sliced text files to `data/processed/<source>/texts/<source>_<decade>.txt`. One doc per line, whitespace-tokenised. PubMed uses `ProcessPoolExecutor` for parallelism; ACL uses `ThreadPoolExecutor`.

2. **`embed train`** — reads all `data/processed/<corpus>/texts/<corpus>_<decade>.txt` files for each decade, merges them (or trains separately with `--separate`), trains a gensim Word2Vec or FastText model per decade, saves to `models/raw/<config-name>/<decade>.model`.

3. **`align run`** — loads all `.model` files, computes orthogonal Procrustes rotation for each decade onto a reference decade (default: 2010), saves aligned `KeyedVectors` to `models/aligned/<decade>.kv`.

4. **`drift compute`** — loads aligned `.kv` files, computes cosine distance, neighbourhood shift (Jaccard), and average pairwise distance for each word across consecutive decades, writes `data/processed/drift.parquet`.

5. **`app`** — Streamlit dashboard; reads `models/aligned/*.kv` directly at startup. Demo mode (`--demo`) uses seeded synthetic data and requires no corpus.

## Architecture

**`src/interpretant/`**

- **`__main__.py`** — the entire CLI; Click groups `corpus`, `embed`, `align`, `drift`, `pdf`, and command `app`. All pipeline logic lives here; the library modules provide pure functions/classes only.

- **`corpus/`** — all corpus sources implement `CorpusSource` (ABC in `base.py`). The key method is `iter_decade_slices(start, end, step, min_tokens, workers) -> Iterator[tuple[int, list[str]]]`. Sources: `pubmed.py` (XML.gz streaming with `iterparse`), `acl.py` (BibTeX), `arxiv.py` (JSONL), `books.py` (pre-extracted text files + manifest), `s2orc.py` (stub). `preprocessing.py` has shared text-cleaning helpers. `pdf_extractor.py` wraps Docling with PyMuPDF fallback.

- **`embedding/`** — `EmbeddingTrainer` ABC with `Word2VecTrainer` and `FastTextTrainer`. Each `train(sentences, decade)` then `save(output_dir, decade) -> Path`.

- **`alignment/`** — `Aligner` ABC. `ProcrustesAligner` is the active implementation; `TWECAligner` is a stub requiring a custom gensim fork (not on PyPI). `ProcrustesAligner.fit()` loads all decade models, computes shared vocab, and rotates each onto the reference decade. `load_aligned()` is the read path used by `drift compute` and `app`.

- **`drift/metrics.py`** — pure functions: `cosine_distance`, `neighborhood_shift` (Jaccard on top-25 neighbours), `frequency_corrected_drift`, `average_pairwise_distance`. `_top_k_neighbors` is module-level (not private to a class) so it can be used by both metrics and visualisation.

- **`visualization/`** — `trajectory.py` (PCA → 2D Plotly scatter), `neighbors.py` (nearest-neighbour comparison DataFrame), `heatmap.py` (stub for drift heatmap panel).

- **`drift/changepoint.py`** — stub for PELT/BOCPD changepoint detection.

- **`app/main.py`** — Streamlit dashboard. In demo mode, generates synthetic vectors with `_make_synthetic_vectors()`. In real mode, calls `ProcrustesAligner.load_aligned()` at startup.

## Configuration

Hydra/OmegaConf configs live in `conf/`. The CLI merges `conf/config.yaml` (base, defines `data_dir`) with `conf/corpus/<name>.yaml` before constructing corpus sources. Embedding and alignment configs are in `conf/embedding/` and `conf/alignment/` but are currently used only for documentation — the CLI accepts their parameters as explicit options.

## Tests

Tests are **synthetic only** — no real NLP data, no network calls. `tests/conftest.py` provides `synthetic_model` (seeded `{decade: {word: vector}}`) and `synthetic_vocab`. Tests cover: drift metrics, Procrustes alignment, embedding trainers, corpus preprocessing utilities, and PDF extractor quality checks.

## Code style

- Line length: 100 (`ruff.toml`)
- mypy strict — all public functions need type annotations; use `# type: ignore[...]` with a specific code when suppression is necessary
- Long CLI `@click.option` lines use `# noqa: E501`; module-level free functions used with `ProcessPoolExecutor` must be defined at module level (not nested) for pickle compatibility on macOS
