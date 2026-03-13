# interpretant — Roadmap

Diachronic semantic drift tracker for academic corpora.
Tracks how words shift meaning over time across scientific, linguistic, and biomedical discourse.

---

## Pipeline overview

```
corpus ingest → preprocess → embed → align → drift → visualise
```

---

## Stage 1 — Corpus ingestion & preprocessing

### Sources

| Source | Decades covered | Raw docs | Status |
|--------|----------------|----------|--------|
| arXiv (Kaggle snapshot) | 1990–2010 | ~78K | ✅ done |
| ACL Anthology | 1960–2020 | ~90K | ✅ done |
| PubMed baseline (n0001–n0100) | 1970–1990 | ~3M | ✅ done |
| PubMed baseline (n0800–n0900) | ~2000s | ~3M est. | 🔄 downloading |
| PubMed baseline (n1100–n1200) | ~2010s+ | ~3M est. | 🔄 downloading |

### Processed text files

| Decade | arXiv | ACL | PubMed | Total |
|--------|-------|-----|--------|-------|
| 1960s  | —     | 160 | —      | 160   |
| 1970s  | —     | 458 | 1,255,590 | 1,256,048 |
| 1980s  | —     | 2,389 | 615,698 | 618,087 |
| 1990s  | 1,729 | 5,974 | 1,128,709 | 1,136,412 |
| 2000s  | 4,383 | 15,352 | *(pending)* | 19,735+ |
| 2010s  | 72,164 | 34,740 | *(pending)* | 106,904+ |
| 2020s  | —     | 31,388 | *(pending)* | 31,388+ |

### Known data quality notes
- arXiv: clean, no short lines, 25 duplicates in 72K (negligible)
- ACL: ~5% title-only docs (pre-2000, no abstract), ~500 duplicates in 1990s — acceptable
- PubMed: proceedings/venue entries filtered; accent artifacts (`{\'e}`) fixed
- PubMed decade coverage is driven by PMID ordering — n0001–n0100 = pre-1990, n0800+ = 2000s, n1100+ = 2010s+

### Pending
- [ ] PubMed n0800 download completes → re-run `corpus preprocess --config-name pubmed`
- [ ] PubMed n1100 download completes → re-run `corpus preprocess --config-name pubmed`
- [ ] Validate 2000s/2010s PubMed doc counts look reasonable

---

## Stage 2 — Embedding training

**Status: implemented, not yet run on real data**

### What's implemented ✅
- `Word2VecTrainer` — skip-gram, 300-dim, window=10, min_count=10, 5 epochs
- `FastTextTrainer` — same interface, subword-aware
- `embed train` CLI — fully wired:
  - `--corpus` (repeatable): arxiv | acl | pubmed | books — merged per decade
  - `--start / --end` decade range
  - `--vector-size / --window / --min-count / --workers / --epochs`
  - Reads `data/processed/{corpus}/texts/{corpus}_{decade}.txt`
  - Saves to `models/raw/{config-name}/{decade}.model`
  - Rich table: docs / tokens / vocab size per decade

### Pending
- [ ] Wait for PubMed preprocessing to finish, then run: `interpretant embed train`
- [ ] Validate vocab sizes are >5K per decade

---

## Stage 3 — Alignment

**Status: implemented, not yet run on real data**

### What's implemented ✅
- `ProcrustesAligner.fit()` — loads gensim models, computes orthogonal Procrustes rotation per decade
- `ProcrustesAligner.align()` — saves aligned `KeyedVectors` to `models/aligned/`
- `ProcrustesAligner.load_aligned()` — loads saved `.kv` files
- `TWECAligner` — stubbed, raises `NotImplementedError` (requires custom gensim fork)
- `align run` CLI — fully wired: `--model-dir`, `--output-dir`, `--reference-decade`

### Pending
- [ ] Run once Stage 2 completes: `interpretant align run`
- [ ] Validate shared vocabulary size >5K words across all decades

---

## Stage 4 — Drift computation

**Status: implemented, not yet run on real data**

### What's implemented ✅
- `cosine_distance(v1, v2)` — 1 − cosine similarity
- `neighborhood_shift(word, vecs_t1, vecs_t2, k=25)` — Jaccard distance on top-k neighbours
- `frequency_corrected_drift(raw_drift, freq_t1, freq_t2)` — down-weights low-frequency noise
- `average_pairwise_distance(vectors)` — trajectory dispersion
- `drift compute` CLI — fully wired: loads aligned models, computes all metrics, writes parquet
  - Output schema: `(word, decade, cosine_drift, neighborhood_shift, avg_pairwise_distance)`
  - `--words` flag for curated word list; defaults to full shared vocab

### Pending
- [ ] Run once Stage 3 completes: `interpretant drift compute`
- [ ] `changepoint.py` is a stub — implement PELT or BOCPD for drift breakpoint detection

---

## Stage 5 — Visualisation & app

**Status: demo works with synthetic data, real-data path implemented but untested**

### What's implemented
- Streamlit dashboard with 4 panels: sidebar, trajectory scatter (PCA), drift line chart, neighbours table ✅
- `build_trajectory_figure()` — Plotly PCA scatter, one point per decade ✅
- `build_neighbors_table()` — DataFrame of top-k neighbours per decade ✅
- `build_heatmap_figure()` — stub ❌
- `--demo` flag loads synthetic vectors seeded for reproducibility ✅
- Real-data path: loads `models/aligned/*.kv` via `ProcrustesAligner.load_aligned()` ✅

### What needs building
- [ ] Test real-data dashboard path end-to-end once Stages 2–4 complete
- [ ] Implement `build_heatmap_figure()` — word × decade drift heatmap
- [ ] Add corpus selector to sidebar (show drift per source or merged)
- [ ] Add decade range slider to sidebar
- [ ] Export button for drift data (CSV)

---

## Infrastructure

| Item | Status |
|------|--------|
| `pyproject.toml` + `uv` | ✅ |
| Ruff linting | ✅ |
| mypy (strict) | ✅ |
| pytest + synthetic fixtures | ✅ |
| Hydra/OmegaConf config tree | ✅ |
| DVC pipeline (`dvc.yaml`) | ✅ defined, not yet run end-to-end |
| Makefile targets | ✅ |
| MkDocs docs structure | ✅ stubs |
| Git repo | ❌ not initialised |

---

## Stage 1b — Books corpus (PDF extraction)

**Status: not started**

A parallel corpus path for book-length texts (philosophy, history of science, etc.) that don't appear in arXiv/ACL/PubMed. PDFs are dropped into an inbox folder and extracted to clean `.txt` files via Docling, then ingested into decade slices alongside the journal corpora.

### Design
- **Primary backend:** Docling (IBM Research, 2024) — ML layout understanding, no Docker required. Separates body text from headers/footers/references natively. Downloads ~500MB models on first run, then fully offline.
- **Fallback backend:** PyMuPDF — fast, no ML, best for clean born-digital PDFs.
- **Folder structure:**
  ```
  data/
  └── inbox/
      ├── pdfs/           # drop PDFs here
      └── processed/      # PDFs move here after extraction
  data/external/books/{decade}/   # output .txt files
  ```

### New module: `src/interpretant/corpus/pdf_extractor.py`
- `PDFBackend` ABC — `extract(pdf_path)`, `is_available()`
- `DoclingBackend` — uses `DocumentConverter`, exports to markdown then strips formatting
- `PyMuPDFBackend` — page-by-page `fitz` extraction
- `PDFExtractor` — selects backend, runs `_postprocess()`, `quality_check()`
  - `_postprocess`: NFKC normalize → remove soft hyphens → rejoin hyphenated line breaks → strip page-number lines → collapse blank lines
  - `quality_check`: returns `word_count`, `avg_line_length`, `short_token_ratio`, `suspected_ocr_errors`, `recommendation` (ok | try_docling | poor)

### CLI commands (`interpretant pdf`)
- `pdf extract <path>` — extract single PDF, optional `--backend`, `--author/--title/--year/--decade/--fields`, `--add-to-manifest`
- `pdf batch --decade <d>` — process all PDFs in inbox, print Rich summary table
- `pdf status` — show inbox and processed folder contents

### Config: `conf/corpus/pdf.yaml`
```yaml
source: pdf
inbox_dir: data/inbox/pdfs
processed_dir: data/inbox/processed
default_backend: docling
output_base_dir: data/external/books
auto_add_to_manifest: false
```

### Dependencies to add
```
docling>=2.0
pymupdf>=1.24
```

### What's implemented ✅
- `docling>=2.0` + `pymupdf>=1.24` added to `pyproject.toml`
- `src/interpretant/corpus/pdf_extractor.py` — `DoclingBackend`, `PyMuPDFBackend`, `PDFExtractor` with postprocessing + quality check
- `pdf extract / batch / status` CLI commands
- `conf/corpus/pdf.yaml`
- `tests/test_pdf_extractor.py` — 17 tests, all passing
- `data/inbox/pdfs/`, `data/inbox/processed/`, `data/external/books/` created
- `src/interpretant/corpus/books.py` — `BooksSource(CorpusSource)` reads extracted `.txt` files per decade dir
- `corpus books ingest / stats` CLI commands
- `BooksSource` exported from `corpus/__init__.py`
- `embed train --corpus books` supported

### Pending
- [ ] README section on PDF extraction
- [ ] Drop real PDFs into inbox and run end-to-end to validate

### Notes
- Docling v2 API: use `DocumentConverter`, call `result.document.export_to_markdown()`. Do NOT use v1 `DoclingDocument` directly.
- First Docling run downloads ~500MB models — communicate clearly via Rich
- After successful extraction, original PDF moves to `data/inbox/processed/`

---

## Immediate next actions (in order)

1. **PubMed preprocessing finishes** (running now, ~300 files, ~25 min) → validate 2000s/2010s doc counts
2. **Run `interpretant embed train`** — merged arxiv + acl + pubmed, 1970–2020
3. **Run `interpretant align run`** — Procrustes alignment, reference decade 2010
4. **Run `interpretant drift compute`** — writes `drift.parquet`
5. **Validate dashboard on real data** — `interpretant app`
6. **Implement heatmap panel** (`build_heatmap_figure`)
7. **Drop PDFs into inbox, run `pdf extract` + `corpus books ingest`** — validate books pipeline end-to-end
8. **Initialise git repo**
