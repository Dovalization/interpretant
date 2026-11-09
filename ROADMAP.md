# interpretant — Roadmap

Diachronic semantic drift tracker for academic corpora.
Tracks how words shift meaning over time across scientific, linguistic, and biomedical discourse.

---

## Pipeline overview

```
corpus ingest → preprocess → embed → align → drift → visualise
```

---

## Stage 0 — Corpus redesign (November 2026)

The November 2026 audit showed that the current corpus cannot test the README's goals:
target words are too rare in most decades, single authors dominate decades, translations
are dated by the original, sources enter partway through the series, and fields are merged
into one model per decade. The rationale and target design are in `docs/corpus.md`.

- [x] `corpus audit` — tokens per cell, target coverage and reliable transitions, author
      concentration, dating risks (`--output` writes JSON)
- [ ] Gate drift on the audit: skip transitions below `--min-occurrences`, show counts in
      the dashboard
- [ ] Compare *network*, *learning* and *representation* against HistWords to see which
      devlog 02 findings survive
- [x] Phrase detection (gensim `Phrases`) before training, so `neural_network` and
      `social_network` get their own trajectories (`corpus phrases learn|apply`)
- [ ] Optional: re-process ACL from raw with a language filter (French papers surfaced in
      the phrase review) if ACL is added back as its own layer
- [x] Manifest: English edition year, translator and original language for every
      translation; misfiled entries moved (`scripts/redate_books.py`; 18 wrong or unreadable
      files replaced, 2 duplicates excluded, 1 removed, 2 pre-1820 books moved to `canon/`)
- [x] Author cap per cell and a leave-one-author-out check (books: 250,000 tokens per
      author per decade; OpenAlex: 1% of works; `corpus books ingest --leave-out-largest-author`)
- [ ] Field × decade grid from 1960: OpenAlex plus PubMed, ACL and arXiv on one taxonomy,
      equal-size systematic samples (replaces the PubMed files lost in June 2026)
      - [x] `corpus openalex download|ingest`: seeded samples of up to 10,000 abstracts per
            cell for 11 fields, 1950s–2020s
      - [x] Decided (November 2026): the main corpus is books plus the OpenAlex grid. The
            grid already covers ACL's, arXiv's and PubMed's fields with one taxonomy and
            one sampling rule; they stay out as optional layers
- [ ] Control words: 15–20 frequency-matched everyday words in `data/control_words.txt`; a
      target drifts only if it moves more than the controls in the same cell
- [ ] TWEC slices per field and decade on a shared compass
- [ ] PPMI-SVD per cell as a count-based robustness check (Church 2017: word2vec approximates
      a PMI factorisation; Hamilton et al. 2016: SVD is more stable on small corpora). A drift
      finding counts only if TWEC and PPMI-SVD agree
- [ ] Cross-perplexity between cells: a Kneser-Ney n-gram model per field × decade, each cell
      scored by every other cell's model. Perplexity is asymmetric, so the field × field
      matrix measures the direction of borrowing, and its change over decades measures
      convergence and divergence
- [ ] Public-domain books before 1930, sampled by Library of Congress class and dated by
      edition
- [ ] Decide how to treat 1930–1960

### After the first results

- [ ] Field separability over time: a TF-IDF and logistic-regression classifier that guesses
      the field from an abstract, per decade. Falling accuracy between two fields measures
      convergence; misclassified abstracts are the borrowing cases to read
- [ ] Sense-level drift with contextual embeddings: encode every occurrence of polysemous
      targets (*network*, *representation*, *interface*), cluster the occurrences into
      senses, and track each sense's share by field and decade (Giulianelli et al. 2020).
      Catches bare uses that phrase detection cannot separate. Runs on the homelab
      RTX 4070 SUPER (12 GB)

---

## Stage 1 — Corpus ingestion & preprocessing

### State on disk (November 2026)

Run `uv run interpretant corpus audit` for current per-cell counts.

| Source | Processed decades | Raw files | Status |
|--------|-------------------|-----------|--------|
| Books | 1820s–2020s | 364 manifest entries, all present | ✅ intact; processed files predate ~90 manifest additions, re-run `corpus books ingest` |
| ACL Anthology | 1960s–2020s | `anthology+abstracts.bib.gz` | ✅ intact |
| arXiv (Kaggle snapshot) | 1990s, 2000s, 2020s | deleted | ⚠️ `arxiv_2010.txt` (72K docs) deleted June 2026 |
| PubMed baseline | 1960s only | deleted | ❌ 1970s–2020s processed files deleted June 2026; replaced by the Stage 0 field grid |

### Known data quality notes
- arXiv: clean, no short lines, 25 duplicates in 72K (negligible)
- ACL: ~5% title-only docs (pre-2000, no abstract), ~500 duplicates in 1990s — acceptable
- PubMed: proceedings/venue entries filtered; accent artifacts (`{\'e}`) fixed
- PubMed decade coverage is driven by PMID ordering — n0001–n0100 = pre-1990, n0800+ = 2000s, n1100+ = 2010s+

### Pending
- [x] ~~PubMed n0800/n1100 downloads and preprocessing~~ — done in March 2026, then deleted;
      superseded by the Stage 0 field grid (the baseline file numbering changes yearly, so
      the ranges above no longer map to decades)
- [x] Re-run `corpus books ingest` so processed books match the manifest (42.3M tokens)

---

## Stage 2 — Embedding training

**Status: ran in March 2026 with TWEC (see Stage 3); models partly deleted, retrain after Stage 0**

Surviving TWEC models in `models/twec/`: 1820s–1970s and 2020s. The compass and the
1980s–2010s decade models lost their `.vectors.npy` files in June 2026 and cannot be loaded.

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
- [x] ~~Run `interpretant embed train`~~ — superseded by `twec train` (March 2026)
- [x] ~~Validate vocab sizes are >5K per decade~~ — 108K–196K per decade from the 1960s
- [ ] Retrain after Stage 0, with phrase detection and per-field slices

---

## Stage 3 — Alignment

**Status: TWEC in use; Procrustes kept as the fallback**

TWEC is implemented as a trainer, not an aligner: `embedding/twec.py` and the `twec train`
command train every decade against a shared compass, so decade vectors share one coordinate
system and need no post-hoc alignment. It runs on standard gensim, so the `TWECAligner`
stub in `alignment/twec.py` (which asks for a custom gensim fork) is obsolete.

### What's implemented ✅
- `ProcrustesAligner.fit()` — loads gensim models, computes orthogonal Procrustes rotation per decade
- `ProcrustesAligner.align()` — saves aligned `KeyedVectors` to `models/aligned/`
- `ProcrustesAligner.load_aligned()` — loads saved `.kv` files
- `TWECAligner` — stubbed, raises `NotImplementedError` (requires custom gensim fork)
- `align run` CLI — fully wired: `--model-dir`, `--output-dir`, `--reference-decade`

### Pending
- [x] ~~Run `interpretant align run`~~ — not needed with TWEC
- [ ] Remove the obsolete `TWECAligner` stub

---

## Stage 4 — Drift computation

**Status: ran in March 2026 on the TWEC models — `data/processed/drift.parquet`, 592,584 rows**

Since then: per-pair vocabularies instead of a global intersection, neighbourhood `k`
lowered to 10 (`--neighborhood-k`), and chunked target-row similarity to bound memory.
The results predate the corpus audit: many transitions fall below the reliability threshold.

### What's implemented ✅
- `cosine_distance(v1, v2)` — 1 − cosine similarity
- `neighborhood_shift(word, vecs_t1, vecs_t2, k=25)` — Jaccard distance on top-k neighbours
- `frequency_corrected_drift(raw_drift, freq_t1, freq_t2)` — down-weights low-frequency noise
- `average_pairwise_distance(vectors)` — trajectory dispersion
- `drift compute` CLI — fully wired: loads aligned models, computes all metrics, writes parquet
  - Output schema: `(word, decade, cosine_drift, neighborhood_shift, avg_pairwise_distance)`
  - `--words` flag for curated word list; defaults to full shared vocab

### Pending
- [x] ~~Run `interpretant drift compute`~~ — done March 2026
- [ ] Skip transitions below the audit threshold (Stage 0)
- [ ] `changepoint.py` is a stub — implement PELT or BOCPD for drift breakpoint detection

---

## Stage 5 — Visualisation & app

**Status: demo works with synthetic data; the real-data path reads Procrustes output
(`models/aligned/*.kv`), not the TWEC models, and is untested**

The plots used in the devlogs come from standalone scripts in `scripts/` (`plot_drift.py`,
`plot_fields.py`, `plot_word.py`), not from the dashboard.

### What's implemented
- Streamlit dashboard with 4 panels: sidebar, trajectory scatter (PCA), drift line chart, neighbours table ✅
- `build_trajectory_figure()` — Plotly PCA scatter, one point per decade ✅
- `build_neighbors_table()` — DataFrame of top-k neighbours per decade ✅
- `build_heatmap_figure()` — stub ❌
- `--demo` flag loads synthetic vectors seeded for reproducibility ✅
- Real-data path: loads `models/aligned/*.kv` via `ProcrustesAligner.load_aligned()` ✅

### What needs building
- [ ] Point the real-data path at the TWEC models and test it end to end
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
| DVC pipeline (`dvc.yaml`) | ✅ defined, not yet run end-to-end; does not include the `twec train` stage |
| Makefile targets | ✅ |
| MkDocs docs structure | ✅ `corpus.md` written (Nov 2026); other pages are stubs |
| Git repo | ✅ initialised March 2026 |

---

## Stage 1b — Books corpus (PDF extraction)

**Status: implemented and in use — 364 manifest entries across 1820s–2020s**

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
- [x] ~~Drop real PDFs into inbox and run end-to-end to validate~~ — done; the books layer
      was built this way (Docling handled scanned books such as Hansen)

### Notes
- Docling v2 API: use `DocumentConverter`, call `result.document.export_to_markdown()`. Do NOT use v1 `DoclingDocument` directly.
- First Docling run downloads ~500MB models — communicate clearly via Rich
- After successful extraction, original PDF moves to `data/inbox/processed/`

---

## Immediate next actions (in order)

The March 2026 list (PubMed preprocessing, train, align, drift, books validation, git) is
done. The work now follows Stage 0:

1. **Gate drift on the audit** — `drift compute` skips transitions below `--min-occurrences`
2. **HistWords comparison** — check *network*, *learning* and *representation* against
   general English to see which devlog 02 findings survive
3. **Phrase detection** before training
4. **Manifest dating** — English edition, translator, original language; fix misfiled entries
5. **Field × decade grid from 1960** — OpenAlex, PubMed, ACL, arXiv
6. **Retrain TWEC** with per-field slices, then recompute drift
7. **Dashboard on TWEC output**, heatmap panel
