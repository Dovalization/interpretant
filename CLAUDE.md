# CLAUDE.md

Guidance for Claude Code in this repository.

## Commands

```bash
uv sync --all-extras         # install, with dev extras

make lint                    # ruff check + ruff format --check (src/, tests/)
make fmt                     # ruff format + ruff check --fix
make type-check              # mypy src/ (strict, zero errors; CI enforces it)
make test                    # pytest with coverage
uv run pytest tests/test_drift_grid.py -v     # one file
uv run pytest tests/ -k "neighbour" -v        # one test by name

make figures                 # redraw figures/*.png from the models and drift tables
make app-demo                # dashboard on synthetic data
make app                     # dashboard on models/twec_grid
```

## Web explorer (`web/`)

A static Next.js app (pnpm, TypeScript strict, Tailwind 4, shadcn base-nova on Base UI,
recharts, nuqs) that reads the JSON written by `uv run python -m interpretant.export web/data`
(`src/interpretant/export.py`; `export web` joins the CLI after the lemmatised run). The JSON is
tracked so the site builds without Python.

```bash
cd web && pnpm install
pnpm dev                     # local dev server
pnpm lint && pnpm typecheck && pnpm test && pnpm build   # the CI gates; build writes out/
```

Conventions follow axm: `eslint.config.js` bans `any`, `!` and `let`; imports sorted; UI
primitives in `src/components/ui`; pages under `src/app`; pure data helpers in `src/lib` with
vitest tests next to them; chart colours come from `--chart-N` tokens, one per field, never
cycled (`src/lib/fields.ts`).

## Pipeline

One pipeline, run from the CLI in this order. Every command has `--help`.

```
corpus books ingest / corpus openalex download|ingest   data/processed/<layer>/texts/<cell>.txt
corpus lemmas learn|apply                               data/lemmatised/<layer>/texts/<cell>.txt
corpus phrases learn|apply                              data/phrased/<layer>/texts/<cell>.txt
corpus audit --processed-dir data/phrased --output data/phrased/audit.json
twec grid [--seed 43 --output-dir models/twec_grid_seed43]   models/twec_grid/<layer>/<field>_<decade>.kv
ppmi grid                                               models/ppmi_grid/...
drift grid                                              data/drift/within_field.parquet, cross_field.parquet
drift perplexity --drop-shared --fixed-budget 200000 --seed 1 ... --seed 5
figures                                                 figures/*.png
```

A cell is one `<field>_<decade>` text file: one document per line, whitespace-tokenised,
phrases joined with `_`. Layers: `books`, `openalex` (the equal-size grid), `openalex_full`
(whole fields, within-field trajectories only).

## Pre-registration

Predictions are written to `docs/predictions.md` or `docs/predictions_cross_perplexity.md`
and **committed before the code that tests them is run or the evidence is inspected**. A
scorer in `scripts/score_*.py` is committed next, then run; its verdicts are appended under the
predictions, which are never edited after the fact. A falsified prediction stays falsified;
a corrected interpretation is written as a new section, not an edit. Follow this for any new
analysis: predictions first, in their own commit.

## Architecture

`src/interpretant/`

- `__main__.py`: the Click CLI (`corpus`, `twec`, `ppmi`, `drift`, `figures`, `pdf`, `app`).
  Orchestration still lives here; the library modules are pure functions and small classes.
- `corpus/`: `base.py` (`CorpusSource` ABC, `iter_decade_slices`), `books.py` (manifest +
  extracted texts, author cap), `openalex.py` (API client, sampling, cleaning at ingest),
  `caps.py` (author caps), `lemmas.py` (type-level plural and spelling map), `phrases.py`
  (gensim `Phrases` on a balanced sample),
  `preprocessing.py` (`preprocess_text`, target variant map), `audit.py` (cell counts, target
  coverage, manifest risks), `pdf_extractor.py` (Docling with PyMuPDF fallback).
- `embedding/`: `base.py` (`training_sentences`: splits lines at 10,000 tokens, gensim's
  silent truncation limit), `twec.py` (`TWECTrainer`: balanced compass, frozen context
  vectors per cell), `grid.py` (cell discovery, compass sample), `ppmi.py` (PPMI-SVD with
  Levy et al. 2015 settings).
- `drift/`: `grid.py` (`compare_cells`, `add_noise_floor`, `cell_neighbours`, pair
  enumeration; the constants `MEASURE_MIN`, `NEIGHBOUR_MIN`, `NEIGHBOURS`, `FLOOR_QUANTILE`),
  `metrics.py` (cosine distance, batched neighbourhood shift), `perplexity.py` (Kneser-Ney
  bigrams, shared-abstract removal, fixed training budget, excess cross-entropy).
- `visualization/`: `figures.py` (the README figures), `style.py` (palette, fonts),
  `trajectory.py` and `neighbors.py` (dashboard panels).
- `app/main.py`: Streamlit dashboard; demo mode is synthetic, real mode reads
  `models/twec_grid` for one field, restricted to `data/words_of_interest.txt`.

`scripts/`: `redate_books.py` (the record of every book edition decision), `fetch_*.py`
(book acquisition), `select_*_controls.py` (control words), `score_*.py` (one per prediction
round; frozen once run).

## Data and models

`data/` and `models/` are local; only `data/words_of_interest.txt`, the control-word files
and the books manifest are tracked. Book source files are copyrighted and never committed,
nor are their filenames. `.env` holds `OPENALEX_API_KEY`.

## Tests

Synthetic only: no real models, no corpus files, no network. Figures are tested on tiny
`KeyedVectors` and DataFrames built in the test.

## Code style

- Line length 100 (`.ruff.toml`); mypy strict; `# type: ignore[code]` with a specific code.
- Long `@click.option` lines take `# noqa: E501`; functions used with `ProcessPoolExecutor`
  are module-level for pickling on macOS.
- British spelling in prose and identifiers (`neighbour`), American where a library fixes it.
