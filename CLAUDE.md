# CLAUDE.md

Guidance for Claude Code in this repository.

## Commands

```bash
uv sync --all-extras         # install, with dev extras
uv run pre-commit install    # once: run every check before each commit
make check                   # every check, as pre-commit and CI run them

make lint                    # ruff check + ruff format --check (src/, tests/)
make fmt                     # ruff format + ruff check --fix
make type-check              # mypy src/ (strict, zero errors; CI enforces it)
make test                    # pytest with coverage
uv run pytest tests/test_drift_grid.py -v     # one file
uv run pytest tests/ -k "neighbour" -v        # one test by name

```

## Pipeline

The pre-rebuild pipeline, run from the CLI in this order. Every command has `--help`. The
rebuild replaces it phase by phase (`docs/decision_agenda.md`, steps 2–9).

```
corpus openalex download|ingest                         data/processed/<layer>/texts/<cell>.txt
corpus lemmas votes|learn|apply                         data/lemmatised/<layer>/texts/<cell>.txt
corpus phrases learn|apply                              data/phrased/<layer>/texts/<cell>.txt
corpus audit --processed-dir data/phrased --output data/phrased/audit.json
twec grid [--seed 43 --output-dir models/twec_grid_seed43]   models/twec_grid/<layer>/<field>_<decade>.kv
ppmi grid                                               models/ppmi_grid/...
drift grid                                              data/drift/within_field.parquet, cross_field.parquet
```

A cell is one `<field>_<decade>` text file: one document per line, whitespace-tokenised,
phrases joined with `_`. Layers: `openalex` (the equal-size grid) and `openalex_full` (whole
fields, within-field trajectories only). The `books` layer was retired on 2026-11-11.

## Pre-registration

Predictions are written to `docs/predictions.md` or `docs/predictions_cross_perplexity.md`
and **committed before the code that tests them is run or the evidence is inspected**. A
scorer in `scripts/score_*.py` is committed next, then run; its verdicts are appended under the
predictions, which are never edited after the fact. A falsified prediction stays falsified;
a corrected interpretation is written as a new section, not an edit. Follow this for any new
analysis: predictions first, in their own commit.

## Architecture

`src/interpretant/`

- `__main__.py`: the Click CLI (`corpus`, `twec`, `ppmi`, `drift grid`). Commands parse options,
  call one library function and print; the loops live in the library with tests.
- `catalog/`: `catalog.yaml`'s reader (`read_catalog`), the raw OpenAlex cells (`raw.py`), the
  abstracts table's pandera schema and its parquet parts, the curated cleaning rules and spaCy
  patterns (`load_cleaning_patterns`), and the annotation cache (DocBin). It imports no other
  interpretant package.
- `corpus/` (rebuild): `repair.py` (text repair before spaCy, mojibake detection),
  `annotation.py` (the spaCy pass: sentences, labelled removal spans, the pass's keeping rules,
  one abstracts-table row per work), `language.py` (lingua's sentence languages, stored on the
  spans), `deduplication.py` (one row per work across cells; MinHash near-duplicates) and
  `cleaning.py` (template sentences, repeated sentences, the author cap).
- `corpus/` (old pipeline, replaced as the rebuild lands): `openalex.py` (API client, sampling,
  cleaning at ingest, `ingest_fields`), `files.py` (`source_texts`, `rewrite_texts`: one stage's
  cell files into the next), `caps.py` (author caps), `lemmas.py` (type-level map: *-s*
  inflections from spaCy's readings in context, and British spellings), `phrases.py` (gensim
  `Phrases` on a balanced sample), `preprocessing.py` (`preprocess_text`, target variant map),
  `audit.py` (cell counts and target coverage; `audit_corpus`).
- `embedding/`: `base.py` (`training_sentences`: splits lines at 10,000 tokens, gensim's
  silent truncation limit), `twec.py` (`TWECTrainer`: balanced compass, frozen context
  vectors per cell), `grid.py` (cell discovery, compass sample, `train_twec_grid`,
  `build_ppmi_grid`), `ppmi.py` (PPMI-SVD with Levy et al. 2015 settings).
- `drift/`: `grid.py` (`compare_cells`, `add_noise_floor`, `cell_neighbours`, pair
  enumeration, `measure_grid`; the constants `MEASURE_MIN`, `NEIGHBOUR_MIN`, `NEIGHBOURS`,
  `FLOOR_QUANTILE`), `metrics.py` (cosine distance, batched neighbourhood shift),
  `senses.py` (the cased abstracts behind the cells and seeded samples of a word's occurrences,
  used by the contextual scripts), `pairs.py` (XL-LEXEME encoding, APD, J and PRT), `labels.py`
  and `classifier.py` (sense labelling).
- `predictions.py`: `report`, shared by the remaining scorers.

`scripts/`: the contextual scripts still in use until the rebuild replaces them, and
`scripts/evidence/` (the scripts behind the decision agenda's numbers, kept as a record). Every
past experiment is recorded in `docs/experiments.md`; deleted scorers stay in git.

## Data and models

`data/` and `models/` are local; only `data/words_of_interest.txt`, the control-word files
and the round data under `data/labels/` and `data/diagnostics/` are tracked. Human judgements
live in `curated/` (tracked). Old results are set aside under `data/retired/`. `.env` holds
`OPENALEX_API_KEY`.

## Tests

Synthetic only: no corpus files, no network, and no real models except the ones installed as
pinned dependencies and run offline (`en_core_web_sm` and lingua's language models, which
`tests/test_annotation.py` loads once per session). `tests/test_pipeline.py` runs the whole CLI,
ingest to drift grid, on a synthetic two-field, two-decade grid in a few seconds.

## Code style

Follow `docs/code_standards.md`: self-documenting code first (comments say why, never what),
names and verb prefixes with fixed meanings, keyword-only booleans, docstrings on public code, and the
enforcement table (pre-commit and CI run every automatic check). Also:

- Line length 100 (`.ruff.toml`); mypy strict; `# type: ignore[code]` with a specific code.
- Long `@click.option` lines take `# noqa: E501`; functions used with `ProcessPoolExecutor`
  are module-level for pickling on macOS.
- British spelling in prose and identifiers (`neighbour`), American where a library fixes it.
