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

What remains of the pre-rebuild pipeline, run from the CLI. Every command has `--help`. The
commands that wrote the cell text files (`corpus openalex ingest`, `corpus lemmas`,
`corpus phrases`, `corpus audit`) retired on 2026-11-17; the grid commands read the cells
already on disk (`data/phrased` by default). The rebuild replaces the rest phase by phase
(`docs/decision_agenda.md`, steps 2–9).

```
corpus openalex download                                data/raw/openalex/<field>/<decade>.jsonl.gz
corpus openalex sources                                 data/raw/openalex/<field>/sources_<decade>.jsonl
twec grid [--seed 43 --output-dir models/twec_grid_seed43]   models/twec_grid/<layer>/<field>_<decade>.kv
ppmi grid                                               models/ppmi_grid/...
drift grid                                              data/drift/within_field.parquet, cross_field.parquet
```

A cell is one `<field>_<decade>` text file: one document per line, whitespace-tokenised,
phrases joined with `_`. Layers: `openalex` (the equal-size grid) and `openalex_full` (whole
fields, within-field trajectories only). The `books` layer was retired on 2026-11-11.

The rebuild's stages so far, with their settings in `params.yaml` and their data in
`catalog.yaml`:

```
fetch-metadata [--field ai --decade 2010 --workers 8]       data/raw/openalex_metadata/<field>/<decade>.jsonl.gz
annotate-corpus [--field ai --decade 2010 --process-count 10]
                                                             data/interim/annotations/<part>.spacy,
                                                             data/interim/annotated_abstracts/<part>.parquet
build-corpus                                                 data/processed/abstracts/<part>.parquet
report-corpus                                                data/processed/corpus_report.md
```

`fetch-metadata` is not a stage: like the downloads it runs by hand once (resuming where it
stopped) and its output is registered. `annotate-corpus` (stage 2, the slow spaCy pass) refuses
cells whose metadata is missing and may annotate some parts only; `build-corpus` (stage 3, the
corpus-level rules and review flags, minutes) always reads every part of the corpus in
`params.yaml`'s `corpus` section, so a threshold change reruns only stage 3. `report-corpus`
writes the corpus report the author reads before the review (4b, layer 2).

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
- `cli/` and `pipelines/` (rebuild): one thin command per pipeline (`fetch-metadata`,
  `annotate-corpus`, `build-corpus`, `report-corpus`), and the pipelines composing the jobs,
  each with its pydantic settings object reading a section of `params.yaml`
  (`pipelines/params.py`; `.env` for the OpenAlex key); `pipelines/corpus.py` holds what the
  two corpus stages share.
  Import contracts: `__main__` → `orchestration` → `cli` → `pipelines` → `review` → `corpus` →
  `catalog`; only `orchestration` may import Prefect.
  Each job's command module has a `run_<job>` function doing its work (options to pipeline call,
  printed result, counts for the run log); the command runs it through `cli/jobs.py`
  (`run_on_command_line`) and a Prefect flow runs the same function.
- `orchestration/`: the long jobs as Prefect flows, one per job (`flows.py`, deployed by
  `prefect.yaml` on the runner and locally; each flow's parameters are its command's options),
  each job's limits, timeout, retries and DVC stage (`registry.py`, Prefect-free), running a job's
  work under its limits with its output in the log file and the Prefect run (`jobs.py`), the
  shared steps (`steps.py`: the run's log file, `dvc pull`/`commit`/`push`, the record commit
  under a git lock), progress published as an artifact (`progress.py`), and
  `interpretant submit <job>` / `interpretant runs` (`submit.py`, `commands.py`). Adding a job:
  its `run_<job>` function, a registry entry, a flow, two deployments and its command in
  `commands.JOB_COMMANDS`; `tests/test_orchestration_jobs.py` checks they agree. Needs the
  `orchestration` extra.
- `review/` (rebuild): the corpus review (4b): `corpus_report.py` (what each rule dropped,
  removed and flagged, by decade, field and repository; rendered as Markdown by Polars), and
  `judging.py` (review items as the judges see them: the answer schemas, the rendered items,
  parts of 50). The judges themselves are `pipelines/judge_api.py` (Message Batches API, within
  a dollar cap; the key from the checkout's secrets file only) and
  `pipelines/judge_claude_code.py` (Claude Code's command line on the subscription, never given
  the API key), run as `judge-api` and `judge-claude-code`.
- `catalog/`: `catalog.yaml`'s reader (`read_catalog`), the raw OpenAlex cells (`raw.py`) and
  each raw work's OpenAlex metadata (`metadata.py`), the
  abstracts table's pandera schema and its parquet parts, the curated cleaning rules and spaCy
  patterns (`load_cleaning_patterns`), and the annotation cache (DocBin). It imports no other
  interpretant package.
- `corpus/` (rebuild): `repair.py` (text repair before spaCy, mojibake detection),
  `annotation.py` (the spaCy pass: sentences, labelled removal spans, the pass's keeping rules,
  one abstracts-table row per work), `language.py` (lingua as the pipeline's last component,
  setting each sentence's language on its span), `deduplication.py` (one row per work across
  cells; MinHash near-duplicates), `cleaning.py` (template sentences, repeated sentences, the
  author cap) and `work_metadata.py` (OpenAlex's record of each work, fetched by id with pyalex;
  the repository-only flag).
- `corpus/` (old pipeline, replaced as the rebuild lands): `openalex.py` (API client, sampling,
  `OpenAlexSource`, which cleans a cell as it loads it), `files.py` (`source_texts`: a stage's
  cell files), `caps.py` (author caps), `lemmas.py` (the voted lemma rules: `voted_lemma` for
  *-s* inflections from spaCy's readings in context, `american` for British spellings; tools for
  finding spaCy's errors and the spelling map's backstop), `preprocessing.py` (`preprocess_text`,
  target variant map), `audit.py` (`load_target_words`). The type-level map, the phrase model,
  the audit and ingest retired with their commands on 2026-11-17.
- `embedding/`: `base.py` (`training_sentences`: splits lines at 10,000 tokens, gensim's
  silent truncation limit), `twec.py` (`TWECTrainer`: balanced compass, frozen context
  vectors per cell), `grid.py` (cell discovery, the compass sample (`sample_chunks`),
  `train_twec_grid`, `build_ppmi_grid`), `ppmi.py` (PPMI-SVD with Levy et al. 2015 settings).
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
`tests/test_annotation.py` loads once per session). `tests/test_pipeline.py` runs the grid
commands (`twec grid`, `ppmi grid`, `drift grid`) on a synthetic two-field, two-decade grid of
lemmatised cells in a few seconds.

## Code style

Follow `docs/code_standards.md`: self-documenting code first (comments say why, never what),
names and verb prefixes with fixed meanings, keyword-only booleans, docstrings on public code, and the
enforcement table (pre-commit and CI run every automatic check). Also:

- Line length 100 (`.ruff.toml`); mypy strict; `# type: ignore[code]` with a specific code.
- Long `@click.option` lines take `# noqa: E501`; functions used with `ProcessPoolExecutor`
  are module-level for pickling on macOS.
- British spelling in prose and identifiers (`neighbour`), American where a library fixes it.
