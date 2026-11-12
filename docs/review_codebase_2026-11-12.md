# Codebase QA report: interpretant, 2026-11-12

*Suggestions only; nothing in this report has been applied. Merged from a style review (naming,
dead code, structure, duplication) and an architecture review (layers, coupling, single
responsibility), both read-only, run after the night's Polars port and the removal of the old
scorers. Line numbers are from the tree at commit 438585d.*

## Executive summary

**Overall: Good.** No import cycles, the package layers (corpus → embedding → drift → export and
visualization → CLI) mostly hold, and every module type-checks strictly. The problems are where
code lives, not how it is written: corpus cleaning sits in `drift/`, the usage pipeline (usage
identity, store, encoders) is spread over `drift/` and ten scripts, and the CLI file has grown to
1,251 lines.

| Domain | Critical | Moderate | Minor |
|---|---|---|---|
| Style (naming, dead code, structure, readability) | 3 | 20 | 9 |
| Architecture (layers, coupling, responsibilities) | 2 | 9 | 4 |

The domains overlap: findings flagged by both reviews are merged below and counted once, in
the domain where they mainly belong.

**Top priority:** give the usage pipeline one home (`src/interpretant/usages/`) before the store
rebuild, since five scripts each define their own usage key today.

## Architecture

### Layer violations

- **Corpus cleaning lives in `drift/` (critical).** `drift/senses.py` holds `cased_abstracts`
  (:87), `drop_shared` (:100), `foreign_share` (:81) and the URL and label regexes;
  `drift/classifier.py:159–183` holds `tidy_text`, `tidy_usage` and `boilerplate`. The corpus
  command `corpus lemmas votes` imports `drift.senses` (`__main__.py:323`), and
  `scripts/build_vector_store.py:33–36` draws one pipeline from three modules. Cleaning is now
  spread over four modules, with three language checks (`openalex.looks_english`,
  `quality.english_share`, `senses.foreign_share`). *Fix:* move the reading functions into
  `corpus/openalex.py` and the cleaning rules into `corpus/quality.py` (or a new
  `corpus/cleaning.py`).
- **`export.py` imports matplotlib to get a dictionary (moderate).** `export.py:25` imports
  `FIELD_LABELS` from `visualization/figures.py`; the 11 field slugs are also listed in
  `corpus/openalex.py:189` (`FIELDS`). *Fix:* keep `FIELD_LABELS`, `field_label` and
  `decade_label` next to `FIELDS` in the corpus layer.
- **No import contract (minor).** *Fix:* once the two moves above are done, one import-linter
  `layers` contract keeps them done.

### Coupling

- **Usage identity and store layout are copied across scripts (critical; blocks the store
  rebuild).** `key_of(text, start, end)` is defined five times (`build_vector_store.py:60`,
  `score_labels_known.py:159`, `score_labels_head_to_head.py:67`, `score_labels_corpus.py:63`,
  `review/build.py:71`), and `pilot_labellers.py:64` has a second scheme (`usage_id`). The store
  layout is written in `build_vector_store.store_paths` (:149) and read back separately in
  `dev_harness.py:45–59` and `finetune_classifier.py:28–43`. Reading a manifest, tidying and
  dropping boilerplate is repeated in four scripts. *Fix:* a `usages/` package (last section).
- **Scripts used as libraries through `sys.path.insert` (moderate).** `pilot_labellers` is
  imported by four scripts, `score_labels_benchmark` (as `round_y`) by three,
  `check_known_presence` by two; `pilot_labellers.jev_questions` copies
  `JevLabeller.questions` (`labels.py:427`). *Fix:* move what live work uses (`usage_id`,
  `key_rows`, `usages`, the DWUG `clusters` and `divergences`) into the library.
- **Cell discovery and reading reimplemented about six times (moderate).** Raw cells are globbed
  at `__main__.py:328`, `corpus/quality.py:103` and `build_vector_store.py:72`, and read at
  `openalex.py:438, 486`, `senses.py:91`, `quality.py:104` and `build_vector_store.py:65`;
  `<field>_<decade>` stems are parsed in six places. *Fix:* `corpus/files.py` owns
  `raw_cells`, `read_raw_cell` and `parse_cell_stem`.
- **Neighbours computed twice (moderate).** `export.NeighbourIndex` (`export.py:73–98`)
  re-implements `drift/grid.cell_neighbours` (:145), which the figures use, so the explorer and
  the README figures can quietly disagree. *Fix:* one implementation in `drift/grid.py`.
- **`perplexity.py` is a hub for sampling helpers (minor).** `read_cell`, `training_sample` and
  `remove_shared` are imported by separability, held-out and direction; `remove_shared` and
  `senses.drop_shared` are one rule keyed differently. *Fix:* a `drift/samples.py`.

### Single responsibility

- **`drift/senses.py` does five jobs (moderate):** raw reading, quality filtering, occurrence
  sampling, encoding and clustering. Eight scripts import only the sampling half. *Fix:* reading
  to corpus, sampling to `usages/`, clustering and analysis stay.
- **`corpus/lemmas.py` does three (moderate):** the spaCy tagging pass, about 150 lines of
  British-spelling tables and rules, and the map itself. Its top-level `import spacy` makes every
  `LemmaMap` user (`score_venue.py`, `measure_label_noise.py`, `corpus lemmas apply`) load spaCy.
  `_lemmatizer` hardcodes the small model while the votes accept `--model`. *Fix:*
  `corpus/spelling.py`, `corpus/annotate.py` (the spaCy pass) and `lemmas.py` (the map, spaCy
  imported lazily).
- **`corpus/openalex.py` mixes cleaning rules, the HTTP client, downloads and ingest
  (moderate).** `cased_abstracts` also reads every cell twice and runs the whole cleaning pass to
  recover kept ids. *Fix:* rules to `corpus/cleaning.py`; `clean_cell` keeps ids beside the raw
  records so cased reading is one pass.
- **`drift/labels.py` (494 lines) holds four labellers' worth of code (moderate).** *Fix:* a
  package: `senses.py` (lists, items), `claude_cli.py`, `batches.py`, `jev.py`.
- **Logic in CLI commands (moderate).** `corpus lemmas votes` (`__main__.py:325–343`, the
  de-duplication closure), `phrases learn` (:238–241, the review file) and `lemmas learn`
  (:427–438, the per-rule counts). *Fix:* `unique_cased_abstracts(raw_dirs)`,
  `PhraseModel.write_review`, `LemmaMap.rule_counts`.
- **Encoder code duplicated (moderate).** Torch device choice is written four times, and
  `senses.py:262` and `induction.py:76` never choose CUDA; padding and batching is duplicated in
  `XLLexeme` and `TargetTokenEncoder`; `build_vector_store._prepare` builds a throwaway class to
  call a method unbound and switches on the encoder kind in three places. *Fix:*
  `torch_device()`, `padded_batches()`, a module-level `target_token_ids`, and one table from
  encoder kind to model, revision and preparation.
- **`export_web` writes six files in one 160-line function (minor),** and `export.py:329` has a
  second entry point duplicating the CLI's defaults. *Fix:* one function per file; delete the
  `__main__` block.

## Style and conventions

### Code structure

- **`__main__.py`, 1,251 lines (critical).** *Fix:* a `cli/` package: `corpus.py`
  (lines 26–575), `embedding.py` (618–753), `drift.py` (761–1143), `outputs.py` (export and
  figures), `options.py` (shared options and a `write_table` helper); move
  `_configure_gensim_logging` (583) into `embedding/twec.py`.
- **`visualization/figures.py`, 551 lines (moderate).** *Fix:* `same_family` and
  `distinct_neighbours` (62–87) filter neighbours, so they belong in `drift/grid.py`;
  `readme_figures` (485–550) can be `visualization/readme.py`.

### Duplicated logic (three or more real occurrences)

| What | Where | Extract as |
|---|---|---|
| Usage from a row, `Usage(str(row["text"]), int(row["start"]), int(row["end"]))` | head_to_head:148, score_labels_known:196, finetune_classifier:33, dev_harness:106, score_labels_corpus:139, 182 | `Usage.from_row`, which also removes six `type: ignore` |
| Cue lists read from the sense-list TOML | head_to_head:153, score_labels_corpus:158, score_labels_known:187, finetune_classifier:53, dev_harness:218, review/build.py:78 | `load_cue_lists` beside `load_sense_lists` |
| Label tables read and written | benchmark, head_to_head, induced, wordnet, pilot, classifier scripts | `read_label_table`, `write_label_table` in `drift/labels` |
| Repeat consistency and repeat items | benchmark, wordnet, induced, pilot scripts | `repeat_consistency`, `with_repeats` beside `REPEAT_SUFFIX` |
| Squared Jensen-Shannon divergence | senses.py:406, labels.py:243, score_labels_known.py:457 | one `jensen_shannon_bits` in `drift/metrics.py` |
| Unit-normalised rows | grid.py:99, export.py:85, classifier.py:66, metrics.py:50; without a zero-norm guard at senses.py:308, 356 and pairs.py:278, 285 | one `unit_rows` in `drift/metrics.py` |
| No-op logger `_ignore` | heldout, embedding/grid, direction, perplexity, separability; senses and export default to `print` | one logging type and silent default |
| Word lists read by hand | build_vector_store:53, score_labels_known:71, score_pairs_known:36, longlist:35 | `corpus/audit.load_target_words`, moved to `corpus/files` |
| Default paths and CLI options | `data/raw/openalex` as `GRID` in eight scripts and five CLI defaults; `--words-file` declared six times | shared constants and reusable `click` options |
| Cell model and word count | predictions.py:31, 36; figures.py:490; discovery scripts | `load_vectors` and `word_count` in `embedding/grid.py` |

### Naming

- **Critical: `scripts/discovery/*.py`** are written in single letters and abbreviations (`D`,
  `M`, `kv()`, `nb()`, `cosd`, `jac`, `ok()`), and point at `models/twec_grid_unlemmatised`.
  *Fix:* rename, or delete the folder if the exploration is over.
- **Critical: `drift/metrics.py:85–112`** (`idx_t1`, `i1`, `sim_t1`, `top1`) and American
  spelling (`batch_neighborhood_shift`, `_top_k_neighbors`) in a British codebase; its one caller
  (grid.py:130) passes the same word list twice. *Fix:* `neighbour_shift(targets, vocabulary,
  unit_a, unit_b, k)`.
- **One concept, three names:** a cell's field is `name` (`embedding/grid.Cell`,
  `drift/grid.CellModel`), `source` (`audit.CellStats`, `figures.corpus_grid`) and `field`
  elsewhere. *Fix:* `field` everywhere.
- **One concept, two names; one name, two meanings:** `remove_shared` (perplexity) and
  `drop_shared` (senses) do one job, and `drop_shared` is also a boolean parameter;
  `distribution` means different things in `labels.py:236` and `senses.py:400`.
- **`drift/labels.py`:** `shown` → `marked_context`; `parsed` → `parse_block_labels`;
  `prompt`/`schema` against `single_prompt`/`label_schema` → `block_prompt`/`block_schema` and
  `single_prompt`/`single_schema`; inner `one` → `label_item`.
- **`corpus/lemmas.py`:** `american` → `american_spelling`; `regular_plural` →
  `is_regular_plural`; `MIN_SHARE` is easy to confuse with `MIN_VOTE_SHARE`; a word type is
  called `token`, `form`, `word` and `type`; two shapes are both "votes" (`Votes` and the counter).
- **Module aliases named after rounds** (`round_y`, `round_q`, `check_1a`) hide what they hold.
- **Minor:** `Sampling`'s booleans `replace`, `plural`, `quality` → `with_replacement`,
  `include_plural`, `apply_quality_rules`.

### Dead code

Some of this became dead tonight, when the old scorers and `essay_figures.py` were removed.

- **`drift/direction.py:94–315` (moderate):** `FieldCosts`, `ClassifierCosts`,
  `LanguageModelCosts`, `cost_rows`, `frame_runs`, `planted_runs`, `cost_matrix`, `contrast`,
  `movement`, `relative_movement`, `share`, `gain`, `relative_gain` and `PLANTED_SHARE`: about 220
  lines called only by `tests/test_direction.py`. The CLI uses only `assignment_runs`.
- **`visualization/figures.py:358–482` (moderate):** `known_answer_ranks`, `ReadingPanel` and
  `readings`, used only by tests since `essay_figures.py` went.
- **`drift/metrics.py` (moderate):** `_top_k_neighbors` (25) and `build_decade_index` (38) have
  no callers.
- **`predictions.py:28–41` (moderate):** `load_vectors`, `word_count` and `overlap` are
  test-only since the scorers went; figures re-implements the first two inline.
- **Minor:** `export.METHODS` (28) is unused; `corpus/__init__.py` re-exports `OpenAlexSource`,
  which nothing imports through the package; 95 `# noqa: PLR0913/PLC0415/S310` comments (47 in
  `__main__.py`) suppress rules that ruff does not select, and some `# noqa: E501` sit on short
  lines.

### Readability

- `scripts/score_labels_known.py:505`: `score` is about 200 lines (moderate).
- `scripts/score_labels_induced.py:622, 691`: `score` and `score_rules` are near-identical
  65-line functions; one `score_arm` would do (moderate).
- `drift/senses.py:178`: `sample_many` nests five loops deep (minor).
- `__main__.py:509`: `corpus_audit` is 67 lines, mostly table rendering (minor).
- `drift/classifier.py`: the `boilerplate` docstring says sentences but counts texts (minor).
- `corpus/preprocessing.py`: comments at 70–79 restate the code; `TARGET_VARIANTS` overlaps the
  new lemma map, which matters only if the frozen grid is ever rebuilt (minor).
- `scripts/score_labels_known.py:508`: a file opened for `csv.DictReader` is never closed
  (minor).

## For the store rebuild: what to reuse and where

A small `src/interpretant/usages/` package between corpus and drift:

- `usage.py`: `Usage` and `centred` (from `pairs.py`), `tidy_usage` and `tidy_text` (from
  `classifier.py`), one `key_of` (or `Usage.key`) and `Usage.from_row`; `senses.Occurrence`
  gains a `.usage` or gives way to `Usage` plus cell metadata.
- `sampling.py`: `Sampling`, `word_pattern`, `context_around`, `sample_many` (from
  `senses.py`); spaCy's sentence boundaries later replace the sentence regex here.
- `store.py`: the Parquet manifest schema, reader and writer, duplicate marking, store paths,
  shard saving with the float16 check, the index and the vector lookup now in `dev_harness.py`.
- `encoders.py`: `XLLexeme` and `TargetTokenEncoder` (from `pairs.py`) with the device and
  batching helpers; `pairs.py` keeps the distance measures and the DWUG readers.
- `corpus/annotate.py`: the chunked worker-pool and GPU loop of `lemma_votes`, generalised to
  take a per-document function; lemma votes becomes its first user and usage annotation its
  second, fed by one `unique_cased_abstracts`.

`scripts/build_vector_store.py` then keeps only its scope decisions and calls into the package.

## Recommendations

### Critical (before the store rebuild)

1. Create `usages/` with one usage key, `Usage.from_row`, the store paths and the manifest reader
   (coupling, store section).
2. Move corpus reading and cleaning out of `drift/` (`cased_abstracts`, `drop_shared`,
   `foreign_share`, `tidy_text`, `boilerplate`).
3. Split `__main__.py` into a `cli/` package, moving the logic in three commands into the
   library.
4. Rename `drift/metrics.py`'s neighbour code; rename or delete `scripts/discovery/`.

### Moderate

1. Delete the dead code in `direction.py`, `figures.py`, `metrics.py` and `predictions.py`, with
   the tests that only exercise it.
2. Move `FIELD_LABELS` and its helpers beside `FIELDS`; unify neighbours in `drift/grid.py`.
3. Extract the repeated helpers (unit rows, Jensen-Shannon, torch device, cell discovery and
   reading, word lists, label tables, cue lists).
4. Split `lemmas.py` (spelling, annotation, map) and import spaCy lazily; split `openalex.py` and
   `labels.py`.
5. Use `field` for a cell's field everywhere; tidy the names in `labels.py` and `lemmas.py`.
6. Move the scripts' shared helpers into the library and drop `sys.path.insert` imports.

### Minor

1. Remove the unselected `noqa` suppressions and the stale `E501` ones.
2. Add an import-linter layers contract once the moves are done.
3. Split `export_web` by output file and delete its `__main__` block.
4. The smaller readability items above.
