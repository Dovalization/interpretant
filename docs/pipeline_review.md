# Pipeline review

A critical review of the corpus pipeline (`dvc.yaml`: `build_linked_tables` to `report_corpus`),
asked for by the author on 2026-11-16 after the early language stage was tried and reverted
(`docs/plan_early_languages.md`). This file holds the scope, the stage map, the findings and
the proposals.

## Scope

1. **Responsibilities.** Stages and modules that do more than one job (`count_word_repairs`
   counts, decides and applies three repairs; `clean_corpus` merges raw cells, matches patterns
   and applies the keeping rules; `build_corpus` runs seven corpus-level rules).
2. **Order.** What has to run before the spaCy pass and what can run after it; reorderings.
3. **Dependencies.** What makes each stage rerun, and where a stage depends on more than it uses.
4. **Granularity.** Splitting stages into smaller steps where a split saves reruns that cost
   time, set against the extra intermediate files.
5. **What the measurements need.** Which steps change the static and contextual measurements
   (`alignment`, `drift`, `embedding`, not yet in `dvc.yaml`), what those pipelines read (the
   static view, the usages), and whether a thin end-to-end path should come first, so that a
   cleaning step is judged by whether the measurement moves (by its stability only, with the
   threshold set before looking).
6. **Typical changes as the yardstick.** For five typical changes (a cleaning pattern, a word
   repair, a language setting, a corpus-level rule, a top-up download): which stages rerun and
   how many minutes they take. This includes the catalog package marking every stage changed
   and reusing unchanged documents in the spaCy pass.
7. **Filter early, annotate late.** Drops that do not need spaCy (near-duplicates, record types,
   letters) before the pass, so every later stage reads fewer texts.
8. **A built-in report of what changed.** Each run compared with the previous one (texts
   changed in kept abstracts, final-text differences, decision tables), in place of one-off
   comparison scripts, with the outputs a comparison needs kept in the DVC cache.
9. **Interfaces and duplicated work.** One definition of an abstract's final kept text, what
   each stage guarantees to the next, raw cells merged in several stages, the CLI commands'
   repeated setup, and stage names that say what a stage does.
10. **Library use.** Whether each step uses spaCy, Polars, lingua, DVC and datasketch features
    effectively and correctly, checked against the libraries' documentation, and where
    hand-written code duplicates a library feature (the author).

Out of scope: code changes during the review, new cleaning rules, and the way runs are launched
and recorded on the run machine.

## Method and deliverable

- Read from the code, `dvc.yaml`, `params.yaml`, the run log (`metrics/runs.jsonl`) and the
  profiles in `docs/performance.md`; library behaviour from the libraries' documentation.
- Numbers measured, not guessed: rerun minutes from the run log, effects from read-only counts.
- The result: a stage map (each stage's inputs, outputs, parameters, cost, rerun triggers and
  jobs), findings, and proposals, each with its gain on the yardstick (item 6) and its cost in
  code. Nothing changes until the author approves a proposal; a change that is not objectively
  better is not made.

## Stage map (run 8's code, timings from run 8's record `abd16663`)

The rebuild after the revert (record `8bc0fdb`) reproduced every run 8 output byte for byte, so
the pipeline is deterministic today; only the report's column for the previous run differed
(finding 20).

| Stage | Reads | Writes | Params | Seconds | Jobs |
|---|---|---|---|---|---|
| `build_linked_tables` | OpenAlex metadata, Crossref and PubMed records, DOI corrections | the three staging tables | none | 12 | one checked table per source |
| `count_word_repairs` | raw cells, work metadata, both kept lists | three decision tables; repaired texts (not cached) | `corpus`, `count_word_repairs` | 220 | merge the raw cells (10 s), check the metadata (3), repair the cells' text (49), run 7 count, decide, apply (22), run 7c (28), run 8 (101), write (4) |
| `annotate_corpus` | repaired texts | the DocBin cache (not cached, persists) | `corpus`, `annotate_corpus` | 654 | the spaCy pass |
| `detect_languages` | the DocBin cache | sentence languages | `corpus`, `detect_languages` | 121 | lingua on every cached sentence |
| `clean_corpus` | raw cells, work metadata, cleaning rules and patterns, DocBin cache, sentence languages | the annotated table | `corpus`, `clean_corpus` | 210 | merge the raw cells again (10), check the metadata (3), patterns, removals, `not_english` and keeping rules (197) |
| `build_corpus` | annotated table, raw cells, staging tables, record types | the abstracts table | `corpus`, `build_corpus` | 62 | record types, template and repeated sentences, near-duplicates (44), author cap, review flags |
| `report_corpus` | abstracts table, OpenAlex staging table | the report and `metrics/corpus.json` | none | 1 | counts |

A full run takes 21.3 minutes; the spaCy pass is half of it.

## Typical changes (scope item 6)

| Change | Stages that rerun | Minutes |
|---|---|---|
| A cleaning pattern or threshold | clean, build, report | 4.6 |
| A language setting | languages, clean, build, report | 6.6 |
| A word repair (a rule, a kept list) | everything after the staging tables | 21.1 |
| A corpus-level rule | build, report | 1.1 |
| A top-up download | everything | 21.3 and more |
| A shared module (`catalog`, `corpus/repair.py`, `progress`, `workers`, `params`, `pipelines/corpus.py`) | everything, even when no output can change | 21.3 |

The last row is the common one: since 2026-11-13, 42 of 79 commits to `src/` touched a module every
stage depends on, 32 of them in `catalog`; 8 commit messages record a `dvc commit -f` by hand to
avoid the rerun.

## Findings: structure

1. **Shared modules couple every stage.** Each stage lists the whole `catalog` package, whose
   `__init__.py` re-exports all 18 table modules, and every stage that imports a `corpus` module
   loads `corpus/__init__.py`, which re-exports `repair.py`; importing a submodule runs its
   package's `__init__`, so `tests/test_dvc_stages.py` rightly counts them. A change to one
   table's load function, or to text repair, reruns the staging tables and the report too.
2. **A word repair reruns the spaCy pass for every text** although a repair run changes about
   22,000 of 705,283 (3%). The pass, the languages and cleaning could reuse a document whose
   input text and settings are unchanged. Estimate from measured DocBin speeds (read 7.6–13k
   documents a second per process, write 2.5–4.4k): the pass from 654 s to about 60 s, so a word
   repair change from 21.1 minutes to about 6.5. To be measured with a prototype before anything
   is decided; the risk is a stale document if the fingerprint misses a setting.
3. **`count_word_repairs` has four jobs** (cell repair and runs 7, 7c, 8): a run 8 change also
   redoes the cell repair and runs 7 and 7c (about 110 s of its 220). Splitting saves at most
   two minutes per word repair change, for three more uncached copies of the texts. The name
   hides that it rewrites the texts.
4. **The raw cells are merged twice**, by `count_word_repairs` and `clean_corpus` (13 s each);
   cleaning reads the raw cells and metadata only for each work's metadata and raw text.
5. **Filtering before the pass gains little on its own.** 9.3% of works are dropped, about 4.5%
   by rules that need no spaCy (citation and catalogue lines, web scrapes, letters, mojibake,
   exam spam, near-duplicates): at most about 45 s per full run. Moved before the pass, those
   rules would rerun it when they change (4.6 minutes becomes 21), unless finding 2 exists.
6. **The run comparison already exists** (`interpretant compare-corpus`, `review/run_comparison.py`:
   outcomes, removals, sentence languages, samples) but needs the earlier table set aside by hand,
   and covers neither the decision tables nor the final kept text. It went unused while eight
   one-off scripts were written for the early language plan.
7. **Nothing reads the rebuilt corpus yet.** The measurement code (`embedding`: TWEC, PPMI;
   `drift`) is from before the rebuild: it reads the old files (`data/phrased/...`) through
   commands in `__main__.py` (about 800 lines). A thin path needs the static view of the rebuilt
   table, one model per cell and the drift scores of targets and controls (agenda steps 4, 6, 6b).
8. **The stage commands repeat their setup** (seven commands of 36 to 72 lines: catalog, live
   progress, stage timer, run record, result).

## Run 8's profile (py-spy, 20 samples a second, record `abd16663`)

CPU seconds per process role; py-spy sees Python code only, so the native threads of Polars and
lingua are missing from these counts.

| Role | CPU s | Busy CPUs over the stage | Where the time goes |
|---|---|---|---|
| spaCy workers (10) | 6,358 | 9.7 of 12 | tok2vec's maxout layers about 37%; the sentence rules component (`annotation.py`, a Python loop over every token) 6.3%; the tokenizer 6.0%; `LookupAttributeRuler` 4.7%; the lemmatizer 2.7% |
| cleaning workers (10) | 1,823 | 9.3 of 12 | `SpanRuler.match` 66%; reading the DocBins 13% |
| repair workers | 808 | about 9 | `repair_text` 53% (`_add_missing_spaces` 12.5%), the three repairs about 30% |
| language workers (10) | 301 (plus lingua's native time) | | reading the DocBins 44.5% of the Python time, only to get each sentence's text and offsets |
| main process, word repairs | 94 | 1 | the lower-case glued word count: one process for 84 s of the stage's 220 (spell segmentation 38.5%, lemma lookups 38.2%, one candidate word at a time) |
| signature workers (build) | 105 | | shingles and MinHash; 22% importing modules in each new worker |
| main process, build | 37 | 1 | LSH insert and query 60% |

What this adds to the findings:

9. **Cleaning is mostly one spaCy component**: `SpanRuler.match` is about 130 s of the stage's
   197 (the regex patterns, an open item in `docs/performance.md`).
10. **Two stages read the DocBin cache only for the sentences**: the language stage spends 44.5%
    of its Python time deserialising documents, cleaning 13%. A small sentence table (work id,
    start, end) written by the pass would let the language stage skip the cache.
11. **The lower-case glued word count runs in one process** for 84 s; its candidates (168,146)
    are checked one by one, so the check could be split across processes like the other steps.
12. **The sentence rules are a Python loop over every token** in each document, 6.3% of the
    pass (about 40 s).

## Library use: spaCy (spaCy 3.8.18, `en_core_web_sm` 3.8.0)

Checked against the installed source and spaCy's documentation, with small snippets run in the
project's environment (a separate check; the snippets reproduced findings 13 and 14).

13. **A `rest_of_sentence` removal can end before its match** (`corpus/removals.py:211`):
    `span.sent` gives only the first sentence of a span that crosses a boundary (documented on
    the Span page, with the fix `doc[sent.start : max(sent.end, span.end)]`); "c) 2006" with a
    sentence start at "2006" would keep part of a notice. Probably rare: the patterns are two to
    four tokens and the sentence rules block starts inside brackets.
14. **The abbreviation special cases only cover the spellings listed** (`annotation.py:35-51`):
    `add_special_case` is case-sensitive, so "fig.", "eq.", "sect.", "figs.", "eqs." are split and
    end sentences ("see fig. 2 and eq. 3 here." gives three), and a digit after them is not
    rejoined. How often this happens in the corpus is not counted yet.
15. **Why cleaning is mostly `SpanRuler.match`** (finding 9): spaCy's matcher evaluates every
    REGEX and IN predicate of a pattern step on every token before it checks plain attributes,
    so `{"LIKE_URL": true, "TEXT": {"REGEX": ...}}` runs both web-address regexes (one with 274
    domain endings) on every token. Single-token `LIKE_URL` patterns with the regex checked on
    their few matches in the registered spans filter would give the same matches; the gain is to
    be measured.
16. **Minor:**
    - `store_user_data` is now the only reason for the srsly registry patch
      (`catalog/annotations.py`), since documents no longer cross processes; one work-id list
      per part would drop both (the trade-off: `user_data` is the documented place);
    - the sentence-rules factory declares `requires` where it means `assigns`
      (`annotation.py:154`; `nlp.analyze_pipes()` reports it);
    - the language stage loads documents on a bare `Vocab()`, whose lexical flags read False
      (safe today, a trap if one is used);
    - the `doc_cleaner` comment's reason is out of date;
    - the two lemmatizer loaders (`corpus/dictionary.py`, `corpus/lemmas.py`) load more than the
      lemmatizer and duplicate each other;
    - `memory_zone` would be safe in `_annotate_part`, for a small gain.

Done well and to keep: loading with `exclude` then `enable_pipe("senter")`; removing the hyphen
infix through `compile_infix_regex`; DOIs through `url_match`; the sentence rules before `senter`
(whose `overwrite = false` keeps them); per-part worker processes instead of `n_process`; the
DocBin round trip (sentence starts, norms, lemmas, work ids); `LookupAttributeRuler` and its tests
against the stock ruler; the SpanRuler set-up and its registered filter.

## Library use: lingua, DVC, datasketch and the smaller libraries

Checked against lingua 2.1.1, datasketch 2.0.0, DVC 3.67.1, symspellpy 6.10.2 and pandera 0.36,
their documentation and source, with scripts run on the Mac's local tables (which may predate
run 8: the counts below show the mechanism, not run 8's exact numbers).

17. **A sentence lingua cannot score is labelled Arabic** (`corpus/language.py:187-190`,
    `_close_call`): when no candidate covers the script every confidence is 0.0, detection
    returns None for the tie, and the close-call rule takes the first candidate, `ar`. In the
    local table: 493 sentences in 195 works (30% of all `ar` labels): Greek, Tamil, Devanagari,
    Thai, letter-spaced OCR English; 92 of the works mostly English. They are removed as
    `not_english` for the wrong reason.
18. **The close-call rule overrides lingua's "not reliably possible"** (lingua's README) for any
    pair of languages, not only the confusable pairs its docstring names (Croatian and Bosnian,
    Indonesian and Malay): 2,975 local labels come from other pairs (German and Dutch 361,
    Spanish and Italian 231), 1,820 of them five words or fewer ("P. T. Geach." as German), and
    each is removed as `not_english`. A list of confusable pairs in `params.yaml`, `und` for the
    rest, would follow lingua. Effect on drops not measured.
19. **lingua reads every detected sentence twice** (`_detect_texts`, `language.py:163-184`):
    one `compute_language_confidence_values_in_parallel` call with the distance test in Python
    gave identical labels on 163,660 sentences in half the time (5.6 s instead of 11.0 s).
20. **DVC:**
    - the shared `catalog` package marks every stage changed (finding 1, found independently);
    - `cache: false` and `persist` turn off DVC's run cache, so `count_word_repairs`,
      `annotate_corpus` and `report_corpus` can never be restored from it;
    - `report_corpus` uses `persist` to read the previous metrics, so the report depends on a file
      that is not a dependency and repeats `dvc metrics diff`;
    - the CLI modules (`cli/<stage>.py`, which build the settings objects) are not dependencies
      of their stages, so a change there would not rerun them;
    - `tests/test_dvc_stages.py` does not check each stage's `params` against its settings'
      section (they match today).
21. **Pinning:** `uv.lock` resolves lingua 2.1.1 on Python 3.11 but 2.2.0 (new models) from 3.12,
    so a Python upgrade would change detection with no DVC signal.
22. **Minor:** LSH set through a stand-in threshold where datasketch documents `params=(b, r)`
    (the figures in `params.yaml` are right: 0.6 gives bands and rows 18 and 7);
    `minhash_signatures` re-implements `MinHash.bulk` and its comment about datasketch is wrong;
    the sentence-language and word tables are checked by schema only, where a pandera check
    (confidence above 0 unless `und`) would have caught finding 17; an out-of-date docstring.

Done well and to keep: lingua's restricted candidates, preloading, one rayon thread per worker,
the rejection of multi-language detection, rounding confidences; LeanMinHash with explicit scheme
and seed, the mmh3 digest, exact Jaccard; DVC `vars` from `catalog.yaml`, per-stage params,
`.dvcignore`, metrics in git (`foreach` and `matrix` do not fit: DVC runs stages one at a time);
pydantic-settings with `yaml_config_section`, `extra="forbid"` and environment variables removed;
symspellpy at edit distance 0 for pure segmentation.

## Library use: Polars (2.1.0)

Checked against the installed source, Polars' API reference and its 2.0 upgrade guide, with
small local snippets.

23. **The code relies on Polars 2's `explode` while the pin allows 1.x** (`cleaning.py:154-169`;
    `pyproject.toml`, `polars>=1.0`): under 1.x a kept abstract with no sentences explodes to a
    null row, 20 or more such rows form one template group with a NaN share, and they are dropped
    as `template` (reproduced with `empty_as_null=True`: 25 of 25). The schema would not catch it:
    `list.all()` ignores nulls, so a removal with a null start passes `ABSTRACTS_SCHEMA`.
    `uv.lock` holds 2.1 today; pin `polars>=2.1`, pass `empty_as_null` explicitly, fill nulls in
    the span checks.
24. **Row order the documentation does not guarantee:** two left joins without
    `maintain_order="left"` (`record_types.py:69-78`, whose docstring promises the order, and
    `work_metadata.py:155`, before the staging table's hash); the corpus report sorts by counts
    with no tie-break after an unordered `group_by` (`corpus_report.py:157,174,224,238,310`), so
    equal counts can swap in the tracked report. Order was kept in a 2M-row test: a
    reproducibility risk, not a bug seen.
25. **Memory in `build_corpus` (11.7 GB peak):** all three staging tables are collected in full
    (`build_corpus.py:88`) where about five columns of each are used; full-width filters, sorts
    and partitions copy the nested `sentences` and `removals` columns; every candidate text and
    its removals become Python objects twice (`build_corpus.py:131`, `deduplication.py:112-118`).
    Selecting columns before collecting and passing per-part frames to the workers would save
    gigabytes (not measured).
26. **Memory in `count_word_repairs` (about 10–11 GB peak):** the texts go from Python tuples to a
    DataFrame three times, with two generations of strings alive at once; per-part frames or
    parquet parts read lazily would keep them out of the main process.
27. **Minor:**
    - `mark_truncated` loops over 705k rows (`cleaning.py:96-104`) where one expression gives the
      same result;
    - 88 filters over the full table where one `partition_by` would do (`build_corpus.py:117`);
    - the raw cells are read line by line through pydantic and merged in Python (`raw.py:58-60`,
      `works.py:28-53`; 10 s in each of two stages), where `scan_ndjson` with a `group_by` has
      the shape that took the metadata check from 30 s to 3 s;
    - three `str.extract` calls with one regex where `str.extract_groups` makes one pass;
    - the report collects every column (4.6 GB peak).

Not worth changing: the Python loops in the word counts wrap calls Polars cannot make (the
lemmatizer, symspellpy, regex callbacks with address checks); `str.replace_many` cannot match
whole words only; `map_elements` in `crossref.py` is justified.

Done well and to keep: the word counts as lazy streaming queries with `extract_many`
(overlapping, padded patterns); explicit null handling before comparisons; decision tables sorted
on unique keys; `scan_parquet` over the parts with schema-only pandera checks; writes through a
`.partial` file against an explicit schema; spawn for worker processes; the staging builders as
pure expressions.

## Proposals

Each proposal says what it gains on the typical changes (item 6), what it costs in code, and how
it is shown to be objectively better before it is kept. Nothing here is approved yet.

**A. Measure first.**

- **A1. Make the run comparison routine** (finding 6): `compare-corpus` reads the earlier build
  from the DVC cache by git revision instead of a folder set aside by hand, and adds the decision
  tables and the final kept text of kept abstracts. Every later proposal is judged with it.
  Cost: one module and its tests. Shown by: it reproduces what the one-off scripts of the early
  language plan found (49 kept abstracts against run 8 for the last check run).

**B. Correctness** (each with a test that fails before the change, and its effect on the kept
corpus counted with A1 before it is kept):

- **B1.** A `rest_of_sentence` removal reaches the end of its match (13). Reruns cleaning, 4.6 min.
- **B2.** Sentences lingua cannot score get their own label instead of `ar`, and the close-call
  rule applies only to listed confusable pairs (17, 18). Reruns from the languages, 6.6 min.
- **B3.** The lower-case abbreviation forms (14), counted first: how many kept abstracts'
  sentence breaks change. Reruns everything, 21 min.
- **B4.** Pins and determinism: `polars>=2.1` with explicit `empty_as_null` and null-checked
  spans (23), lingua pinned (21), `maintain_order="left"` on the two joins and tie-breaks in the
  report's sorts (24). Shown by: identical outputs.
- **B5.** The CLI modules as stage dependencies, and a test that each stage's `params` match
  its settings' section (20). No output changes.

**C. Rerun cost and speed** (each measured on a run against the record before it):

- **C1. Narrow imports** (1): the stages import `catalog` and `corpus` submodules, and the two
  packages' `__init__.py` stop re-exporting, so a table's code reruns only the stages that read
  or write it. Shown by: replaying the 42 recent commits that touched a shared module against the
  narrower dependency lists and counting the minutes of reruns avoided. Cost: mechanical import
  edits; no output changes.
- **C2. lingua once per sentence** (19): half of lingua's time, identical labels.
- **C3. The web-address patterns in cleaning** (9, 15): single-token `LIKE_URL` patterns with the
  regex checked on their matches; same removals, cleaning's 197 s measured before and after.
- **C4. The lower-case count across processes** (11): 84 s in one process, the same counts.
- **C5. A sentence table from the pass** (10), so the language stage stops deserialising
  documents; measured on the language stage.
- **C6. Reuse unchanged documents** in the pass, the languages and cleaning (2): estimated to take
  a word repair change from 21 to about 6.5 minutes. A prototype first; shown by a full run and an
  incremental run giving identical outputs, and the time saved.

**D. Memory** (25, 26, 27): columns selected before collecting, per-part frames to workers, no
second copy of the texts as Python objects; measured by the run log's peak memory, outputs
identical.

**Not proposed now** (not objectively better on their own): splitting `count_word_repairs` (3) or
`build_corpus`; filtering before the pass (5), which only pays once C6 exists; DVC `foreach`.

**A separate decision: what the measurements need** (5, 7). Nothing reads the rebuilt corpus yet.
A thin path (the static view of the rebuilt table, one static model per cell, drift scores of
targets and controls) would let a cleaning step be judged by whether the measurement moves. It
changes agenda 9.1's order, which puts the corpus sign-off before any experiment.

Suggested order: A1, then B1 to B5 (small, and each measured with A1), C1, C2 to C5, C6, D; the
measurement question whenever the author wants to take it up.

## More performance tips from the libraries' documentation

Asked for by the author after the proposals were approved. Timings are from synthetic runs on the
Mac unless stated; each is to be measured on a real run before it is kept.

**Polars, multiprocessing and DVC:**

28. **Every new worker re-imports the whole command line.** Spawned workers import
    `interpretant.__main__`, which imports every command and spaCy: 2 to 3 s per worker (the
    signature workers' 22% importing modules in run 8's profile). `count_word_repairs` opens a new
    10-worker pool four times (`pipelines/workers.py:47-48`); one pool reused for the four rounds
    took 11 s instead of 48 in the test (about 10 s on the run machine), with the same outputs.
    Lighter imports in `__main__` would shorten every worker's start in every stage.
29. **`POLARS_MAX_THREADS=1` in worker processes** (Polars' `thread_pool_size` notes), set in
    the parent's environment before the pool starts: worker peak memory fell from about 290 MB to
    180 MB in a 10-worker test (about 1 GB in all), time within noise, parquet bytes identical.
30. **Explode lazily** (Polars 2.0 upgrade guide: lazy queries drop unused columns and stream):
    the report explodes the whole abstracts table, text and sentences included
    (`review/corpus_report.py:46` and others); lazily, a 0.9 GB test table peaked at 1.28 GB
    instead of 2.11 GB, five times faster. Expected: the report's 4.6 GB peak to about 1 GB.
31. **One scan for the 88 annotated parts** (`scan_parquet` with `include_file_paths`) instead of
    88 reads: 2.2 s to 0.9 s on a test table, identical frames.
32. **DVC's hashing needs no tuning**: a full run's outputs hash in a few seconds; on the run
    machine's btrfs, outputs are reflinked into the cache.

Checked and not worth it: parquet codec, level and row groups (one or two seconds, and they change
the bytes DVC hashes), `sink_parquet` (`write_parquet` already streams), categoricals (tens of MB
against gigabytes of text), `engine="streaming"` (lazy queries already stream in Polars 2).

**spaCy, lingua and datasketch** (timed on the Mac with 1,000 to 3,000 real abstracts; ratios
only):

33. **The sentence rules set each sentence start one token at a time**, and each
    `token.is_sent_start =` calls `doc.has_annotation("DEP")`, which scans the whole document when
    there is no parser (spaCy's source), so the cost grows with the square of an abstract's length.
    Setting all starts in one `Doc.from_array` write ran 5.4 times faster with identical sentence
    starts on 2,000 documents: about 30 s of the pass.
34. **Exact-word cleaning patterns as phrase patterns** (spaCy's rule-based matching docs,
    "Efficient phrase matching"; `phrase_matcher_attr: "LOWER"`): `{"IN": ...}` predicates are
    Python calls on every token; 18 patterns as 43 phrases cut the SpanRuler's time by 29% with
    identical spans. Two patterns with hyphens stay token patterns, since the blank tokenizer splits
    hyphens and the pass does not.
35. **The other single-token regexes off the per-token path** (docs: matching regular expressions
    on the full text): regex predicates are 63% of the SpanRuler's time; one scan of the text per
    document, then the token regex only where it overlaps, gives the same matches. Up to about 30%
    more of cleaning; not prototyped.
36. **`gc.freeze()` and `gc.disable()` around the LSH loop** (Python's `gc` docs): each insert
    creates about 18 sets; on 300,000 synthetic signatures, query and insert took 30 µs instead of
    57 to 69. Perhaps 10 s of the build.
37. **Minor:** once the sentence table exists (C5), the language stage in one process on lingua's
    multi-threaded methods (one copy of the models instead of ten); DocBin `attrs` without the empty
    parser and entity attributes (files 16% smaller); the cleaning pipeline built once per worker
    instead of once per part.

Checked and not applying: the GPU (spaCy's figures give a CNN pipeline about 1.5 times on one GPU,
less than ten CPU workers, and GPU arithmetic could change tags or sentence starts); BLIS threads
(each operation is single-threaded already); `nlp.pipe` batch sizes (within noise); a smaller
model (none smaller than `sm`, and a different model changes outputs); datasketch's insertion
sessions (only for Redis or Cassandra storage).

These extend proposals C and D: **C7**, one worker pool per stage and lighter worker start-up (28);
**C8**, the sentence rules in one array write (33); **C9**, phrase patterns and full-text regexes in
cleaning (34, 35, beside C3); **C10**, the garbage collector frozen around the LSH loop (36); **D**
gains 29, 30, 31 and 37.
