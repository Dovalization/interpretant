# Pipeline review

A critical review of the corpus pipeline (`dvc.yaml`: `build_linked_tables` to `report_corpus`),
asked for by the author on 2026-11-16 after the early language stage was tried and reverted
(`docs/plan_early_languages.md`). This file holds the scope now; the stage map, findings and
proposals follow it.

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
