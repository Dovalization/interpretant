# Predictions: meaning by usage pairs

Pre-registered predictions for the usage-pair rounds (ROADMAP item 8c; literature and design in
`docs/related_work.md`). As in the other predictions files, a round's predictions are committed
before any code that tests them is run on its evidence; results are appended below them, and
predictions and verdicts are never edited afterwards. Corrections go in new sections.

## Round U: does the pipeline reproduce a published result on a human benchmark? (committed 2026-11-10, before any benchmark usage was encoded)

Rung 1 of the calibration ladder. The project's earlier instruments were checked against their
own noise, then against known answers the author chose. This round checks the new pipeline
against a benchmark nobody on this project built or chose: DWUG EN, about 9,000 English usages
of 46 words from two periods (1810–1860 and 1960–2010), with a graded change score per word
derived from human judgements of usage pairs (Schlechtweg et al. 2021). Periti and Tahmasebi
(2024, Table 1) report a Spearman correlation of 0.886 between that score and XL-LEXEME's
average pairwise distance (APD) on version 2.0.1, and 0.676 for the distance between mean
vectors (PRT).

The round also tests the measure the redesign proposes. A word whose usages vary more has a
larger APD whether or not it changed (Kutuzov et al. 2022), so the redesign nets the distance
across two cells against the distance within each. A control with the periods shuffled checks
which measure still responds when no change can be present (Dubossarsky et al. 2017).

### Method (fixed before any benchmark usage is encoded)

- **Data.** DWUG EN version 2.0.1 (Zenodo record 7387261, md5
  80c3f00f3d13e5ad396e8464ebe67bbc, kept under `data/external/`, not redistributed): every
  usage in `data/<word>/uses.csv` with its period (`grouping` 1 or 2) and target span
  (`indexes_target_token`); the gold score is `change_graded` in `stats/opt/stats_groupings.csv`.
- **Encoder.** XL-LEXEME, `pierluigic/xl-lexeme` at revision c9f85c6f, run through
  `interpretant.drift.pairs.XLLexeme`: the target wrapped in `<t>` and `</t>`, the input centred on
  the target and cut to 128 tokens, masked mean pooling, as in the authors' `WordTransformer`,
  which is reimplemented rather than installed. Every usage is encoded once.
- **Measures**, per word, with cosine distance:
  - **APD**: the mean distance over all pairs of one period-1 and one period-2 usage;
  - **PRT**: the distance between the two periods' mean vectors;
  - **J**: APD minus the mean of the two within-period APDs (each over all pairs of distinct
    usages of one period), the redesign's across-minus-within measure.
- **Agreement** is Spearman's ρ between a measure and `change_graded` over the 46 words.
- **Shuffle control.** For each word, its usages are reassigned to the two periods at random,
  keeping each period's size; 20 shuffles with a fixed seed (2026). A measure's shuffled
  agreement is the mean ρ over the 20 shuffles.
- Already seen: the release's file layout, its 46 words, 9,107 usages and the gold column; the
  tokenizer's handling of the markers on one made-up sentence. Not computed: any vector of a
  benchmark usage, any distance, any correlation.

### Predictions

**U1 — The pipeline reproduces the published result.** APD's agreement with the graded change
score is at least 0.85 (published 0.886). *Falsified* if under 0.85.

**U2 — APD beats PRT, as published.** APD's agreement is higher than PRT's. *Falsified*
otherwise.

**U3 — The redesign's measure tracks change too.** J's agreement is at least 0.75. *Falsified*
if under 0.75.

**U4 — J responds to change, APD also to variety.** With the periods shuffled, J's mean
agreement is under 0.2 in absolute value, while APD's stays at 0.4 or more: shuffling removes
change but not a word's variety of uses, and only APD measures variety. *Falsified* if either
part fails.

Reading rule:

- **U1 fails:** rung 2 does not start. One rerun is allowed, as a new round, only if a fault in
  the pipeline is found and recorded; if there is none, or the rerun fails, the usage-pair
  design stops here.
- **U1 passes:** the pipeline is validated, and rung 2 uses this configuration unchanged. Its
  primary measure is J if J's agreement is at least APD's minus 0.05 and U4's J part passes;
  otherwise APD, with the README noting that it also responds to a word's variety of uses.
- U2 checks the published ranking; U4's APD part tests the concern that motivates J. Neither
  decides whether the pipeline proceeds.

Scored by `scripts/score_pairs_benchmark.py`, committed before it is run.
