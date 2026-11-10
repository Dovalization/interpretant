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

### Round U results (2026-11-10, `scripts/score_pairs_benchmark.py`)

Scored as written. 3 passed, 1 falsified. 46 words, 9,107 usages.

| | Verdict | Numbers |
|---|---|---|
| U1 | **Passed** | APD's agreement with `change_graded`: 0.864 (published 0.886, needed 0.85) |
| U2 | **Passed** | APD 0.864, PRT 0.650 (published 0.886 and 0.676) |
| U3 | **Falsified** | J's agreement: 0.539 (needed 0.75) |
| U4 | **Passed** | Periods shuffled 20 times: J's mean agreement −0.124 (range −0.384 to +0.350); APD's +0.842 (range +0.838 to +0.846) |

The model's revision c9f85c6f holds its weights as `pytorch_model.bin`; `transformers` also
fetched a safetensors copy from an open pull request by Hugging Face's conversion bot
(`refs/pr/1`). The two files were compared after the run: all 392 tensors are identical, so the
run used the pinned weights.

### Reading

U1 passed, so the pipeline is validated and rung 2 may start with this configuration. J's
agreement is more than 0.05 under APD's, so by the reading rule rung 2's primary measure is APD,
and the README is to note that APD also responds to a word's variety of uses when the instrument
is reported there.

### Exploratory, after scoring: how much of APD is change

Not predicted in size; U4 bet only that APD would keep at least 0.4.

With every word's usages shuffled between the two periods, so that no difference between periods
can be measured, APD still agrees with the graded change score at 0.842, against 0.864 with the
periods intact. Almost all of APD's agreement on this benchmark comes from how varied a word's
usages are overall, not from a difference between the periods: words with more senses in the
pooled data also have higher graded change scores, so a measure of variety ranks the 46 words
nearly as well as a measure of change. J nets each period's own variety out and falls to noise
when the periods are shuffled; its 0.539 is the agreement that remains attributable to change
between the periods. A reproduction of 0.864 therefore validates the pipeline, not APD as a
measure of change. Rung 2's predictions should carry the shuffled control and J beside APD, or
APD would mostly rank the corpus's words by their number of senses.

## Round X: does the pipeline find this project's known changes over time? (committed 2026-11-10, before any corpus usage was encoded with XL-LEXEME)

Rung 2 of the calibration ladder. Round U validated the pipeline on a human benchmark and showed
that APD mostly measures a word's variety of usages. This round gives the frozen configuration
the documented changes the earlier instruments were tested on, and the quiet transitions where
nothing documented happened. Round U's reading rule makes APD the primary measure; this round
scores APD and the across-minus-within measure J side by side, each against two baselines that
round U showed are needed.

### Method (fixed before any corpus usage is encoded)

- **Configuration, frozen from round U.** XL-LEXEME at revision c9f85c6f through
  `interpretant.drift.pairs.XLLexeme`; APD and J as round U defines them.
- **Usages.** From the cased abstracts behind each cell (round C′'s quality rules: abstracts
  more than 20% outside the Latin script skipped, URLs and DOIs removed), abstracts found in both
  cells of a transition removed, the word matched with its regular plural, one occurrence per
  abstract, at most 150 abstracts per cell, drawn in a seeded random order (seed 1), the
  sentence cut to 100 words around the occurrence (`interpretant.drift.senses`).
- **Transitions.**
  - Documented changes: *attention* in AI, 2010s → 2020s (328 and 899 abstracts available);
    *prompt* in AI, 2000s → 2020s, word-targeted cells (298 and 756); *platform* in
    communication, 2000s → 2010s (142 and 560); *network* in AI, 1980s → 1990s (1,230 and
    2,513); *alignment* in AI, 2010s → 2020s (105 and 320), the weak case, as in round 5.
  - Quiet transitions: *attention* in AI, 1990s → 2000s (186 and 174), and in psychology,
    2010s → 2020s (623 and 658).
  - Controls: round C's 30 random control words, on each transition's grid cells (for *prompt*,
    AI's 2000s and 2020s grid cells), every one with at least 100 abstracts in both cells.
- **Found.** A measure finds a change in a word on a transition when its value is (a) above
  the 95th percentile of the same measure over 200 shuffles of the word's usages between the two
  cells, keeping each cell's size, and (b) above the 90th percentile of the same measure over
  the transition's control words. (a) rules out variety alone; (b) rules out the drift every
  word shows between decades.
- **Sizes.** A word with fewer than 100 usages in either cell is not measured.
- Already seen: the counts above, round U's results, and every earlier instrument's verdict on
  these transitions (round 5's grid, rounds C–C″ for *attention* and *prompt*). Not computed:
  any XL-LEXEME vector of a corpus usage.

### Predictions

**X1 — APD finds the documented changes.** APD finds at least three of *attention*, *prompt*,
*platform* and *network*. *Falsified* if it finds two or fewer.

**X2 — APD stays quiet where nothing documented changed.** APD finds neither quiet transition.
*Falsified* if it finds either.

**X3 — J finds the documented changes.** As X1, for J.

**X4 — J stays quiet.** As X2, for J.

**X5 — Every word drifts between decades.** Under baseline (a) alone, J finds a change in more
than 30% of the control words' transitions, pooled over all transitions: contexts shift between
decades for ordinary words too, which is why baseline (b) is needed. *Falsified* if 30% or
fewer.

Reading rule:

- **APD's pair (X1 and X2) passes:** rung 3 uses APD, as round U's rule requires, with J
  reported beside it.
- **Only J's pair (X3 and X4) passes:** J replaces APD as the primary measure from rung 3 on,
  and the README says why: on this corpus, APD did not separate change from variety and drift.
- **Neither pair passes:** the usage-pair instrument does not proceed to rung 3, and the design
  stops at this rung.
- *alignment* is reported for each measure and counts towards nothing. X5 describes the corpus
  and decides nothing about the instrument.

Scored by `scripts/score_pairs_known.py`, committed before it is run.
