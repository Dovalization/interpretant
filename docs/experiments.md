# Past experiments

This is the record of the experiments run before the 2026-11-12 rebuild plan, written before their
code is deleted. Git keeps the code; this record keeps each experiment's question, method,
result, limits and status in one place. The predictions files (`docs/predictions.md`,
`docs/predictions_cross_perplexity.md`, `docs/predictions_pairs.md`, `docs/predictions_labels.md`,
`docs/predictions_senses.md` and `docs/diagnostics.md`) stay as the detailed ledger, with every
prediction, number and correction. There is one entry per round or check, 37 in all. Dates are the
git dates of the commit that committed the predictions. Two rounds carry the letter U (one in
usage pairs, one in sense labels); the second is called round U (labels, head-to-head). Round 4
has two entries, one for the static grid and one for the whole-field comparison, because one
commit pair holds both.

Status legend:

- **holds**: the recorded result stands and nothing later contradicts it.
- **withdrawn**: the entry's own source, or a later round, undid it.
- **superseded**: a later round replaced it.
- **untestable**: it could not be scored as written, or cannot be tested.

A result that stands while a reading of it was later corrected is marked **holds**, and the
entry says which reading changed.

## Summary

In this table the line is shortened to its first word: grid is the static grid, whole-field the
whole-field comparison, pairs the contextual usage pairs, labels the sense labels, clustering the
sense clustering and check the diagnostic checks.

| Round | Line | Finding | Status |
|---|---|---|---|
| Round 1 | grid | 3 of 9 passed; misses restated later | superseded |
| Round 2 | grid | 7 of 9 passed on a second seed | superseded |
| Round 3 | grid | PPMI-SVD: 4 of 8 passed | holds |
| Round 4 (grid) | grid | lemmatised corpus: 6 of 8 passed | holds |
| Round 5 | grid | misses 3 of 4 documented changes | holds |
| Round 6 | grid | SemEval: ρ 0.285 and 0.295 | holds |
| First run | whole-field | 5 of 6 passed, with design flaws | superseded |
| Corrected run | whole-field | subfield predicts parent; fields converge | holds |
| HCI-philosophy follow-up | whole-field | one-sided; not concentrated after 2000 | holds |
| Technology turn | whole-field | philosophy did not turn to computing | holds |
| Round 4 (whole-field) | whole-field | L8 to L10 all passed | holds |
| Round G | whole-field | gain spread across HCI abstracts | holds |
| Round J | whole-field | 7 of 9 models gain on HCI | holds |
| Round K | whole-field | too few shared venues | untestable |
| Round K′ | whole-field | gain persists in fixed venues | holds |
| Round S | whole-field | classifier: HCI drifts, not one-sided | holds |
| Round V | whole-field | drift is not technology words | holds |
| Round M | whole-field | no finding rests on a flagged cell | holds |
| Round N | whole-field | gains net of drift; blind spot found | holds |
| Round W | whole-field | planted check failed | untestable |
| Round U (pairs) | pairs | APD keeps 0.842 when shuffled | holds |
| Round X | pairs | J finds 4 of 4, one false alarm | holds |
| Round Y | labels | labels rank DWUG at 0.767 and 0.821 | holds |
| Round Z | labels | WordNet lists score about 0.18 lower | holds |
| Round Q | labels | induced lists score 0.39 to 0.59 | holds |
| Round E | labels | classifier 0.669, partial 0.509 | holds |
| Round T | labels | classifier narrowly misses rung 2 | holds |
| Round U (labels, head-to-head) | labels | classifier chosen 43.9% | holds |
| Round A | labels | 2 of 4 changes against controls | holds |
| Round A′ | labels | test set fixed; never run | superseded |
| Round C | clustering | joins mechanism and cognitive attention | superseded |
| Round C′ | clustering | mixed sense in two seeds | superseded |
| Round C″ | clustering | *prompt* fails both ways | holds |
| Check 1a | check | all 5 changes present, 2 quiet | holds |
| Check 2b | check | XL-LEXEME and author 0.431 | holds |
| Check 1b | check | plant seen from 30%; placebo found | holds |
| Check 2b′ | check | judges agree 0.616 | holds |

## Static grid

One TWEC (and later PPMI-SVD) model per field and decade, drift judged against the corpus's own
noise floor. Source: `docs/predictions.md`.

### Round 1: does the grid's drift and neighbour evidence match nine predictions? (2026-11-09)

- **Line:** static grid
- **Question and prediction(s):** Do nine predictions about named words (P1 to P9), written after
  the models were trained and before `drift grid` was run, match the grid? Bars as written: P1,
  under 5% of control measurements are candidates and no control is a candidate in more than 3
  pairs; P2, *network* in AI is a candidate at 1980s → 1990s or 2000s → 2010s; P3, *representation*
  in AI has 1980s neighbours near {knowledge, logic, rule, symbolic, reasoning, frame} and 2010s
  or 2020s neighbours near {feature, learned, embedding, deep, latent, encoder} ("near" is at
  least 2 of the set in the 25 neighbours); P4, *affordance* nearer DESIGN than ECOLOGICAL in
  HCI and the reverse in philosophy, in both 2010s and 2020s; P5, *embodiment* near {tangible,
  interaction, physical, wearable, design, gesture} in HCI and near {phenomenology, mind,
  cognition, merleau-ponty, perception, enactive} in philosophy (2020s); P6, *consciousness* in
  philosophy a candidate at 2010s → 2020s with at least 2 new neighbours from an AI set; P7,
  *intentionality*
  in philosophy stable (a null: no candidate from the 1980s to the 2020s); P8, *information*
  diverges between psychology and computer science (mean cross-field rank over the 1960s–70s
  lower than over the 2000s–10s); P9, *network* converges between sociology and computer science
  (rank lower in the 2010s than in the 1990s).
- **Method and settings:** Main grid `models/twec_grid/openalex` (eleven fields, 1950s–2020s,
  up to 10,000 abstracts per cell) and full cells `models/twec_grid/openalex_full`, seed 42,
  unlemmatised corpus. TWEC settings as stated in round 6 and `ROADMAP.md`: 100 dimensions,
  window 5, skip-gram, 10 epochs, 5 negative samples, compass of 200,000 tokens per cell. A
  candidate is in the top 5% of its pair on both cosine drift and neighbourhood shift (words with
  at least 150 occurrences in both cells, ranked within frequency bins). Neighbours are the 25
  nearest words among words with at least 50 occurrences in the cell. A word under 150
  occurrences in a required cell is untestable. Disclosure: one pair (computer science, 1950s →
  1960s) was inspected while testing the code; no prediction uses it.
- **Result:** 3 passed, 6 falsified. P1 falsified on its second criterion: control candidates
  0.7% within field (worst control *day*, 1 pair) and 0.9% cross field, but *food* is a
  candidate in 4 cross-field pairs, above the limit of 3. P2 falsified: AI 1980s → 1990s cosine
  rank 84, neighbour rank 90; 2000s → 2010s 79 and 54. P3 passed: 1980s neighbours include
  *knowledge, declarative, frame-based, formalism, reasoning*; 2020s include *embeddings, encoder,
  latent, contrastive, self-supervised, learned* (5 of the learned set). P4 falsified:
  philosophy leans ecological in both decades (2 vs 0), HCI matches neither set (0 vs 0). P5
  falsified: HCI 2020s neighbours (*sense of embodiment, ownership, disembodiment, presence,
  illusion*) hit 0 of the committed set; philosophy 2020s (*embodied, corporeality, spatiality*)
  hit 0 of the set. P6 falsified: a candidate (ranks 98 and 99) but only 1 new word of the AI
  set (*computational*). P7 passed: no transition is a candidate. P8 falsified, opposite
  direction: psychology × computer science rank 51 and 39 in the 1960s–70s, 18 and 15 in the
  2000s–10s. P9 passed, negligible: 96.6 in the 1990s, 96.3 in the 2010s. By P1's own rule the
  results were "not yet interpreted".
- **Downsides:** The older cleaning applies: curly apostrophes were deleted (fusing tokens such
  as *one's* → *ones*), four rule sets ran on different paths, and the tokenisers were regexes.
  The grid's two-threshold rule (top 5% on both measures) applies. P1's limit of 3 pairs was
  set without computing what chance gives, and *food* is a weak cross-field control because
  fields study food differently (the source's own reading). Cross-field ranks (P8, P9) were later
  shown not to register a new sense well (round 5, E6: ranks rise but stay under 90). P4 to P6
  got the direction right and the vocabulary wrong; hypotheses drawn from the observed
  neighbours are new and were not tested on these results. Round 2 notes that 10,000-abstract
  cells sit at a resolution limit for *consciousness*.
- **Status:** **superseded**, because rounds 2 to 4 restated its misses as predictions and
  retested them on a second seed, a second method and the lemmatised corpus, and round 5 changed
  how P7's null is read.
- **Retested by the rebuild:** partly. The static grid (TWEC, PPMI-SVD, `drift grid`) is rerun on
  the rebuilt text (detection experiment 4; step 6b). The agenda names no retest of these
  individual word findings.
- **Source and commits:** `docs/predictions.md`, "Predictions" and "Results (2026-11-09,
  `scripts/score_predictions.py`)"; predictions f76a915 (2026-11-09); results 929afb1.

### Round 2: do round 1's surprises hold on a second seed and unseen cells? (2026-11-09)

- **Line:** static grid
- **Question and prediction(s):** Do round 1's surprises, restated as predictions, hold on
  evidence not yet seen? A prediction passes only if every listed piece of evidence agrees. R1,
  P3 replicates (seed 43). R2, P7 replicates. R3, HCI's *embodiment* has at least 3 of
  {presence, ownership, illusion, immersion, self-location, avatar, virtual, agency} among its
  neighbours (seed 43 2020s and seed 42 2010s, full cells). R4, philosophy's *embodiment* has at
  least 3 of {corporeality, spatiality, intersubjectivity, affectivity, temporality, lived,
  flesh, embodied}. R5, philosophy's *consciousness* 2010s → 2020s is a candidate with at least 3
  of {computation, computational, neural, substrate, information_processing, informational,
  intelligence} among its 2020s neighbours (seed 43 full cells and seed 42 main grid). R6,
  *information* converges between psychology and computer science: mean cross-field rank over
  the 1960s–70s at least 15 points higher than over the 2000s–10s (seed 43; and seed 42 cognitive
  psychology × computer science). R7, HCI's *affordance* has at least 3 of {experiential,
  qualities, physicality, interactivity, metaphors, sensorial, materiality} in each of the 2010s
  and 2020s (seed 43, full cells). R8, 15 cross-field controls with the lowest variation across
  fields (`data/control_words_cross_field.txt`): under 5% candidates and no control a candidate
  in more pairs than the 99th percentile of a binomial. R9, at least 80% of within-field and
  80% of cross-field target candidates change more between their two cells than between two
  trainings of the same cell.
- **Method and settings:** Grid retrained with seed 43 (`models/twec_grid_seed43`, same text,
  compass sample and settings); unlemmatised corpus. The two seeds have separate compasses, so
  neighbour lists are compared, not vectors. Seed noise is the neighbourhood shift (Jaccard over
  25 neighbours from words with at least 50 occurrences) between the seed-42 and seed-43 models
  of the same cell. R9 was committed before the seed-43 grid finished training.
- **Result:** 7 passed, 2 falsified. R1 passed: seed 43 2020s neighbours include *embeddings,
  encoder, learned, contrastive, latent* (5 of the learned set). R2 passed. R3 passed: 6 of the
  set in both cases (*ownership, presence, illusion, self-location, agency, immersion*). R4
  passed: 6 and 7 of the set. R5 falsified: seed 43 full cells is a candidate with 7
  computational neighbours; seed 42 main grid is not a candidate (3 hits). R6 passed: seed 43
  psychology × computer science 1960s–70s mean 36, 2000s–10s mean 10; seed 42 cognitive
  psychology × computer science 55 and 36 (gap 19). R7 passed: 6 and 5 of the set. R8 falsified:
  0.96% candidates, but *food* exceeds its binomial bound (4 of 27 pairs, bound 3). R9 passed:
  54 of 54 within-field and 473 of 473 cross-field target candidates. Reading in the source:
  retraining a cell moves an ordinary word's 25 neighbours by a median Jaccard distance of about
  0.4, against about 0.6 between consecutive decades; the top-5% candidates (median 0.89,
  minimum 0.72) clear it.
- **Downsides:** The older cleaning applies (curly apostrophes, four rule sets, regex
  tokenisers). The grid's two-threshold rule applies. The two seeds share the same text, so
  this tests training randomness, not the corpus. R3, R4 and R7 use word sets taken from round
  1's observed neighbours. R5 marks a resolution limit: the computational turn is robust in the
  full cells but does not clear the noise floor in 10,000-abstract cells. *food* is still a weak
  cross-field control (R8).
- **Status:** **superseded**, because round 4 retested R1 to R7 on the lemmatised corpus, and
  R6 (*information* converges) later failed under PPMI-SVD (T6) and on the lemmatised grid (L6).
- **Retested by the rebuild:** partly. The static grid is rerun on the rebuilt text (detection
  experiment 4; step 6b); the agenda names no retest of these word findings.
- **Source and commits:** `docs/predictions.md`, "Round 2" and "Round 2 results"; predictions
  5dec5c1 (2026-11-09), R9 added in 0d60817; results e715cc4.

### Round 3: do the findings hold under a count-based method (PPMI-SVD)? (2026-11-09)

- **Line:** static grid
- **Question and prediction(s):** Do round 2's confirmed findings hold under PPMI-SVD vectors,
  which share only the text with TWEC? T0, the median Jaccard similarity between TWEC (seed 42)
  and PPMI neighbour lists for target words in the full philosophy and HCI cells (2010s, 2020s)
  is at least 0.2. T1 to T7 restate P3, R3, R4, R7, R5, R6 and P7: T1 *representation* as P3 (main
  grid); T2 and T3 *embodiment* with at least 3 of the committed set in each decade; T4
  *affordance* at least 3 in each decade; T5 *consciousness* PPMI shift rank for 2010s → 2020s
  above 95 and at least 3 computational neighbours; T6 *information* mean PPMI shift rank gap of
  at least 15 points between the 1960s–70s and the 2000s–10s; T7 *intentionality* no transition
  with shift rank above 95.
- **Method and settings:** `ppmi grid`, `models/ppmi_grid`, unlemmatised cells; settings from
  `ROADMAP.md`: window 5, context-distribution smoothing 0.75, no shift, eigenvalue weight 0.5.
  Cells are factorised separately and not aligned, so every test uses neighbour lists or
  neighbourhood shift (`shift_rank` from `compare_cells` and `add_noise_floor`), never cosine
  drift. Disclosure: one PPMI cell was inspected (philosophy, 1990s, main grid, neighbours of
  *consciousness*).
- **Result:** 4 passed, 4 falsified. T0 passed: median Jaccard 0.32 over 144 word-cells. T1
  falsified: 1980s 1 of the symbolic set (*knowledge*), 2010s and 2020s 3 and 5 of the learned
  set. T2 passed: 4 of the set in each decade. T3 passed: 7 and 6. T4 falsified: 6 in the 2010s,
  2 in the 2020s (*experiential, qualities*). T5 falsified: shift rank 91, 1 computational
  neighbour; 2020s neighbours *qualia, orch-or, workspace, gwt, iit, integrated, substrates*. T6
  falsified: PPMI shift rank 76 and 58 (1960s–70s), 63 and 85 (2000s–10s), no convergence. T7
  passed: shift ranks 60, 79, 82, 76, none above 95. Source summary: *embodiment*'s two senses
  and *intentionality*'s stability are robust across seeds and methods; *representation* and
  *affordance* in part; *information* and *consciousness* not robust.
- **Downsides:** The older cleaning applies (curly apostrophes, four rule sets, regex
  tokenisers). The committed word sets came from TWEC observations, so a PPMI miss can reflect
  the set as well as the method (T1's list also has *declarative, procedural, frame-based*,
  outside the set; T4's leans *interactional, relational, affective, theories*). No cosine
  drift is available. Lemmatised PPMI-SVD cells were built later for the record but no
  prediction uses them.
- **Status:** **holds**, because nothing later replaced or contradicted the PPMI test; no round
  predicted on lemmatised PPMI cells.
- **Retested by the rebuild:** partly. PPMI-SVD is kept and rerun on the rebuilt text (step 1a;
  detection experiment 4; step 6b, option 4). The agenda names no retest of these word findings.
- **Source and commits:** `docs/predictions.md`, "Round 3" and "Round 3 results"; predictions
  18943e4 (2026-11-09); results 2834107.

### Round 4 (static grid): do the findings survive a lemmatised corpus? (2026-11-09)

- **Line:** static grid
- **Question and prediction(s):** Do the findings survive when plurals and British spellings are
  collapsed? L0, under 5% of control measurements are candidates and no control other than *food*
  is a candidate in more than 3 pairs. L1, every cell's vocabulary at least 4% smaller (8% in
  every main-grid and books cell) and, in the four full philosophy and HCI cells of the 2010s and
  2020s, the neighbour pool at least 3% larger. L2, *embodiment*'s two senses (at least 3 of the
  committed set each, 2020s, both seeds). L3, *intentionality* no candidate from the 1980s to the
  2020s, both seeds. L4, *representation* as P3 (seed 42). L5, HCI's *affordance* at least 3 of
  the experiential set (2010s, seed 42). L6, *information* converges (cross-field cosine rank gap
  of at least 15, seed 42, stated as a prediction for the first time). L7, philosophy's
  *consciousness* 2010s → 2020s is a candidate (direction not predicted).
- **Method and settings:** Cells rebuilt from a lemmatised corpus (`corpus lemmas learn|apply`,
  rules in `src/interpretant/corpus/lemmas.py`, overrides in `data/lemma_overrides.tsv`); verbs,
  participles and *-ing* nominalisations left as they are; phrases relearned; grid retrained with
  seeds 42 and 43; `drift grid` rerun. Word sets of rounds 1 to 3 unchanged. Disclosure: the 200
  most frequent mappings, the spelling merges and the share of each cell's vocabulary touched
  (4.6% and 5.6% of the full philosophy and HCI 2020s cells, 9.9% to 12.6% of the main-grid and
  books cells inspected) were reviewed first. The source heading says "committed 2026-11-10"; git
  dates the commit 2026-11-09.
- **Result:** 6 passed, 2 falsified (L8 to L10 are in the whole-field line). L0 passed: control
  candidates 0.4% within field (worst *forest*, 1 pair) and 0.3% cross field (worst *home*, 3
  pairs); *food* in 1 cross-field pair. L1 falsified: vocabulary shrink median 8.4%, minimum
  4.1%; 37 main-grid and books cells between 6.0% and 8.0%, under the 8% floor; neighbour pools
  shrank (−11.4% and −11.7% for philosophy 2010s and 2020s, −12.5% and −12.6% for HCI). L2
  passed in both seeds: HCI 5 of the set (*illusion, immersion, ownership, presence,
  self-location*), philosophy 6 (*affectivity, corporeality, embodied, intersubjectivity,
  spatiality, temporality*). L3 passed: no candidate in either seed; highest cosine rank 93 (seed
  42, 2000s → 2010s) with shift rank 5. L4 passed: 1980s 2 of the symbolic set (*knowledge,
  reasoning*), 2010s 3 and 2020s 5 of the learned set. L5 passed: 5 of the set. L6 falsified:
  38 and 23 (1960s, 1970s), 12 and 24 (2000s, 2010s); early mean 30, late mean 18, gap 13. L7
  passed: cosine rank 99, shift rank 96; 2020s neighbours carry both earlier readings (*qualia,
  substrate, computational, information_processing, sentience*).
- **Downsides:** The older cleaning applies, and the lemmatised path adds a further flaw:
  hyphenated words never got lemma votes. The grid's two-threshold rule applies. L1's reasoning
  was wrong (the source says so): many plural types had no singular in the cell to merge into,
  and merging frequent plurals removed pool members. L3's "stability" was later restated as "no
  change detected" because round 5 showed the grid misses most documented changes. L6's gap of 13
  missed the bar of 15.
- **Status:** **holds**, because nothing later contradicts the scored results; the reading of
  L3 changed (round 5) but the verdict did not.
- **Retested by the rebuild:** partly. The static grid is rerun on the rebuilt text (detection
  experiment 4; step 6b); the agenda names no retest of these word findings.
- **Source and commits:** `docs/predictions.md`, "Round 4" and "Round 4 results"; predictions
  6eba849 (2026-11-09; the source heading says "committed 2026-11-10"); results 87cc152.

### Round 5: does the grid find changes whose history is known? (2026-11-10)

- **Line:** static grid
- **Question and prediction(s):** Does the lemmatised grid find documented changes and stay
  quiet where nothing documented happened? E1 to E3, *attention* in AI (2010s → 2020s), *platform*
  in communication (2000s → 2010s) and *network* in AI (1980s → 1990s) are each a candidate in
  both seeds. E4 (weak case, not counted), *alignment* in AI 2010s → 2020s is a candidate in both
  seeds. E5, *attention* in AI 1990s → 2000s and in psychology 2010s → 2020s are not candidates
  in either seed. E6, *attention*'s cross-field cosine rank between AI and each of psychology,
  cognitive psychology and neuroscience is higher in the 2020s than the 1990s and at least 90 in
  the 2020s, in both seeds. Reading rule, committed first: fewer than two of E1 to E3 passing
  means the grid misses documented changes and its nulls are not evidence of stability.
- **Method and settings:** Lemmatised main grid, seeds 42 and 43 (seed 43 ranks computed with
  `compare_cells` and `add_noise_floor`); candidate as in round 1. Occurrences per cell:
  *attention* 379 and 1,121; *platform* 173 and 822; *network* 2,119 and 3,924; *alignment* 201
  and 533. Documented sources: Bahdanau et al. 2014 and Vaswani et al. 2017; Gillespie 2010;
  Rumelhart, Hinton and Williams 1986; Ouyang et al. 2022.
- **Result:** 2 passed, 4 falsified (cosine rank, neighbour rank; seed 42 then seed 43). E1
  falsified: 92 and 98, 90 and 97. E2 falsified: 61 and 13, 61 and 31. E3 falsified: 86 and 23,
  82 and 47. E4 passed: 100 and 100, 100 and 98. E5 passed: AI 1990s → 2000s 17 and 41, 47 and
  69; psychology 17 and 42, 27 and 42. E6 falsified: rises in all six comparisons but stays
  under 90 (AI × psychology 2 → 53 and 2 → 73; AI × cognitive psychology 32 → 61 and 32 → 75;
  AI × neuroscience 24 → 29 and 33 → 51). By the reading rule the grid misses documented
  changes; its candidates remain evidence, its nulls do not.
- **Downsides:** The older cleaning applies, with the lemmatised path's extra flaw (hyphenated
  words never got lemma votes). The two-threshold rule applies: *attention*'s neighbourhood
  changed more than 97–98% of words of its frequency, but its cosine drift ranked 90–92 (a near
  miss). Only four known changes, chosen with approximate dates. Decade cells may place a change
  inside a cell; the two explanations for the *platform* and *network* misses were found after
  the fact and are untested. The main grid's 10,000-abstract cells were tested, not the larger
  full cells. Check 1a later found *alignment*'s 2020s uses are only about one in seven about
  aligning AI systems, so the one hit may not be the documented change.
- **Status:** **holds**, because the README restates it as "the grid is cautious" and nothing
  later contradicts the scored verdicts.
- **Retested by the rebuild:** detection experiment 4 (step 6b tests whether the old grid's
  misses came from its rule, its windows or the embeddings); the documented changes and quiet
  transitions are checks every experiment reads (step 6).
- **Source and commits:** `docs/predictions.md`, "Round 5" and "Round 5 results"; predictions
  5c53205 (2026-11-10); results a989b14.

### Round 6: is the grid's method weak on a clean human benchmark? (2026-11-10)

- **Line:** static grid
- **Question and prediction(s):** Does the grid's method work on clean data, where only the
  method is in question? B1, on the 19 targets the grid as built can measure, cosine drift's
  agreement (Spearman) with the graded change score is at least 0.25 in both seeds. B2, the
  same on all 37 targets with the floor lowered to 85 occurrences. B3, at most 4 of the 19
  targets are candidates per seed and either none is flagged or at least 75% of the flagged are
  changed. B4, the candidate rule's accuracy on all 37 targets is above 0.568 (always predicting
  no change) in both seeds. Reading rule: B2 passing means the method is comparable to published
  systems; B2 under 0.15 in either seed means the method is weak on clean data.
- **Method and settings:** SemEval-2020 Task 1 English: two lemmatised CCOHA corpora (1810–1860,
  6.56 million tokens; 1960–2010, 6.77 million tokens), 37 targets (16 changed), binary and
  graded truth (official test data, `semeval2020_ulscd_eng.zip`). TWEC unchanged (100
  dimensions, window 5, skip-gram, 10 epochs, 5 negative samples, compass vocabulary threshold
  5), a compass trained on 6,500,000 tokens from each corpus, seeds 42 and 43. `compare_cells`
  and `add_noise_floor` as the grid uses them (25 neighbours from words with at least 50
  occurrences in both corpora). Published references: best system 0.436, static systems
  0.30–0.42 (SGNS with orthogonal Procrustes 0.422), count baseline 0.022; binary accuracy 0.730
  best, 0.568 for always no change.
- **Result:** 3 passed, 1 falsified (seed 42; seed 43). B1 falsified: ρ 0.219; 0.211. B2 passed:
  0.285; 0.295. B3 passed: one target flagged of 19 in each seed, *plane* (changed). B4 passed:
  accuracy 0.595; 0.595 (majority 0.568), from the single correct flag. By the reading rule the
  method is comparable to published systems at the low end of the static range, and the grid's
  misses on this corpus come from the cells or the candidate rule, not the method.
- **Downsides:** Small benchmark: 37 targets, only 19 measurable at the 150 floor. The grid's
  two-threshold rule is the source of the caution (B3 flagged one of 19 where 16 of 37 changed).
  B4's pass rests on one correct flag. During training gensim printed several hundred warnings
  ("Exception ignored in 'gensim.models.word2vec_inner.our_dot_float'"); results are in the
  expected range, but the warning may signal a numerical problem, and it is not known whether
  the grid's own training printed it (its logs were not kept). Different corpus from the
  project's, so the older cleaning does not apply.
- **Status:** **holds**, because nothing later contradicts it.
- **Retested by the rebuild:** detection experiment 4 (step 6b decides the static model "on the
  synthetic set and the benchmark"; the agenda does not say which benchmark).
- **Source and commits:** `docs/predictions.md`, "Round 6" and "Round 6 results"; predictions
  ebbb298 (2026-11-10); results 0859135.

## Whole-field comparison

Bigram language models and field classifiers compared fields' whole abstracts. This line left the
project (decision agenda step 1a), so none of these entries is retested. Every entry below uses
the processed (static) text, so the older cleaning applies to all of them; from round 4 on the
text is also lemmatised, with the extra flaw that hyphenated words never got lemma votes. Source:
`docs/predictions_cross_perplexity.md`.

### First cross-perplexity run: who predicts whom? (2026-11-09)

- **Line:** whole-field comparison
- **Question and prediction(s):** How predictable is one field's language from another's? Bars as
  written: C0, for at least 95% of qualifying cells the field's own model gives its held-out text
  the lowest cross-entropy. C1, E(computer science → AI) < E(AI → computer science) and
  E(psychology → cognitive psychology) < E(cognitive psychology → psychology), each in at least 6
  decades (or every decade, if fewer than 6). C2, computer science has the lowest excess for AI's
  text and psychology for cognitive psychology's, each in at least 6 decades. C3, E(AI →
  philosophy) is lower in the 2020s than the 2010s. C4, E(philosophy → HCI) is lower in the 2020s
  than the 1990s. C5, the mean E(A → B) over all ordered pairs is higher in the 2010s than the
  1980s (described as a genuinely open question).
- **Method and settings:** Main OpenAlex grid only (`data/phrased/openalex/texts`), unlemmatised.
  One bigram model per cell, interpolated Kneser-Ney, discount 0.75. Within a decade every
  qualifying field trains on the same number of tokens (the smallest qualifying field's, capped
  at 1,000,000; a field qualifies with at least 200,000 training tokens). Held out: 1,000
  abstracts per cell, fixed seed. One vocabulary per decade (words with at least 5 occurrences).
  Excess cross-entropy E(A → B) is B's held-out bits per token under A's model minus under B's
  own model (Vilhena et al. 2014).
- **Result:** 5 passed, 1 falsified. C0 passed: own model best in 76 of 79 cells (96%); the 3
  misses are AI and computer science in the 2010s–2020s. C1 falsified: computer science → AI
  below AI → computer science in 5 of 6 decades; psychology → cognitive psychology below the
  reverse in 2 of 7. C2 passed: computer science closest to AI in 6 of 6 decades, psychology
  closest to cognitive psychology in 7 of 7. C3 passed: 1.61 bits (2010s), 1.56 (2020s). C4
  passed: 2.12 bits (1990s), 1.94 (2020s). C5 passed: mean excess over 11 fields 1.08 bits
  (1980s), 1.20 (2010s).
- **Downsides:** Three design flaws, found after the run. Parent and subfield cells share
  abstracts (18–29% of AI's abstracts are also in computer science's cell, 10–20% of cognitive
  psychology's in psychology's), so a parent's model could train on the subfield's held-out text;
  this produced negative excess values and C0's three misses and favours the parent in C1 and
  C2. Training size was equal within decades, not across them (C4 compares 858,655 with
  1,000,000 tokens; C5 compares 205,731 with 1,000,000), so C5 may reflect training size. No
  uncertainty estimate (C3's change is 0.05 bits, 3%). Single seed. The older cleaning applies.
- **Status:** **superseded**, because the corrected run replaced it and reversed C1 and C5.
- **Retested by the rebuild:** no (whole-field comparison left the project).
- **Source and commits:** `docs/predictions_cross_perplexity.md`, "Predictions" and "Results
  (2026-11-09, `scripts/score_cross_perplexity.py`)", with "Design flaws found after the run";
  predictions 9a143c4 (2026-11-09); results 17ce4a0.

### Corrected cross-perplexity run: do the results hold without the design flaws? (2026-11-09)

- **Line:** whole-field comparison
- **Question and prediction(s):** Do the first run's results hold with shared abstracts removed,
  one training size and five seeds? D0, in every seed own model best in at least 95% of cells.
  D1, using the mean excess over 5 seeds, E(computer science → AI) < E(AI → computer science)
  and E(psychology → cognitive psychology) < E(cognitive psychology → psychology), each in at
  least 6 decades. D2, computer science is closest to AI and psychology closest to cognitive
  psychology in at least 6 decades. D3, E(AI → philosophy) lower in the 2020s than the 2010s in
  all 5 seeds. D4, E(philosophy → HCI) lower in the 2020s than the 1990s in all 5 seeds. D5, mean
  E(A → B) over all pairs higher in the 2010s than the 1980s in all 5 seeds.
- **Method and settings:** As the first run, except: any abstract in more than one cell of a
  decade is removed from every cell of that decade; every qualifying field in every decade
  trains on exactly 200,000 tokens (qualifies with at least 200,000 tokens after removal); seeds
  1 to 5, each with its own held-out abstracts and training sample. Unlemmatised.
- **Result:** 3 passed, 3 falsified (two reversed). D0 passed: own model best in 100% of cells in
  every seed. D1 falsified, reversed: parent → subfield lower in 0 of 6 decades for both pairs.
  D2 passed: 6 of 6 decades for both. D3 falsified: E(AI → philosophy), 2020s minus 2010s,
  −0.055, −0.053, −0.041, +0.024, +0.005. D4 passed: E(philosophy → HCI), 2020s minus 1990s,
  −0.072, −0.103, −0.065, −0.056, −0.096. D5 falsified, reversed: mean excess over 10 fields,
  2010s minus 1980s, −0.042, −0.046, −0.040, −0.061, −0.056.
- **Downsides:** The older cleaning applies. Each model trains on only 200,000 tokens. Five seeds
  resample sampling noise only. The source's first reading of D4 ("HCI's language moved towards
  philosophy's") was wrong: E(A → B) also falls when A's language broadens to cover B (see the
  follow-up and rounds G and J). The reversed D1 and D5 were observations needing their own
  predictions; round 4 made them (L10 and L9).
- **Status:** **holds**, as a set of measurements: D1, D4 and D5 were replicated on the lemmatised
  corpus as L10, L8 and L9. The reading of D4 was corrected later (see the follow-up).
- **Retested by the rebuild:** no (whole-field comparison left the project).
- **Source and commits:** `docs/predictions_cross_perplexity.md`, "Corrected run" and "Corrected
  run results"; predictions 8468bb7 (2026-11-09); results fd2086b.

### HCI and philosophy follow-up: where does the philosophy → HCI gain come from? (2026-11-09)

- **Line:** whole-field comparison
- **Question and prediction(s):** F1, HCI's move towards philosophy is concentrated after 2000:
  mean E(philosophy → HCI) falls more from the 2000s to the 2020s than from the 1990s to the
  2000s. F2, the movement is one-sided: in every seed E(HCI → philosophy) changes less between
  the 1990s and 2020s, in absolute value, than E(philosophy → HCI). F3, at least 2 of {embodiment,
  embodied, affordance, experience, experiential, phenomenological, phenomenology, body, bodily,
  meaning, practice, perception} are among the top 50 words ranked by fall in per-word excess
  (weighted by 2020s occurrences).
- **Method and settings:** The corrected run's table
  (`data/drift/cross_perplexity_corrected.parquet`: shared abstracts removed, 200,000 training
  tokens, seeds 1 to 5, unlemmatised). Only the 1990s and 2020s values of E(philosophy → HCI)
  had been inspected. Per-word excess is the bits per occurrence that HCI's own model saves over
  philosophy's on HCI's held-out text, pooled over the 5 seeds.
- **Result:** 1 passed, 2 falsified. F1 falsified: mean E(philosophy → HCI) 1.700 (1990s), 1.652
  (2000s), 1.521 (2010s), 1.622 (2020s); fall 1990s → 2000s 0.048, 2000s → 2020s 0.031. F2 passed:
  per seed, change in E(philosophy → HCI) against E(HCI → philosophy): (−0.072, +0.018), (−0.103,
  +0.002), (−0.065, −0.040), (−0.056, +0.003), (−0.096, −0.006). F3 falsified: none of the 12
  words in the top 50, which is led by *the, user, design, system, of, interface, research,
  virtual, digital, interaction, environment*. An exploratory count after scoring: per million
  tokens, philosophy uses *digital* 6 → 126, *interface* 10 → 49, *virtual* 10 → 40, *user* 8 →
  29, *design* 47 → 134 between the 1990s and 2020s.
- **Downsides:** The older cleaning applies. F3 ranks words weighted by occurrences, so very
  frequent words dominate (the source says a per-occurrence ranking would be better). The
  reinterpretation added in the same section (philosophy turning towards technology) rests on an
  exploratory count, not a test; round H then falsified it. The README and ROADMAP were corrected
  after this round.
- **Status:** **holds**, as scored. Its reinterpretation (philosophy broadening) was falsified by
  the technology-turn round and replaced by the account in rounds G and J.
- **Retested by the rebuild:** no (whole-field comparison left the project).
- **Source and commits:** `docs/predictions_cross_perplexity.md`, "Follow-up: HCI and
  philosophy", "Follow-up results" and "Reinterpretation of D4 and F2"; predictions f9af0b2
  (2026-11-09); results f3a6044.

### Philosophy's turn towards technology: did philosophy broaden into technology? (2026-11-09)

- **Line:** whole-field comparison
- **Question and prediction(s):** H1, in all 5 seeds E(philosophy → computer science) and
  E(philosophy → AI) are each lower in the 2020s than the 1990s. H2, in all 5 seeds
  E(computer science → philosophy) changes less, in absolute value, than E(philosophy → computer
  science). H3, in the full philosophy cells the share of abstracts containing at least one of 18
  technology words (*internet, online, software, computer, ... platforms*) rises at every step
  from the 1980s to the 2020s. H4, on computer science's held-out text, at least 3 of {ethics,
  ethical, social, society, human, humans, privacy, trust, moral, responsibility, values,
  fairness} are in the top 50 words by fall in per-occurrence excess (words with at least 100
  pooled occurrences in each decade).
- **Method and settings:** The corrected run's models (seeds 1 to 5, 200,000 tokens, unlemmatised)
  and the full philosophy cells. The technology lexicon excludes every word of the exploratory
  count (*digital, virtual, design, user, interface, technology, algorithm*). Words ranked per
  occurrence, not weighted by frequency.
- **Result:** 1 passed (by the letter only), 3 falsified. H1 falsified: change 1990s → 2020s in
  E(philosophy → computer science) −0.025, +0.016, +0.079, +0.103, +0.109; in E(philosophy → AI)
  +0.060, +0.031, +0.035, +0.003, −0.013. H2 passed by the letter only: E(computer science →
  philosophy) changed less in every seed (+0.001, +0.010, −0.048, −0.014, +0.015), but E(philosophy
  → computer science) mostly rose, so the pass does not support the hypothesis. H3 falsified:
  share with a technology word 3.1% (1980s), 3.3%, 2.9%, 3.6%, 6.5% (2020s). H4 falsified: none
  of the 12 words in the top 50, led by *digital, signal, compression, memory, interface,
  implementation*.
- **Downsides:** The older cleaning applies. H2's pass is by the letter only (source). The word
  lists in H3 and H4 are committed choices. The source's reading: neither explanation tested so
  far (HCI borrowing philosophy's language, philosophy broadening into technology) accounts for
  the philosophy → HCI result, and its cause was left open.
- **Status:** **holds**, as scored: the hypothesis it tested is falsified, and no later round
  contradicts that.
- **Retested by the rebuild:** no (whole-field comparison left the project).
- **Source and commits:** `docs/predictions_cross_perplexity.md`, "Philosophy's turn towards
  technology" and "Technology-turn results"; predictions 7b2456e (2026-11-09); results ae05a41.

### Round 4 (whole-field): do the cross-perplexity effects hold when lemmatised? (2026-11-09)

- **Line:** whole-field comparison
- **Question and prediction(s):** L8, E(philosophy → HCI) is lower in the 2020s than the 1990s in
  all 5 seeds, and E(HCI → philosophy) changes less in absolute value in every seed (D4 and F2
  again). L9, the mean E(A → B) over all pairs is lower in the 2010s than the 1980s in all 5
  seeds (D5's reversal, now predicted). L10, using the mean excess over 5 seeds, E(AI → computer
  science) < E(computer science → AI) and E(cognitive psychology → psychology) < E(psychology →
  cognitive psychology) in at least 5 of the 6 decades where both qualify (D1's reversal).
- **Method and settings:** `drift perplexity` rerun on the lemmatised cells with the corrected
  design (shared abstracts removed, 200,000 training tokens, 1,000 held-out abstracts, seeds 1 to
  5). Absolute excess values are not compared across the lemmatised and unlemmatised corpora.
- **Result:** 3 passed. L8 passed: change 1990s → 2020s in E(philosophy → HCI) −0.071, −0.103,
  −0.066, −0.057, −0.097; in E(HCI → philosophy) +0.014, +0.001, −0.045, +0.002, −0.005, smaller
  in absolute value in every seed. L9 passed: mean excess over 10 fields, 2010s minus 1980s,
  −0.050, −0.049, −0.040, −0.059, −0.056. L10 passed: AI → computer science below the reverse in
  6 of 6 decades; cognitive psychology → psychology below the reverse in 6 of 6.
- **Downsides:** The older cleaning applies, with the lemmatised path's extra flaw (hyphenated
  words never got lemma votes). The source reads the values as within a few thousandths of a bit
  of the unlemmatised run and leaves the cause of the philosophy → HCI effect open. Later rounds
  qualify L9: fields converge under the language models but not under a classifier (round S).
- **Status:** **holds**, because nothing later contradicts the scored verdicts; the README
  reports the asymmetry as robust and the convergence as the language models' finding only.
- **Retested by the rebuild:** no (whole-field comparison left the project).
- **Source and commits:** `docs/predictions_cross_perplexity.md`, "Round 4: the lemmatised
  corpus" and "Round 4 results"; predictions 6eba849 (2026-11-09; the source heading says
  "committed 2026-11-10"); results 87cc152.

### Round G: where in HCI's text does the philosophy → HCI gain sit? (2026-11-09)

- **Line:** whole-field comparison
- **Question and prediction(s):** Is the fall in E(philosophy → HCI) carried by a subset of HCI
  abstracts or spread across all of them? G1, in every seed the median per-abstract excess falls
  by at least 70% of the mean's fall. G2, in every seed the fall of the 25th percentile is at
  most 1.5 times the fall of the 75th. G3, in at least 4 of 5 seeds the fall over HCI abstracts
  with none of 24 philosophy theme words is at least 60% of the fall over those with one. G4, the
  top 50 words account for at least 40% of the sum of positive per-word scores. G5, in every seed
  the median per-abstract excess of philosophy's abstracts under HCI's model changes by less than
  half the fall of HCI's median under philosophy's model. A reading rule was written first: G1 to
  G3 passing together means a register effect, not topics.
- **Method and settings:** Lemmatised cells, the corrected design (shared abstracts removed,
  200,000 training tokens, 1,000 held-out abstracts, seeds 1 to 5), decades 1990 and 2020.
  Per-abstract excess is bits per token under philosophy's model minus under HCI's own model.
- **Result:** 3 passed, 2 falsified. G1 passed: median fall / mean fall 0.84, 1.39, 1.21, 1.53,
  0.84. G2 falsified: q25 fall / q75 fall 1.00, 1.63, 0.55, 2.16, 0.82 (over 1.5 in seeds 2 and
  4). G3 passed: plain against flavoured falls (0.144, −0.026), (0.092, 0.072), (0.030, 0.070),
  (0.069, −0.005), (0.082, 0.032); plain at or above 60% of flavoured in 4 of 5 seeds. G4
  falsified: the top 50 words hold 38.8% (needed 40%); led by *the, user, design, system, of,
  interface, research, interaction*. G5 passed: (0.120, +0.031), (0.190, +0.022), (0.109,
  −0.007), (0.163, +0.026), (0.093, −0.003). Observed, not predicted: the share of HCI abstracts
  with at least one theme word rose from 22–24% in the 1990s to 48–52% in the 2020s in every
  seed. Exploratory mix and rate decomposition (seed by seed): fall 0.071, 0.103, 0.066, 0.057,
  0.097 bits; mix effect 0.039, 0.064, 0.068, 0.086, 0.091; rate effect 0.032, 0.039, −0.001,
  −0.029, 0.005.
- **Downsides:** The older cleaning applies, with the lemmatised path's extra flaw. G2 is a noisy
  statistic (quartile falls swing from 0.55 to 2.16 across seeds). The source's first reading,
  "philosophy acquired HCI's objects", was replaced after the exploratory decomposition: the
  larger part is HCI's word mix moving away from its 1990s GUI-engineering vocabulary (*interface,
  user_interface, menu, display, graphical, software*) towards *research, study, participant*.
  The exploratory sections are unscored. The HCI label's widening (found in round K) was not yet
  known.
- **Status:** **holds**, as scored. The two readings it first offered (register effect;
  philosophy acquiring HCI's objects) were corrected by its own exploratory decomposition and by
  round J.
- **Retested by the rebuild:** no (whole-field comparison left the project).
- **Source and commits:** `docs/predictions_cross_perplexity.md`, "Round G", "Round G results",
  "Exploratory, after scoring: function words, and a mix/rate decomposition" and "Corrected
  reading"; predictions 0cb3971 (2026-11-09); results 37ae490; exploratory correction 28b1459.

### Round J: is the gain on HCI every field's gain, through HCI's word mix? (2026-11-09)

- **Line:** whole-field comparison
- **Question and prediction(s):** If HCI's own word mix moved, every field's model should predict
  HCI's 2020s abstracts better. J1, for at least 8 of the 9 other models mean E(X → HCI) is lower
  in the 2020s than the 1990s. J2, for at least 6 of 9 it is lower in all 5 seeds. J3, for
  psychology's and sociology's models the mix effect exceeds the rate effect in at least 4 of 5
  seeds. J4, for at least 7 of 9 the fall on HCI exceeds the median fall of E(X → Y) over other
  text fields Y.
- **Method and settings:** The corrected lemmatised table
  (`data/drift/cross_perplexity_corrected.parquet`, seeds 1 to 5, 200,000 training tokens). The
  nine other models: AI, cognitive psychology, communication, computer science, history and
  philosophy of science, linguistics, neuroscience, psychology, sociology. Mix and rate effects as
  in round G's exploratory decomposition.
- **Result:** 3 passed, 1 falsified. J1 falsified: 7 of 9 models gain (needed 8); falls of mean
  E(X → HCI): neuroscience 0.640, sociology 0.256, psychology 0.252, communication 0.110,
  cognitive psychology 0.082, linguistics 0.048, history and philosophy of science 0.015; AI
  −0.033, computer science −0.004. J2 passed: 6 of 9 fall in all 5 seeds. J3 passed: mix exceeds
  rate in 5 of 5 seeds for both; psychology (fall, mix, rate) ≈ (0.25, 0.25, 0.01), sociology ≈
  (0.26, 0.18, 0.07). J4 passed: 7 of 9 fall more on HCI than on other texts; the exceptions are
  AI (−0.033 vs 0.079) and computer science (−0.004 vs 0.105).
- **Downsides:** The older cleaning applies, with the lemmatised path's extra flaw. The HCI cell
  is a label that widened over time (915 sources in the 1990s, 3,601 in the 2020s, found in round
  K), so whole-cell gains mix the field's move with the label's. The source's reading, that HCI's
  language "moved out of computing's register and towards the behavioural and social sciences'",
  rests on language models that weigh every word by frequency; round N's classifier disagreed on
  direction, and the README says neither that statement nor "HCI and psychology met" survives
  round N.
- **Status:** **holds**, reading corrected: the measured gains stand and the README reports them,
  but the direction reading the round gave them was later withdrawn by the README.
- **Retested by the rebuild:** no (whole-field comparison left the project).
- **Source and commits:** `docs/predictions_cross_perplexity.md`, "Round J" and "Round J
  results"; predictions 008d3e5 (2026-11-09); results b9e6c31.

### Round K: is the move the field's or the label's? (2026-11-09)

- **Line:** whole-field comparison
- **Question and prediction(s):** K1, E(psychology → HCI_v) and E(sociology → HCI_v) are lower
  in the 2020s than the 1990s in all 5 seeds. K2, the mix effect exceeds the rate effect in at
  least 4 of 5 seeds for both. K3, the mean fall on HCI_v is at least half the mean fall on the
  whole cell for both. K4, the fall of E(AI → HCI_v) is less than a third of psychology's.
- **Method and settings:** HCI_v is the venue-stable cell: cleaned HCI documents from sources
  with at least 100 cleaned documents in both decades (OpenAlex `primary_location.source`).
  Corrected design on the lemmatised text, 100,000-token models, 500 held-out abstracts, seeds 1
  to 5; the round is untestable if HCI_v has under 100,000 training tokens in either decade.
- **Result:** Untestable as written. Shared venues were CHI (305 / 218 documents) and CHI
  Extended Abstracts (258 / 155) only; HCI_v is 563 documents in the 1990s and 373 in the 2020s,
  and the 1990s training pool after holding out 500 abstracts is 5,913 tokens. No excess was
  computed. Reported: the 1990s cell (7,294 documents) spans 915 sources, 1,271 documents without
  one; the 2020s cell (9,665 documents) spans 3,601 sources, 177 without; the shared-venue share
  fell from 7.7% to 3.9%. The sources that grew most as a share included IEEE VR workshops
  (VRW), Zenodo, AHFE International, Routledge eBooks, *Sensors* and *Frontiers in Virtual
  Reality*.
- **Downsides:** The 100-document rule left too few venues and too few tokens (small cells: 563
  and 373 documents). The older cleaning applies, with the lemmatised path's extra flaw. The
  venue lookup shows the HCI label is far more diffuse in the 2020s than the 1990s, whatever the
  prose did.
- **Status:** **untestable**, because it could not be scored as written; round K′ repeated it at
  feasible sizes.
- **Retested by the rebuild:** no (whole-field comparison left the project).
- **Source and commits:** `docs/predictions_cross_perplexity.md`, "Round K" and "Round K results";
  predictions ec7b188 (2026-11-09); the untestable result is recorded in bb1d546.

### Round K′: the same venue test at feasible sizes (2026-11-09)

- **Line:** whole-field comparison
- **Question and prediction(s):** K1 to K4 as in round K (the prediction bars are unchanged).
- **Method and settings:** As round K with three sizes changed before any excess was computed:
  shared venues need at least 20 cleaned documents in both decades (9 venues: CHI, CHI EA, UIST,
  VRST, IJHCI, HFES proceedings, CRC Press eBooks and others; 1,471 / 856 documents), 150
  held-out abstracts per text, 90,000-token models. Seeds 1 to 5, lemmatised, corrected design.
- **Result:** 3 passed, 1 falsified. K1 passed: falls on HCI_v, psychology 0.054, 0.163, 0.109,
  0.025, 0.108; sociology 0.243, 0.346, 0.290, 0.322, 0.231. K2 passed: mix > rate in 5 of 5
  for both; psychology (fall, mix, rate) ≈ (0.09, 0.17, −0.07), sociology ≈ (0.29, 0.22, 0.06).
  K3 falsified: psychology 0.092 / 0.184, just under 0.50 (under by less than a thousandth);
  sociology 0.286 / 0.234 = 1.22. K4 passed: AI's mean fall on HCI_v −0.053 against psychology's
  0.092. Shared venues hold 20.2% of the 1990s cleaned cell and 8.9% of the 2020s one (CHI 305 /
  218, CHI EA 258 / 155, HFES proceedings 440 / 35).
- **Downsides:** Small cells: HCI_v is 90,000-token models with 150 held-out abstracts. The cell
  holds venues fixed, not their shares (HFES proceedings 440 / 35 documents). The older cleaning
  applies, with the lemmatised path's extra flaw. The label widened (915 to 3,601 sources), so
  how much of round J's size is the field's depends on the model: for sociology's the effect is
  larger inside fixed venues, for psychology's about half.
- **Status:** **holds**, because the README relies on it for the change inside HCI's own venues.
  The direction reading it supported was later withdrawn (see round J).
- **Retested by the rebuild:** no (whole-field comparison left the project).
- **Source and commits:** `docs/predictions_cross_perplexity.md`, "Round K′" and "Round K′
  results"; predictions bb1d546 (2026-11-09); results 4f366d5.

### Round S: can a classifier tell the fields apart, and has HCI moved? (2026-11-09)

- **Line:** whole-field comparison
- **Question and prediction(s):** Does an instrument that shares only the corpus with the
  language models reproduce round J? S0, every pair's mean accuracy is at least 0.75 in the 1990s
  and 2020s and the three parent–subfield pairs (computer science–AI, computer science–HCI,
  psychology–cognitive psychology) are among the five least separable. S1, for at least 4 of the 5
  behavioural and social fields the HCI–X accuracy is lower in the 2020s than the 1990s in every
  seed comparison. S2, for computer science and AI the 2020s HCI–X accuracy is at least the 1990s
  mean minus 0.01. S3, the 1990s and 2020s classifiers send more of HCI's abstracts to the
  behavioural and social fields (at least 4 of 5 seeds) and fewer to computing (at least 4 of 5).
  S4, the 1990s classifier sends a larger share of HCI's 2020s than 1990s abstracts to the
  behavioural and social fields (4 of 5 seeds) and no larger share of psychology's (and of
  sociology's) 2020s abstracts to HCI. S5, mean pairwise accuracy is lower in the 2010s than the
  1980s in all 5 seeds.
- **Method and settings:** Lemmatised, phrased main grid; shared abstracts removed per decade;
  2,000 abstracts per field per decade per seed (1 to 5), decades 1980s–2020s. TF-IDF
  (unigrams, sublinear term frequency, words in at least 2 training abstracts) and logistic
  regression (L2, C = 1); pairwise 5-fold cross-validation; one eleven-way classifier per decade
  (1,600 train, 400 held out); the 1990s classifier also applied to 400 abstracts per field from
  the 2020s.
- **Result:** 1 passed, 5 falsified. S0 falsified: lowest mean accuracy 0.848 (1990s), 0.836
  (2020s); computer science–AI and psychology–cognitive psychology are in the bottom five,
  computer science–HCI is not (0.929 and 0.932). S1 falsified: 3 of 5 (needed 4). S2 passed:
  HCI–computer science 0.929 → 0.932; HCI–AI 0.949 → 0.956. S3 falsified: share to behavioural and
  social fields rises in 4 of 5 seeds (about 0.06 → 0.07), share to computing falls in only 2 of
  5. S4 falsified: HCI's part passes in 5 of 5 seeds (0.055–0.070 → 0.110–0.123); psychology's
  2020s abstracts go to HCI more often (0.003–0.013 → 0.018–0.028), and so do sociology's
  (0.000–0.007 → 0.010–0.022). S5 falsified: 0.9488 → 0.9487 in seed 1, a small rise in seeds
  2 to 5. Exploratory: a classifier trained on the 1990s recognises fewer 2020s abstracts for
  most fields (computer science 0.796 → 0.621, neuroscience 0.841 → 0.634, AI 0.772 → 0.624, HCI
  0.865 → 0.748); only psychology, cognitive psychology and linguistics hold steady.
- **Downsides:** The older cleaning applies, with the lemmatised path's extra flaw. The effect is
  small against a ceiling (pairwise accuracies move by about 0.01 around 0.95–0.97). The
  classifier is symmetric between two fields, so direction rests on S4's cross-decade design.
  Every field's 2020s abstracts leave home more often under a 1990s classifier, so raw shares
  mislead. The exploratory note that technology vocabulary explains the drift towards HCI was
  tested in round V and was wrong; the note's "ten of the eleven fields" was corrected to 9 of
  the 10 non-HCI fields.
- **Status:** **holds**, as scored. The exploratory technology-vocabulary reading was withdrawn
  by round V.
- **Retested by the rebuild:** no (whole-field comparison left the project).
- **Source and commits:** `docs/predictions_cross_perplexity.md`, "Round S" and "Round S results";
  predictions f4a5a6d (2026-11-09); results 22e2180.

### Round V: is the drift towards HCI technology vocabulary? (2026-11-09)

- **Line:** whole-field comparison
- **Question and prediction(s):** If the drift towards HCI is technology vocabulary, removing it
  should end the drift. V0, every pair's mean accuracy stays at least 0.75 in both decades. V1,
  without the terms the drift towards HCI is zero or negative in at least 4 of 5 seeds for
  psychology and for sociology. V2, at most 5 of the 10 non-HCI fields have a positive mean
  drift towards HCI. V3, HCI's move survives: the 1990s classifier sends more of HCI's 2020s
  abstracts than 1990s ones to the behavioural and social fields in at least 4 of 5 seeds.
- **Method and settings:** Round S's method and samples (same seeds), 1990s and 2020s only, with
  every token that is one of 29 listed terms (`data/technology_terms.txt`: *technology,
  technological, digital, online, internet, web, website, computer, computing, computational,
  software, hardware, device, mobile, smartphone, app, interface, user, algorithm, algorithmic,
  automation, automated, robot, robotic, virtual, platform, machine, ai, artificial*), or a
  phrase containing one, removed.
- **Result:** 2 passed, 2 falsified. V0 passed: lowest mean pair accuracy 0.848 (1990s), 0.826
  (2020s). V1 falsified: the drift towards HCI is still positive in every seed (psychology +0.013
  to +0.035, sociology +0.002 to +0.022). V2 falsified: 9 of 10 non-HCI fields still drift
  towards HCI (largest computer science +0.026, psychology +0.024, history and philosophy of
  science +0.015; only cognitive psychology does not, −0.001). V3 passed: 0.055–0.082 →
  0.120–0.158 in 5 of 5 seeds. Exploratory: without the terms HCI's largest gain is towards
  psychology (+0.039), with them towards AI (+0.038).
- **Downsides:** The older cleaning applies, with the lemmatised path's extra flaw. The 29-term
  list is a committed choice. Raw shares are subject to the drift every field shows (round N).
  The exploratory reading, that HCI and psychology "met, with HCI travelling further", did not
  survive round N (README).
- **Status:** **holds**, as scored.
- **Retested by the rebuild:** no (whole-field comparison left the project).
- **Source and commits:** `docs/predictions_cross_perplexity.md`, "Round V" and "Round V
  results"; predictions 0157f74 (2026-11-09); results 98b465c.

### Round M: does every cell's language model generalise? (2026-11-09)

- **Line:** whole-field comparison
- **Question and prediction(s):** M0, over all cells the Spearman correlation between a cell's
  training tokens and its mean gap is −0.8 or stronger. M1, HCI's main-grid cells of the 1950s,
  1960s and 1970s (18, 152 and 459 abstracts) and AI's of the 1950s (234) are all flagged. M2,
  none of the cells behind the robust findings is flagged (philosophy's full cells 1980s–2020s,
  HCI's 2000s–2020s, AI's main-grid 1980s and 2000s–2020s, and every main-grid cell of the
  1980s, 1990s, 2010s and 2020s).
- **Method and settings:** Every lemmatised, phrased cell (books, main grid, full cells); 10% of
  documents held out, the rest capped at 5,000,000 training tokens; seeds 1 to 3; a Kneser-Ney
  bigram model. The gap is held-out cross-entropy minus the cross-entropy of a same-size sample
  of training documents. A cell is flagged above 1.5 times the median gap of the 33 main-grid
  cells of the 2000s–2020s. Method amendment (c3bd246, before any corpus score): the
  vocabulary is every word of the training part, because a 5-occurrence floor made small cells'
  gaps smaller than large cells' in synthetic tests.
- **Result:** 2 passed, 1 falsified. The reference median gap is 1.787 bits, so cells are flagged
  above 2.681 bits. M0 falsified: ρ = −0.604 over 117 cells. M1 passed: all four flagged (HCI
  1950s 4.80, 1960s 3.91, 1970s 3.21; AI 1950s 3.72). M2 passed: none of the 53 cells the robust
  findings use is flagged. Eleven cells are flagged: books 2000s (3.16) and 2020s (4.00); main
  grid AI 1950s, cognitive psychology 1950s (2.94), communication 1950s (3.32) and 1960s (3.01),
  HCI 1950s, 1960s and 1970s, linguistics 1950s (3.32) and 1960s (2.83).
- **Downsides:** Small cells are the thing measured: the flagged cells are the thinnest ones.
  Size alone does not drive the gap (M0 falsified): documents predict it better than tokens
  outside the main grid, books are unstable (one held-out book changes a cell's gap), and above
  about 10,000 abstracts the gap saturates at 1.2–1.8 bits. The committed measure needed an
  amendment. The older cleaning applies, with the lemmatised path's extra flaw. The books layer
  was retired on 2026-11-11.
- **Status:** **holds**, as scored.
- **Retested by the rebuild:** no (whole-field comparison left the project).
- **Source and commits:** `docs/predictions_cross_perplexity.md`, "Round M", "Method amendment"
  and "Round M results"; predictions ebe35c0 (2026-11-09); amendment c3bd246; results 98f46af.

### Round N: who moved towards whom, net of every field's drift? (2026-11-10)

- **Line:** whole-field comparison
- **Question and prediction(s):** Net of the drift all fields share, did HCI move towards the
  behavioural and social sciences, and psychology towards HCI? N1, HCI's relative gain towards
  the behavioural and social group is present at the 2020s under both the 1980s and 1990s
  classifiers. N2, under the 1990s classifier it is already present at the 2000s. N3, HCI's
  relative gain towards computing is not present at the 2020s under either classifier. N4,
  psychology's relative gain towards HCI is present at the 2020s under the 1980s classifier.
  N5, under the 1990s classifier it is not present at the 2000s. "Present" means positive in all
  five seeds with a mean at least twice its standard deviation.
- **Method and settings:** Round S's samples and classifier (2,000 abstracts per field, 1,600
  train, 400 held out, seeds 1 to 5), trained on the 1980s and on the 1990s and applied to each
  later decade. The relative gain of field A towards target B is A's change in the share sent to
  B minus the median change of every other field. A synthetic calibration (six fields sharing 95%
  of their words, planted moves) found planted moves of 5% and 10% (+0.041 and +0.085) and no
  false move in any tested setting.
- **Result:** 1 passed, 4 falsified. N1 falsified: present under the 1980s classifier (+0.100, sd
  0.036), absent under the 1990s (−0.001, sd 0.013). N2 falsified: +0.001, sd 0.013, absent. N3
  falsified: present under both (+0.110, sd 0.017 from the 1980s; +0.046, sd 0.018 from the
  1990s). N4 falsified: +0.001, sd 0.010, absent. N5 passed: +0.009, sd 0.006, absent. A
  correction (2026-11-10, after a synthetic run for round W): a field that changes more than the
  rest produces present relative gains towards groups it never borrowed from (HCI replacing 45%
  of its words gave +0.057, sd 0.014 towards a two-field group; 80% gave a move towards every
  target). The source says the verdicts stand as scored; N1's 1980s part and the 1980s rows
  "are what this artefact produces", N3 under the 1990s classifier is "unconfirmed rather than
  refuted", and N4 and N5 are unaffected.
- **Downsides:** The relative gain has a blind spot the calibration did not plant: it removes
  drift every field shares, not one field changing more than the rest (HCI's home share under the
  1980s classifier falls from 0.87 to 0.56). The older cleaning applies, with the lemmatised
  path's extra flaw. The five seeds resample sampling noise only. The source's proposed
  text-side correction was superseded by round W.
- **Status:** **holds**, reading corrected: the verdicts stand as scored, but the source's own
  correction shows that a field which changes more than the rest produces apparent moves, so N1's
  1980s part and the 1980s rows are likely that artefact, N3 under the 1990s classifier is
  unconfirmed rather than refuted, and the README says the measure can report a move for a field
  that only changes more than the rest.
- **Retested by the rebuild:** no (whole-field comparison left the project).
- **Source and commits:** `docs/predictions_cross_perplexity.md`, "Round N", "Round N results",
  "Exploratory, after scoring: the decade-by-decade paths" and "Correction (2026-11-10): the
  blind spot, calibrated"; predictions 59f9359 (2026-11-10 at 00:00 -0300; the source heading
  says "committed 2026-11-09"); results 643a201; correction 3feb793.

### Round W: the frame of reference, or what the instrument weighs? (2026-11-10)

- **Line:** whole-field comparison
- **Question and prediction(s):** Holding the frame and the baseline fixed, do the classifier and
  the language models agree on HCI's direction? W1, under both instruments every field's own 1990s
  model gives its 1990s held-out abstracts the lowest cost, in every seed. W2, the classifier's
  contrast shows HCI moving towards computing over the behavioural and social sciences. W3, the
  language models' contrast shows the opposite. W4, W2 inside the venue-stable cell. W5, W3
  inside the venue-stable cell. Each is falsified if absent or present the other way. A planted
  check (a 10% replacement of HCI's late abstracts by computer science's or psychology's) guards
  W2 to W5: an instrument that fails it has its predictions reported as untestable. The source
  states this is the last round on HCI's direction.
- **Method and settings:** Round N's samples (seeds 1 to 5, 2,000 abstracts per field, 1,600
  train and 400 held out, 1990s and 2020s). The classifier is round N's 1990s classifier. The
  language models are one Kneser-Ney bigram model per field on the 1990s training abstracts,
  equal tokens. The contrast of field A is A's mean cost under the behavioural and social models
  minus its mean cost under the computing models (each without A); relative movement subtracts
  the median of the ten other fields. HCI_v is round K′'s nine-venue cell. The planted check
  halves the 1990s held-out abstracts. Synthetic calibration beforehand (calibration scripts not
  committed).
- **Result:** 1 passed, 4 untestable. HCI_v held 1,421 abstracts in the 1990s and 828 in the
  2020s. Planted check, classifier: failed (computer science planted +0.066, sd 0.183, absent;
  psychology planted +0.566, sd 0.160, present; unplanted halves +0.173, sd 0.148, absent).
  Planted check, language models: failed (+0.013, sd 0.067, absent; +0.214, sd 0.060, present;
  unplanted +0.066, sd 0.062, absent). W1 passed. W2 untestable: whole cell towards computing
  −0.867 (sd 0.139), every seed negative. W3 untestable: +0.343 (sd 0.035). W4 untestable:
  HCI_v towards computing −1.059 (sd 0.074). W5 untestable: +0.383 (sd 0.012). Exploratory,
  after scoring, each planted run minus the same seed's unplanted movement: classifier +0.239 (sd
  0.071) and +0.393 (sd 0.034); language models +0.079 (sd 0.021) and +0.148 (sd 0.008); all four
  present. Had the check passed, W3 and W5 would have passed and W2 and W4 would have been
  falsified.
- **Downsides:** The planted check failed on its design, not the instruments: it tested each
  planted run against zero rather than against the same seed's unplanted halves, and it used
  half-size samples; the unplanted halves moved apart by chance far more than on synthetic fields
  (seed sd 0.15 and 0.06, against 0.04 and 0.01). Language models are the weaker instrument in
  the calibration. The README's statement that both instruments point towards the behavioural and
  social sciences in this frame is exploratory only. The older cleaning applies, with the
  lemmatised path's extra flaw. HCI_v is a small cell, and its venue shares changed (HFES
  proceedings fell from 30% to 4% of its documents).
- **Status:** **untestable**, because the check guarding W2 to W5 failed, so those predictions
  could not be scored; W1 passed.
- **Retested by the rebuild:** no (whole-field comparison left the project).
- **Source and commits:** `docs/predictions_cross_perplexity.md`, "Round W", "Round W results"
  and "Exploratory, after scoring: the planted check by seed"; predictions 4bc6af5 (2026-11-10);
  results 687c88c.

## Contextual usage pairs

Usages encoded with XL-LEXEME and compared pair by pair (APD, PRT, J). The design stopped at its
second rung (round X). Source: `docs/predictions_pairs.md`.

### Round U (usage pairs): does the pipeline reproduce a published benchmark result? (2026-11-10)

- **Line:** contextual usage pairs
- **Question and prediction(s):** Does the XL-LEXEME pipeline reproduce Periti and Tahmasebi's
  (2024) result on DWUG EN, and does the across-minus-within measure J track change? U1, APD's
  agreement (Spearman) with the graded change score is at least 0.85 (published 0.886). U2, APD
  beats PRT. U3, J's agreement is at least 0.75. U4, with the periods shuffled J's mean agreement
  is under 0.2 in absolute value while APD's stays at 0.4 or more. Reading rule: if U1 fails,
  rung 2 does not start; if it passes, J is the primary measure if its agreement is at least
  APD's minus 0.05 and U4's J part passes, otherwise APD.
- **Method and settings:** DWUG EN 2.0.1 (Zenodo 7387261, md5 80c3f00f3d13e5ad396e8464ebe67bbc):
  46 words, 9,107 usages, two periods (1810–1860, 1960–2010), gold `change_graded`. Encoder
  XL-LEXEME, `pierluigic/xl-lexeme` at revision c9f85c6f, through
  `interpretant.drift.pairs.XLLexeme`: target wrapped in `<t>` and `</t>`, centred, cut to 128
  tokens, masked mean pooling. Measures per word with cosine distance: APD, PRT (distance between
  period mean vectors) and J (APD minus the mean of the two within-period APDs). Spearman over 46
  words; shuffle control of 20 reassignments of usages to periods (seed 2026). The run also
  fetched a safetensors copy of the weights from an open pull request; all 392 tensors were
  checked identical to the pinned ones.
- **Result:** 3 passed, 1 falsified. U1 passed: APD 0.864 (published 0.886). U2 passed: APD
  0.864, PRT 0.650 (published 0.886 and 0.676). U3 falsified: J 0.539 (needed 0.75). U4 passed:
  with the periods shuffled J's mean agreement −0.124 (range −0.384 to +0.350) and APD's +0.842
  (range +0.838 to +0.846). By the reading rule rung 2's primary measure is APD. The exploratory
  note: APD keeps 0.842 of 0.864 with no change possible, so a reproduction of 0.864 validates
  the pipeline, not APD as a measure of change; APD mostly ranks words by how many senses they
  have.
- **Downsides:** One benchmark of 46 words. APD mostly measures a word's variety of uses rather
  than change between periods. J nets variety out but agrees at only 0.539. DWUG is historical
  fiction and news, not research abstracts (see check 2b).
- **Status:** **holds**, because nothing later contradicts it; the decision agenda reuses its
  numbers to justify choosing J.
- **Retested by the rebuild:** detection experiment 0a (calibration on DWUG EN: encoder and
  measures against human graded change; step 6.3 repeats the APD 0.864 / 0.842 and J 0.54 / −0.12
  figures); step 1b replaces `score_pairs_benchmark`.
- **Source and commits:** `docs/predictions_pairs.md`, "Round U" and "Round U results"; predictions
  e6cd9cf (2026-11-10); results 8691918.

### Round X: does the pipeline find this project's known changes over time? (2026-11-10)

- **Line:** contextual usage pairs
- **Question and prediction(s):** Does the frozen XL-LEXEME configuration find the documented
  changes and stay quiet on the quiet transitions, with APD and J scored side by side? X1, APD
  finds at least three of *attention*, *prompt*, *platform*, *network*. X2, APD finds neither
  quiet transition. X3 and X4, the same for J. X5, under the shuffle bar alone J finds a change
  in more than 30% of the control words' transitions. Reading rule: if neither measure's pair
  (X1 and X2, X3 and X4) passes, the design stops at this rung.
- **Method and settings:** Round U's configuration frozen. Usages from the cased abstracts of
  each cell with round C′'s quality rules, abstracts in both cells of a transition removed,
  matching the word with its regular plural, one occurrence per abstract, at most 150 abstracts
  per cell, seed 1, the sentence cut to 100 words. Documented: *attention* in AI 2010s → 2020s
  (328 and 899 abstracts available); *prompt* in AI 2000s → 2020s, word-targeted cells (298 and
  756); *platform* in communication 2000s → 2010s (142 and 560); *network* in AI 1980s → 1990s
  (1,230 and 2,513); *alignment* in AI 2010s → 2020s (105 and 320, weak case, counts towards
  nothing). Quiet: *attention* in AI 1990s → 2000s (186 and 174) and in psychology 2010s → 2020s
  (623 and 658). Controls: round C's 30 random control words. A change is *found* when the
  measure exceeds the 95th percentile of 200 shuffles of the word's usages and the 90th
  percentile of the transition's control words. Words under 100 usages in a cell are not
  measured.
- **Result:** 3 passed, 2 falsified. X1 falsified: APD found 1 of 4 (*prompt*). X2 passed: APD
  found neither quiet transition. X3 passed: J found 4 of 4 (*attention* 0.0355, *prompt* 0.0101,
  *platform* 0.0107, *network* 0.0045, each above its bars) and also the weak case *alignment*
  (0.0261). X4 falsified: J found the quiet *attention* in AI 1990s → 2000s (0.0019 against bars
  0.0014 and 0.0015). X5 passed: J above its shuffles' 95th percentile in 56 of 175 control
  transitions (32%). By the reading rule neither pair passed, so the design stops at this rung.
- **Downsides:** Matching by `\bword(s?)\b` (the word and its regular plural, no irregular
  plurals or part of speech), with one occurrence per abstract (the first mention, per the
  pipeline map). Small cells: the false alarm came on the smallest cells in the round (about 180
  usages each) and cleared the controls' bar by 0.0004; the source says resampling would test
  whether such margins are noise and the round did not do it. Four documented changes only. The
  diagnostics later showed J follows a word's context within a sense (check 2b) and that a third
  of control words' contexts drift between decades (X5), so J answers "did the contexts change?"
  more than "did the meaning change?".
- **Status:** **holds**, because the scored verdicts are untouched; the later diagnostics changed
  how J is read, not what was found.
- **Retested by the rebuild:** detection experiments 1 and 2 (J judged against matched controls
  and shuffles) with the documented changes and quiet transitions as the checks every experiment
  reads (step 6); step 1b replaces `score_pairs_known` as the store's re-baseline.
- **Source and commits:** `docs/predictions_pairs.md`, "Round X" and "Round X results";
  predictions 3fd5c97 (2026-11-10); results d001878.

## Sense labels

Each usage gets one sense from a fixed per-word list; a cell's meaning is its share of each
sense. Two language-model labellers (Claude Sonnet 5.5, low effort, Message Batches API; JEV at
`jev-1.14.0`) and, from round E, a cue-seeded classifier on XL-LEXEME vectors. The instrument
stopped at rung 3. Source: `docs/predictions_labels.md`.

### Round Y: do the labels rank a human benchmark's changes? (2026-11-10)

- **Line:** sense labels
- **Question and prediction(s):** Do sense labels stand in for DWUG's human clusters? Scored for
  each labeller. Y1, Spearman between the labels' divergence and DWUG's graded change over the 46
  words is at least 0.70 (BERT 0.563, XL-LEXEME 0.864). Y2, mean adjusted Rand index (ARI) with
  DWUG's clusters at least 0.273 (XL-LEXEME's published figure). Y3, with the periods shuffled
  the mean Spearman over 20 shuffles is at most 0.20. Y4, at least 85% of repeated usages get the
  same label. Y5, the two labellers agree on at least 80% of usages (descriptive). Reading rule:
  Y1, Y3 and Y4 passing means rung 2; any failing means the labeller stops at rung 1.
- **Method and settings:** DWUG EN 2.0.1 minus the 320 usages its annotators set aside (-1): 8,787
  usages, each cut to 100 words with the target marked. Sense lists `data/sense_lists/dwug_en.toml`
  (199b273): 2 to 4 senses per word, written by Claude from general knowledge and not reviewed
  by the author. Sonnet with a JSON schema; JEV with a Choice and a companion yes-or-no question
  (top choice above 0.7, *other* below 0.3, *unclear* between). 10% of usages (seed 2026)
  labelled a second time. Shuffles seeds 0 to 19. Disclosure: Claude knew the SemEval-2020
  results in general.
- **Result:** JEV (9,666 requests, none failed; *other* 1.5%, *unclear* 7.1%): Y1 passed 0.821;
  Y2 falsified, ARI 0.267 over 42 words; Y3 falsified, shuffled Spearman 0.363 (range 0.160 to
  0.632); Y4 passed, 96.8% of 879 repeats. Sonnet (*other* 8.4%, *unclear* 0.5%): Y1 passed
  0.767; Y2 passed, ARI 0.287; Y3 falsified, 0.265 (range −0.079 to 0.489); Y4 passed, 98.9%; Y5
  passed, 84.5% of 8,787 usages. By the reading rule both labellers stopped at rung 1 on Y3.
  Correction (2026-11-10): applied to DWUG's own clusters, the Y3 procedure keeps a mean Spearman
  of 0.855 (range 0.812 to 0.885), so the gold standard fails Y3 by more than JEV does.
- **Downsides:** Y3's bar was not checked against the gold before it was committed and cannot
  separate change from variety (it penalises accuracy). The lists were written by Claude, who
  knew the benchmark's results in general, and round Z showed the result depends on the lists.
  Label noise: *other* and *unclear* shares differ between labellers (8.4% against 1.5% *other*).
  Lists cap a word at four senses while DWUG's clusters are uncapped (the source calls Y2 the
  harder test).
- **Status:** **holds**, except Y3, which its own correction withdrew as a test (it fails the
  gold standard); Sonnet's and JEV's stop at rung 1 rested only on Y3.
- **Retested by the rebuild:** step 7a.3 (hand-written lists scored by agreement with DWUG's human
  clusters, "Round Y's bar is 0.273") and step 1b (`score_labels_benchmark`, the human benchmark,
  replaced during the rebuild).
- **Source and commits:** `docs/predictions_labels.md`, "Round Y", "Round Y results for JEV",
  "Correction: Y3's bar fails the gold standard" and "Round Y results for Sonnet, and Y5";
  predictions 4605222 (2026-11-10); results for JEV and the correction 1636897, for Sonnet 6dee4f8.

### Round Z: does round Y's result survive sense lists from WordNet? (2026-11-10)

- **Line:** sense labels
- **Question and prediction(s):** Does round Y's result depend on how its lists were drawn? For
  each labeller: Z1, Spearman with DWUG's graded change at least 0.70. Z2, within 0.10 of the
  labeller's round Y Spearman. Z3, at least 85% of repeated usages get the same label. Z4, ARI
  beside round Y's (descriptive). Reading rule: if Z1 or Z2 fails, every citation of that
  labeller's round Y result says it depends on the sense lists.
- **Method and settings:** As round Y (the same 8,787 usages, labellers, repeats, divergence and
  ARI), with lists that no one wrote: the first four WordNet 3.0 senses of the word's part of
  speech that contain the word, in WordNet's order, with WordNet's definition and example,
  instance synsets left out (30c4bc4). Five words get a single sense (*chairman, chef, lass,
  maxim, quilt*). No variety check.
- **Result:** JEV: Z1 falsified, Spearman 0.635; Z2 falsified, round Z 0.635 against round Y
  0.821 (difference −0.187); Z3 passed, 96.6%; Z4 ARI 0.220 against 0.267. Sonnet: Z1 falsified,
  0.588; Z2 falsified, 0.588 against 0.767 (difference −0.180); Z3 passed, 98.1%; Z4 ARI 0.235
  against 0.287. Exploratory: with WordNet's lists JEV found no fitting sense for 19.2% and 19.4%
  of the two periods' usages (8.3% and 8.8% with round Y's lists); Sonnet 19.4% and 17.4% (8.8%
  and 9.0%).
- **Downsides:** The cap can drop a sense that mattered in the 19th century (*plane* keeps the
  aeroplane and loses the plane tree), and WordNet's order follows modern frequency. The source
  cannot separate two readings: round Y's lists were better because their writer knew which words
  changed, or because they were written for these words.
- **Status:** **holds**, because nothing later contradicts it; the decision agenda cites the
  result (WordNet about 0.18 below hand-written lists).
- **Retested by the rebuild:** no (step 7a records the result and uses Wiktionary only as a gap
  check on hand-written lists).
- **Source and commits:** `docs/predictions_labels.md`, "Round Z", "Round Z results for JEV" and
  "Round Z results for Sonnet"; predictions 80718f3 (2026-11-10); results JEV 4b34ca0, Sonnet
  a43222c.

### Round Q: can sense lists induced from the texts match hand-written ones? (2026-11-10)

- **Line:** sense labels
- **Question and prediction(s):** Do lists built from half of each word's usages, by methods that
  are not generative models and a model that may only edit what the evidence shows, reach round
  Y's level on the other half? Q1, Spearman over the 46 words on half B at least 0.70 (all three
  labellers). Q2, at least the labeller's round Y Spearman on the same half minus 0.10 (Sonnet,
  JEV). Q3, partial Spearman (controlling for the mean shuffled divergence) at least 0.40. Q4, at
  most 10% *other* or *unclear*. Q5, at least 85% of repeats the same. Q6 and Q7 descriptive.
  Amendment (1b49af9, before step B): rule-built lists with predictions R1 to R7 (R1 at least
  0.70, R2 at least Q1's minus 0.10, R3 at least 0.40, R4 at most 10%, R5 at least 85%).
- **Method and settings:** DWUG's kept usages split into half A (lists induced) and half B (4,377
  usages, all measures); seed 2026. Step A: ModernBERT-large (`answerdotai/ModernBERT-large`,
  revision 45bb4654) substitutes with the "(or even [MASK])" pattern and Louvain communities
  (resolution 1, seed 2026), the 15 most frequent neighbouring words, WordNet 3.0 and Wiktionary
  (kaikki.org) senses. Step B: Claude Opus 5.5 at high effort, Batch API, 1 to 8 senses per word,
  with one coverage-pass revision; lists frozen before half B was labelled. Labellers: Sonnet,
  JEV and a substitute labeller with no generative model. Calibration of the variety check on
  DWUG's clusters first (Spearman 0.966, partial 0.849).
- **Result:** Opus's lists held 1 to 5 senses at first; 22 words were over 10% *other* or *unclear*
  on half A; the revision left 14 words at the 8-sense cap. Induced lists (Sonnet; JEV; substitute
  labeller): Q1 0.443, 0.502, 0.433, all falsified. Q2 as the table gives it: 0.693 and 0.800 (round
  Y's half-B figures), both falsified. Q3 0.307, 0.362, 0.183, all falsified. Q4 6.0%, 6.8%, 1.4%,
  passed. Q5 98.4% and 98.6%, passed. Q6 ARI 0.252, 0.245, 0.168 (round Y on half B 0.330 and
  0.302). Rule-built lists: R1 0.385, 0.590 (45 words), 0.420, all falsified; R2 passed for all
  three; R3 0.272, 0.453, 0.207 (JEV passed); R4 11.7%, 16.2%, 1.4% (Sonnet and JEV falsified); R5
  96.8% and 95.7%, passed. No labeller passed rung 1 with either kind of list; induced lists are
  described as weaker than hand-written ones, by 0.25 for Sonnet and 0.30 for JEV on the same half.
  The language model's part in building the lists changes the result little (R2).
- **Downsides:** Scorer correction (578c4a2): JEV left every half-B usage of *afternoon* in one
  period *unclear* with the rule-built lists, so its correlations are over 45 words. The
  substitute communities split usages by topic and grammatical number (*attack*'s military and
  violent-crime groups, *plane*'s singular and plural vehicles), the context-for-meaning
  confusion again. Opus and Sonnet are both Claude models. The round cannot say whether
  hand-written lists win because their writer knew the answers or because they draw the lines a
  reader draws.
- **Status:** **holds**, because nothing later contradicts it; the decision agenda cites 0.39–0.59
  against 0.69–0.80.
- **Retested by the rebuild:** step 7a.2, option (3): substitutes grouped by community detection
  (Leiden) as an aid to the hand-written lists (7a.1), not as the list; step 7a records the
  result.
- **Source and commits:** `docs/predictions_labels.md`, "Round Q", "Round Q amendment: a rule-built
  arm", "Round Q scorer correction" and "Round Q results"; predictions f746233 (2026-11-10);
  amendment 1b49af9; scorer correction 578c4a2; results 357d974.

### Round E: can a cue-seeded classifier on XL-LEXEME rank the benchmark's changes? (2026-11-10)

- **Line:** sense labels
- **Question and prediction(s):** Can a logistic regression on XL-LEXEME vectors, trained from cue
  words and no generative model, match a language-model labeller on half B? E1, Spearman on half
  B at least 0.70. E2, at least Sonnet's 0.693 minus 0.10 (0.593). E3, partial Spearman at least
  0.40. E4 to E6 descriptive. Reading rule: the classifier passes rung 1 if E3 passes and E1 or
  E2 does.
- **Method and settings:** Round Y's lists (149 senses, 46 words) with cue words
  (`data/sense_lists/dwug_en_cues.toml`) written before any DWUG usage was read for them; round
  Q's step A evidence for *plane* and *attack* had been seen. Seeds from half A (a cue of one
  sense within five words either side) plus each sense's invented example. XL-LEXEME as in round
  U; per word a logistic regression (C = 1, balanced class weights, 1,000 iterations) that never
  says *other* or *unclear*. Round Q's halves.
- **Result:** E1 falsified: 0.669 on half B. E2 passed (0.669 against 0.593). E3 passed: partial
  0.509 (gold 0.849). E4 descriptive: same label as Sonnet on 75.2% of half B, as JEV on 76.0%.
  E5: 1,109 cue seeds over 149 senses, 62% with three or more, median 5. E6: examples alone
  0.628. Mean ARI 0.266. By the reading rule the classifier passes rung 1. For context, Sonnet
  with the same lists on half B had Spearman 0.693, partial 0.646, ARI 0.330; JEV 0.800, 0.667,
  0.302.
- **Downsides:** Label noise and cue-labelled error: seeds come from cue words, and the classifier
  tracks change beyond variety less well than either language model (0.509 against 0.646 and
  0.667). Cue words were written knowing round Q's evidence for two words. Round Y's lists,
  written by Claude, stood in for approved ones. 148 of the 149 senses' examples contain a form
  of the lemma; one sense trained on cue seeds alone.
- **Status:** **holds**, because nothing later contradicts it; the classifier it passed remained
  the labeller through rung 3.
- **Retested by the rebuild:** step 7.3, counters (1) the cue classifier, the current baseline,
  and (1b) a supervised classifier on XL-LEXEME vectors; step 1b says
  `score_labels_classifier` is superseded by step 7.
- **Source and commits:** `docs/predictions_labels.md`, "Round E" and "Round E results";
  predictions 9c3ea97 (2026-11-10); results fa09310.

### Round T: does the classifier read this corpus as the author does? (2026-11-10)

- **Line:** sense labels
- **Question and prediction(s):** On this corpus, does the classifier agree with the author's
  pair judgements (check 2b)? T1, the overlap of the two usages' sense probabilities correlates
  with the author's judgement at least as well as XL-LEXEME's raw similarity (0.456). T2, the
  classifier's "same sense" Spearman is at least Sonnet's minus 0.10. T3 descriptive (agreement
  with Sonnet and JEV). Reading rule: the classifier passes rung 2 if T1 and T2 pass.
- **Method and settings:** Classifier of round E, with lists and cue words for *attention, prompt,
  platform, network, component, having* that Claude drafted and the author refined on a review
  page (frozen at 9332d86). Trained on every usage of the six words in the whole-corpus sample
  (at most 150 abstracts per cell, seed 1) whose cues name one sense, plus each sense's example,
  with check 2b's 186 usages left out. Compared with Sonnet and JEV on the same 186 usages and
  96 judged pairs. Disclosure: the labeller pilot showed Claude these 186 usages and three
  drafting choices came from it, so the round is partly in-sample for the lists.
- **Result:** T1 falsified: probability overlap 0.449 (needed 0.456). T2 falsified: same sense
  0.397 (needed 0.414, Sonnet's 0.514 minus 0.10). T3: same sense as Sonnet on 76.3% of the 186
  usages, as JEV on 76.3%. With the correction pass: classifier same sense 0.441, overlap 0.522
  (XL-LEXEME 0.494); Sonnet 0.563; JEV 0.504. Per word, same-sense Spearman for classifier,
  Sonnet, JEV: *attention* 0.87, 0.77, 0.77; *having* 0.54, 0.71, 0.59; *prompt* 0.46, 0.75,
  0.87; *platform* 0.27, 0.41, 0.40; *network* 0.27, 0.25, −0.05; *component* 0.13, 0.35, 0.35.
  By the reading rule the classifier does not pass rung 2.
- **Downsides:** Small samples: 96 judged pairs, 16 per word, so the per-word figures are noisy.
  The author's judgements come from one person and were corrected for five *network* pairs after
  discussion. Matching by `\bword(s?)\b` and the first mention per abstract apply to the training
  sample. Cue-labelled training sentences: the classifier trained on 273 *prompt* sentences and
  fell furthest behind there. Partly in-sample for the lists (disclosed).
- **Status:** **holds**, because nothing later contradicts it; the head-to-head round gave the
  classifier a second attempt on fresh sentences without undoing this verdict.
- **Retested by the rebuild:** steps 7.2 and 7.3 (the author's own random, blind labels replace
  pair judgements; counters are scored against them); step 1b says `score_labels_corpus` is
  superseded by step 7.
- **Source and commits:** `docs/predictions_labels.md`, "Round T" and "Round T results";
  predictions 53157be (2026-11-10), amended in 4c3217f and 8675733 before the run; results
  7a35020 (2026-11-11).

### Round U (labels, head-to-head): is the revised classifier as good as Sonnet? (2026-11-11)

- **Line:** sense labels
- **Question and prediction(s):** Where a revised classifier and Sonnet disagree on fresh
  sentences, does the author, blind to which is which, pick the classifier at least 40% of the
  time? U1, of the disagreements where the author chooses exactly one label, the classifier's is
  chosen in at least 40% (untestable if fewer than 20 are decided). U2 to U4 descriptive
  (agreement, rough accuracy, per word). A one-off second attempt: the plan's rule of no repair
  rounds was amended for this case. Reading rule: if U1 passes the classifier labels rung 3.
- **Method and settings:** `corpus_v2.toml` (0b3bfa3): cue words added to six senses. Classifier:
  XL-LEXEME, one logistic regression per word, C = 1, square-root balanced class weights, and
  cue-first labelling. 240 distinct usages, 40 per word (seed `round-u`), left out of training,
  a development set of 300 random usages labelled blind by Claude beforehand. Sonnet (Batch API,
  low effort) with the same lists. The author judged each disagreement on a page in random order
  without being told which labeller gave which.
- **Result:** U1 passed: the classifier's label chosen in 18 of 41 decided disagreements (43.9%;
  needed 40%); both fit 3, neither 4. The two labellers disagreed on 48 of 240 usages. U2:
  agreement on 192 of 240 (80.0%). U3: rough accuracy, classifier 88.8%, Sonnet 90.8%. U4:
  classifier chosen per word *attention* 2/7, *prompt* 1/8, *platform* 2/6, *network* 6/11,
  *component* 3/3, *having* 4/6. By the reading rule the revised classifier labels rung 3.
- **Downsides:** The pass is narrow: 41 decided disagreements give an interval of roughly 30% to
  60% around 43.9%. Cue additions and weights were chosen knowing round T's misses, and the
  development set samples the same training sample the test sentences come from (some may have
  been shown before). The classifier loses on *prompt*, where Sonnet was chosen in 7 of 8.
  Process departures from the pre-registration: the review page was a local file, gained each
  sentence's field and decade at the author's request, and a UTF-8 declaration; none touches the
  labels or the key. The accuracy estimate counts agreements as right (an upper estimate).
  Cue-labelled error rates (disclosed risk).
- **Status:** **holds**, because nothing later contradicts it.
- **Retested by the rebuild:** steps 7.2 and 7.3 (counters compared with Sonnet as a comparison
  arm; step 7.5 keeps one Claude labelling path); step 1b says `score_labels_head_to_head` is
  superseded by step 7.
- **Source and commits:** `docs/predictions_labels.md`, "Round U: the revised classifier against
  Sonnet, judged blind by the author" and "Round U results" (the note at the end of the file
  calls it the head-to-head round); predictions 8b5d2a8 (2026-11-11); results f4177d0.

### Round A: does the classifier find known changes, a planted change and a placebo? (2026-11-10)

- **Line:** sense labels
- **Question and prediction(s):** Rung 3: does the classifier find round X's documented changes
  and stay quiet on the quiet transitions, with a planted change found and a placebo not? A1,
  at least three of *attention, prompt, platform, network* found. A2, neither quiet transition
  found. A3, in each documented change the documented sense's adjusted share rises by more than
  0.05. A4, a planted mechanism found in at least four of five draws at some share of 30% or less
  and at every larger share. A5, the *years* placebo found in at most one of five draws. A6 to A8
  descriptive. Reading rule: the instrument passes rung 3 if A1 to A5 pass; otherwise it stops
  before open questions.
- **Method and settings:** Round X's transitions and cells with seed 2, at most 150 abstracts per
  cell. Lists `corpus_v2.toml` for targets (amendment 46d7fc5 after round U); ten control words
  (`data/control_words_rung3.txt`). Classifier: XL-LEXEME, one logistic regression per word, C = 1,
  square-root balanced weights, cue-first labelling, trained on the whole-corpus sample's
  cue-labelled sentences. Adjusted classify and count with the confusion matrix from five-fold
  out-of-fold predictions on the cue-labelled training sentences. A change is the Jensen-Shannon
  divergence between two cells' adjusted shares, found when above the 95th percentile of 200
  shuffles and the 90th percentile of the controls' divergence. Planted: psychology's *attention*
  2010s → 2020s with 10% to 60% replaced by AI 2020s usages with attention-mechanism markers.
  Placebo: 40% of psychology's 2020s *years* replaced by sociology's. Amendments before the run:
  text tidying and boilerplate (4c3217f), the merged *alignment* senses (34731a9), exclusion cues
  and out-of-scope usages (8675733) and four named risks.
- **Result:** 20,356 usages measured. A1 falsified: 2 of 4 found: *attention* (0.108 bits;
  shuffles 0.025, controls 0.054) and *platform* (0.024; 0.024, 0.008); *prompt* (0.097; 0.036,
  0.107) and *network* (0.048; 0.027, 0.059) beat their shuffles but not their controls. A2
  passed: *attention* in AI 1990s → 2000s 0.018 bits, in psychology 0.001. A3 passed: attention
  mechanism 0.175 → 0.518, language-model input 0.456 → 0.802, digital platform 0.673 → 0.783,
  artificial neural network 0.610 → 0.811. A4 passed: found in 3 of 5 draws at 10% and 5 of 5 at
  20% to 60%. A5 passed: 0 of 5 draws. A6: 12 of 60 control-word transitions (20%) beat their
  shuffles. A7: *alignment* matching 1.000 → 0.583, goal or value alignment 0.000 → 0.417, found
  under both bars. A8: Sonnet and JEV, raw shares under bar (a), each find all four documented
  changes and neither quiet transition. The instrument does not pass rung 3.
- **Downsides:** Matching by `\bword(s?)\b` and the first mention per abstract. Cue-labelled error
  rates: adjusted shares use a confusion matrix from the easier, cue-labelled sentences, and
  thin senses' levels are likely overstated. The first named risk happened: the classifier
  labelled 66 of AI's 150 2000s *prompt* usages language-model input (63 by the regression),
  mostly spoken-dialogue system prompts and the ontology tool PROMPT, where Sonnet labelled 5, so
  the earlier share is mostly this error and the change falls below the controls' bar. The
  controls' bar was highest on the only two-decade transition (0.107). The *network* miss may be a
  change inside the 1980s that decade cells blur (exploratory, untested). Sonnet and JEV did not
  label the control words, so bar (b) was not applied to them. Shuffle-only comparison: under bar
  (a) alone the classifier finds all four.
- **Status:** **holds**, because nothing later contradicts it; the README reports that the
  instrument stopped at its third rung.
- **Retested by the rebuild:** step 7.3 (hard tests: anachronism for *prompt* and *hallucination*,
  stability for *component* and *having*), step 7.4 (proportions), and the words of step 7.1
  (*attention, network, platform, alignment, prompt, hallucination*); step 1b says
  `score_labels_known` is superseded by step 7.
- **Source and commits:** `docs/predictions_labels.md`, "Round A" with its amendments and "Round A
  results"; predictions 8f233d1 (2026-11-10); results 732e1dd (2026-11-11).

### Round A′: a fresh test of the improved labeller, its test set fixed (2026-11-11)

- **Line:** sense labels
- **Question and prediction(s):** The source records no predictions. It fixes only the test set
  for a fresh round, before any of its cells was fetched: documented changes *transformer* and
  *embedding* in AI, *cloud* in computer science, *hallucination* in computer science (windows
  either side of the documented dates, for example *transformer* 2010–2016 → 2019–2023);
  quiet transitions of *hallucination* in psychology (2010–2014 → 2015–2019) and neuroscience
  (2000–2004 → 2005–2009); a planted change (psychology's 2015–2019 *hallucination* usages with
  10% to 60% replaced by computer science 2021–2025 usages naming a language model); a placebo
  (40% replaced by neuroscience's 2015–2019 usages).
- **Method and settings:** Ten control words of rung 3, at most 150 abstracts per window, seed 3,
  quality rules version 2, bars as round A. *transformer*, *embedding* and *cloud* from
  word-targeted downloads (up to 1,000 works per decade); *hallucination* from the word cells
  already held. Dry-run counts of works held: *transformer* 104 → 10,218; *embedding* 2,471 →
  9,669; *cloud* 940 → 38,170; *hallucination* 113 → 3,979. The source says the predictions and
  scorer would be committed later, before a run.
- **Result:** None recorded. The file has no predictions section and no results section for round
  A′, `scripts/` holds no round A′ scorer, and the README counts seven sense-label rounds, which
  does not include it. ROADMAP item 8f calls DeBERTa-v3-large "round A′'s robustness check, with
  one fixed reading rule". The sources do not say whether the round was run.
- **Downsides:** Not applicable (no result). Source-stated limits: stemming inflates the dry-run
  counts (*embedding* also matches *embedded*); the earlier windows of *transformer* and
  *hallucination* are 7 and 10 years long because OpenAlex holds about 100 works in five.
- **Status:** **superseded**, never run: only its test set was fixed, with no predictions, results
  or scorer; the rebuild's held-out set (decision agenda 6.2) and the second wave of words (7.1)
  take over its job.
- **Retested by the rebuild:** step 7.1 (*hallucination* in the first set, *prompt* for the
  anachronism test; *transformer*, *embedding* and *cloud* in the second wave) and the held-out
  set of step 6.2.
- **Source and commits:** `docs/predictions_labels.md`, "Round A′: the test set, fixed before its
  data"; test set committed in e564693 (2026-11-11); no results commit exists. Related commits for
  the lists: 7ef8f8b (word-targeted cells), 7c7dee5 and 39e6792 (sense lists).

## Sense clustering

Contextual embeddings of every occurrence (SciBERT, ModernBERT) clustered into senses shared by
every cell, calibrated on known changes before any open question. Stopped after round C″ by a
rule committed with it; round D was not run. Source: `docs/predictions_senses.md`. In all three
rounds the word is matched by `\bword(s?)\b` (the word and its regular plural, any case, also
inside hyphenated compounds), and the cells are drawn from the abstracts containing it.

### Round C: does the clustering instrument find senses whose history is known? (2026-11-10)

- **Line:** sense clustering
- **Question and prediction(s):** Does the instrument find the attention-mechanism sense of
  *attention* in AI and AI's sense of *intelligence* spreading? Scored per encoder. C1, in every
  seed the mechanism sense holds at least 80% of the marked uses among AI's 2020s occurrences, at
  most 5% of AI's 1990s occurrences and at least 30% of AI's 2020s occurrences. C2, the mechanism
  sense holds under 5% of psychology's occurrences in both decades in every seed and
  psychology's change in *attention* is no larger than the median of 30 control words'. C3, AI's
  change in *attention* is larger than that of at least 27 of the 30 controls. C4, the AI sense
  of *intelligence* rises from the 2000s to the 2020s in communication, history and philosophy of
  science, sociology and philosophy. C5, it does not rise in cognitive psychology. Reading rule:
  if neither encoder passes C1, the instrument is not used on open questions and a redesign is
  calibrated first.
- **Method and settings:** Cased abstracts behind each cleaned cell, abstracts in more than one
  field's cell of a decade removed; per word, cell and seed (1 to 5) up to 500 abstracts in
  seeded random order, the first occurrence in each, the sentence cut to 100 words. SciBERT
  (`allenai/scibert_scivocab_uncased`) and ModernBERT (`answerdotai/ModernBERT-base`), no revision
  given; vector is the mean of the last four hidden layers. k-means on length-normalised vectors
  with k from 2 to 10, best cosine silhouette (a seeded sample of at most 5,000). Change is the
  Jensen-Shannon divergence of sense distributions in bits. 30 control words drawn at random (seed
  2026, `data/control_words_senses.txt`). Marked uses: contexts with *self-attention, transformer,
  multi-head, attention mechanism* and similar. "Present" means the same sign in every seed and a
  mean at least twice the standard deviation. The first run stopped on a mostly Chinese abstract
  past the encoder's 512 tokens; the fix (39c7f97) re-encodes from 200 characters either side and
  the predictions were not changed.
- **Result:** For each encoder 2 passed, 3 falsified. SciBERT: C1 falsified (144–169 marked uses
  per seed, 0.99 in one sense; share of AI's occurrences 0.147 in the 1990s, 0.580–0.622 in the
  2020s); C2 falsified (the mechanism sense holds 0.366 of psychology's 1990s and 0.374–0.402 of
  its 2020s occurrences; change 0.0005 against the controls' median 0.0147); C3 passed (0.164,
  above all 30 controls, the largest *report* 0.083); C4 passed (rises of 0.554, 0.385, 0.535,
  0.376); C5 falsified (+0.292, sd 0.000). ModernBERT: C1 falsified (share 0.136–0.141, then
  0.564–0.614); C2 falsified (0.327–0.329 and 0.364–0.392; change 0.0018 against 0.0114); C3
  passed (0.165, the largest control *increasing* 0.066); C4 passed (0.505, 0.340, 0.546, 0.395);
  C5 falsified (+0.281, sd 0.003). By the reading rule the instrument is not used on open
  questions.
- **Downsides:** Matching by `\bword(s?)\b` and the first occurrence in each abstract. The
  silhouette favoured the coarsest split: two senses in every *attention* and *intelligence* run,
  and in about two-thirds of all runs (81 of 155 under SciBERT's *attention* design, 110 of 165
  under ModernBERT's). Both found two senses, the stock phrase of academic prose and attention as
  something that operates (mechanism, cognitive, joint attention and ADHD together), so the
  mechanism was not separated from cognitive attention. Small cells: most non-AI *intelligence*
  cells hold fewer than 500 occurrences, so every seed draws all of them and a standard
  deviation near zero makes the presence rule almost automatic. The marked-use list is hand
  chosen. Label noise: OpenAlex field labels are noisy (some cognitive psychology *intelligence*
  uses are mislabelled engineering). C5 was a wrong bet, not an instrument failure (cognitive
  psychology's 2020s abstracts write about artificial intelligence too).
- **Status:** **superseded**, because round C′ redesigned the instrument as the reading rule
  required; its exploratory observations (the two encoders agree within a few hundredths) stand.
- **Retested by the rebuild:** partly. Step 7a.2 tries clustering again in a different
  configuration (XL-LEXEME vectors with affinity propagation, agglomerative clustering or
  HDBSCAN) as an aid to hand-written sense lists; step 1b supersedes the clustering half of
  `drift senses` (SciBERT, k-means, silhouette). The SciBERT design itself is not rerun.
- **Source and commits:** `docs/predictions_senses.md`, "Round C" and "Round C results";
  predictions dfa0837 (2026-11-10); results 3cc3e11.

### Round C′: does a finer instrument find a sense it has not seen? (2026-11-10)

- **Line:** sense clustering
- **Question and prediction(s):** After redesigning on round C's data, does the instrument find
  the language-model sense of *hallucination*? C′1, in every seed the mechanism set holds at most
  5% of AI's 1990s occurrences, at least 30% of AI's 2020s occurrences and under 5% of
  psychology's in each decade. C′2, in every seed the language-model set holds at least 50% of
  AI's 2020s occurrences of *hallucination* and under 5% of psychology's and neuroscience's 2000s
  occurrences. C′3, in psychology and neuroscience the language-model set holds under 10% of the
  2020s occurrences. C′4 (an open question), its share of psychology's occurrences rises from the
  2000s to the 2020s. Reading rule: if C′2 fails the fix was fitted to *attention* and sense-level
  drift is recorded as not achieved with this design.
- **Method and settings:** As round C with: k = 8 for every word (chosen on round C's seed-1
  *attention* data, where the mechanism separates from cognitive attention from 6 to 16 senses);
  SciBERT only; sampling with replacement, up to 500 occurrences per cell and seed; quality rules
  (abstracts over 20% outside Latin script skipped, URLs and DOIs removed); regular plurals;
  nested fields not compared (computer science left out). A meaning is a *sense set*: the fewest
  senses holding 80% of its marked uses among AI's 2020s occurrences. *hallucination* on
  word-targeted cells (AI 2020s; psychology and neuroscience 2000s and 2020s); marked as
  language-model sense by *language model, LLM, ChatGPT, GPT, generative AI, chatbot, factual,
  fabricat*. *attention* had 1,681 occurrences per seed and *hallucination* 2,323.
- **Result:** 2 passed, 2 falsified. C′1 passed: one mechanism sense in every seed (159–183 marked
  uses); share of AI's 1990s occurrences 0.005–0.048, of AI's 2020s 0.550–0.608, of psychology's
  1990s 0.000–0.008 and 2020s 0.006–0.020. C′2 falsified: the language-model set holds
  0.810–0.882 of AI's 2020s occurrences in every seed, but psychology's and neuroscience's 2000s
  shares are 0.115 and 0.058 in seed 1 and 0.211 and 0.150 in seed 5 (0.003–0.028 and
  0.002–0.006 in seeds 2 to 4). C′3 falsified: mean share of psychology's 2020s occurrences
  0.137 (seeds 0.038–0.266), neuroscience's 0.060. C′4 passed: rise of +0.062 (sd 0.020),
  positive in every seed. By the reading rule the fix is taken as fitted to *attention*.
- **Downsides:** Matching by `\bword(s?)\b` and, as in round C, the first occurrence in each
  abstract. The failure was in the rule grouping senses into a meaning: in seeds 1
  and 5 one further sense mixed the language-model meaning with psychology's 2000s usages (15–17%
  of AI's 2020s occurrences, 10–22% of psychology's 2000s) and the 80% coverage rule admitted it.
  The fix was chosen on data already seen. The marked-use list is hand chosen, so the instrument
  is partly supervised without saying so. Label noise: the source mentions label-noise shares
  from `scripts/measure_label_noise.py` but gives no figures.
- **Status:** **superseded**, because round C″ repaired the rule it located; C′1 and C′2's
  verdicts stand as scored.
- **Retested by the rebuild:** partly. As round C: step 7a.2 (clustering as an aid to hand-written
  lists, a different configuration); the SciBERT design is not rerun.
- **Source and commits:** `docs/predictions_senses.md`, "Round C′" and "Round C′ results";
  predictions d83b159 (2026-11-10); results 9ad2378.

### Round C″: does a repaired sense-set rule find a third known sense? (2026-11-10)

- **Line:** sense clustering
- **Question and prediction(s):** After repairing the rule on round C′'s data, does the instrument
  find the language-model sense of *prompt*? C″1, in every seed the meaning set holds at least 50%
  of AI's 2020s occurrences of *prompt*, at most 5% of AI's 2000s occurrences and under 5% of
  psychology's 2000s occurrences. C″2 (an open question), its share of psychology's occurrences
  rises from the 2000s to the 2020s. Stated as the last calibration round: if C″1 fails,
  sense-level drift ends here, every sense result is exploratory only and round D is not run.
- **Method and settings:** Round C′'s method (SciBERT, eight senses, sampling with replacement,
  plurals, quality rules, nested-field rule). The *meaning set* is every sense whose share of
  marked uses, counted over all its occurrences, is at least half that of the sense with the
  highest share. *prompt* on word-targeted cells: AI and psychology, 2000s and 2020s (abstracts
  298 and 756 for AI, 466 and 554 for psychology); 1,764 occurrences per seed. Marked uses:
  *language model, large language, LLM, ChatGPT, GPT, generative AI, chatbot, few-shot,
  zero-shot, in-context, prompt engineering, prompt tuning, prompt learning*.
- **Result:** 0 passed, 2 falsified. C″1 falsified: the meaning set's share of AI's 2020s
  occurrences 0.146, 0.498, 0.092, 0.300, 0.774 (seeds 1 to 5); of AI's 2000s 0.000, 0.191, 0.000,
  0.010, 0.386; of psychology's 2000s 0.000, 0.069, 0.000, 0.009, 0.114; 259–285 marked uses in
  AI's 2020s per seed. C″2 falsified: +0.038 (sd 0.032; seeds +0.026, −0.001, +0.030, +0.049,
  +0.086), absent. By the committed rule sense-level drift ends here.
- **Downsides:** The set was too narrow in seeds 1, 3 and 4 (the top sense was *prompt
  engineering*, a phrase that is itself one of the marks, with marked share 0.89–0.90) and too
  broad in seeds 2 and 5 (senses mixing AI's 2000s dialogue-system prompts with psychology's
  prompts). Each rule that turned clusters into a meaning's share failed on a case it was not
  chosen on; each leans on hand-chosen marks, so the instrument was partly supervised without
  saying so. Matching by `\bword(s?)\b` and the first mention per abstract. Small cells: the
  cells are word-targeted samples of a few hundred abstracts.
- **Status:** **holds**, because nothing later contradicts it; it is the final verdict of the
  line, and the decision agenda records the clustering as "failed calibration", noting each
  method failed in one configuration, not as a method.
- **Retested by the rebuild:** partly. Step 7a.2 (clustering as an aid to hand-written lists,
  different configuration); the SciBERT design is not rerun.
- **Source and commits:** `docs/predictions_senses.md`, "Round C″" and "Round C″ results";
  predictions 3aea6a4 (2026-11-10); results 947212e.

## Diagnostic checks

Checks from the diagnostic plan in `docs/assumptions.md`, run after round X to decide whether the
instrument or the data was at fault. Each check's rules were committed before anything it counts
was looked at. Source: `docs/diagnostics.md`.

### Check 1a: are the known changes present in the cells? (2026-11-10)

- **Line:** diagnostic check
- **Question and prediction(s):** Are the documented changes of rounds 5 and X actually in the
  cells the instruments read (assumption A5)? A documented change is *present* when the marked
  share and the read share (yes over yes plus no) each rise by at least 0.10 from the earlier cell
  to the later; a quiet transition is *quiet* when the marked share changes by less than 0.05 and
  the read share by less than 0.10. The committed consequences: a present change that an
  instrument misses is the instrument's miss; an absent one is not a known answer.
- **Method and settings:** The usages round X's scorer reads (word and regular plural, one
  occurrence per abstract, at most 150 abstracts per cell, seed 1, round C′'s quality rules).
  Seven cases: *attention* in AI 2010s → 2020s, *prompt* in AI 2000s → 2020s (word-targeted
  cells), *platform* in communication 2000s → 2010s, *network* in AI 1980s → 1990s, *alignment*
  in AI 2010s → 2020s, and the quiet *attention* in AI 1990s → 2000s and in psychology 2010s →
  2020s. Marker phrase lists per sense (for example *neural network(s), connectionist,
  backpropagation, perceptron* for *network*). Twenty usages per cell (seed 2026), pooled over
  fourteen cells, shuffled and labelled yes, no or unclear by Claude without the cell, the marker
  match or the case in view (`data/diagnostics/check_1a_labels.tsv`), open to the author's
  correction. The 280 labels were committed before the key was opened (2f4eca4).
- **Result:** All five documented changes present, both quiet transitions quiet. Marked share,
  then read share, earlier → later: *attention* AI 0.13 → 0.31, 0.25 → 0.60; *prompt* 0.01 →
  0.51, 0.00 → 0.95; *platform* 0.08 → 0.47, 0.62 → 0.89; *network* 0.45 → 0.71, 0.65 → 0.95;
  *alignment* 0.01 → 0.13, 0.00 → 0.15; quiet *attention* in AI 0.01 → 0.01, 0.00 → 0.00; in
  psychology 0.00 → 0.00, 0.00 → 0.00. Read, not predicted: about two thirds of the 1980s
  *network* uses and of the 2000s *platform* uses are already the later sense, so decade cells
  catch the end of those changes; only about one in seven of *alignment*'s 2020s uses is about
  aligning AI systems, so the grid's one hit in round 5 is unlikely to be the documented change.
- **Downsides:** Matching by `\bword(s?)\b` (no irregular plurals or part of speech) and the first
  mention per abstract. Small samples: 20 usages per cell, 280 labels. Label noise: the labels are
  Claude's blind readings, not the author's (the author may correct them, and the decision agenda
  has the author refine the 183 "no" and "unclear" items). Marker lists are hand chosen and the
  two shares can disagree. Abstracts with several mentions carry a changing sense more often (57%
  against 26% in the decision agenda's figures), so shares are shares of abstracts.
- **Status:** **holds**, because nothing later contradicts it.
- **Retested by the rebuild:** step 7.2 (the 280 labels are mapped back to works, 279 recovered,
  and the author answers "which sense?" for the 183 "no" and "unclear" items; step 4.3 compares
  first against random mentions); step 1b (the marker words move into the planted-change check
  before `check_known_presence` is deleted).
- **Source and commits:** `docs/diagnostics.md`, "Check 1a" and "Check 1a results"; predictions
  828497b (2026-11-10); results c256821.

### Check 2b: does XL-LEXEME read this corpus's sentences as the author does? (2026-11-10)

- **Line:** diagnostic check
- **Question and prediction(s):** G1, Spearman between XL-LEXEME's similarity and the author's
  judgement over the 96 distinct pairs is at least 0.5 (published agreement on DWUG EN is about
  0.63). The committed consequence: if G1 passes, XL-LEXEME's reading of abstract sentences is not
  the problem; if it fails, the instrument misreads this genre.
- **Method and settings:** From round X's usages, six words: *attention*, *prompt*, *platform*,
  *network* and two random controls, *component* and *having* (AI, 2010s and 2020s). Per word 16
  pairs (5 within the earlier cell, 5 within the later, 6 across; seed 2026); four pairs repeated
  with sentences swapped, 100 pairs in all. The author judged each blind to field, decade and
  cell on the DURel scale (4 identical to 1 unrelated, 0 cannot decide), committed before any
  XL-LEXEME score was computed (f2e4260). XL-LEXEME as frozen in round U; score is the cosine
  similarity of the two usage vectors; pairs judged 0 left out.
- **Result:** G1 falsified: Spearman 0.431 over 96 pairs (needed 0.5). The author's four repeated
  pairs received the same judgement both times. Mean judgement and similarity by kind of pair:
  within earlier cell 2.63 and 0.726, within later 3.07 and 0.772, across 2.67 and 0.737. Per
  word (16 pairs): *attention* 0.70, *prompt* 0.66, *having* 0.54, *platform* 0.43, *network*
  0.10, *component* 0.00. Correction pass (after check 2b′, a sensitivity check, not a result):
  five *network* pairs re-judged from 4 to 2; XL-LEXEME and the author 0.456 → 0.494 over 95
  pairs, *network* alone 0.10 → 0.60; the verdict is unchanged.
- **Downsides:** Matching by `\bword(s?)\b` and the first mention per abstract. One judge (the
  author), judging categorically (mostly 1s and 4s); the guidelines did not say how fine a sense
  should be, and five *network* pairs moved agreement noticeably. Small samples: 16 pairs per word.
  XL-LEXEME's similarity spreads from about 0.45 to 0.97 even where the author judged one meaning
  (*network*, *component*), so it responds to context as well as meaning.
- **Status:** **holds**, because nothing later contradicts it; check 2b′ explained part of the
  gap without changing the verdict.
- **Retested by the rebuild:** partly. The author's judgements are kept as data (converted to a
  TSV in `curated/` before `gold_pairs` is deleted, step 1b); *component* and *having* become
  stable checks (steps 4.7 and 7.1).
- **Source and commits:** `docs/diagnostics.md`, "Check 2b", "Check 2b results" and "Correction
  pass on check 2b"; predictions 7f49200 (2026-11-10); results d001878; correction pass d9e83c0.

### Check 1b: how large a planted change can the usage-pair instrument see? (2026-11-10)

- **Line:** diagnostic check
- **Question and prediction(s):** P1, the smallest detected planted share is at most 30% and every
  larger share is detected (a share is detected when at least four of five draws are found). P2,
  the placebo (a change of source without a change of sense) is found in at most one of its five
  draws. The committed consequence: with P1 and P2 passing, a miss on a change the size of
  *attention*'s, *network*'s or *platform*'s points to the cells, not the instrument's
  sensitivity.
- **Method and settings:** Base: *attention* in psychology, 2010s → 2020s (round X's quiet
  transition), 150 usages per cell. Plant: 10%, 20%, 30%, 40% and 60% of the 2020s cell replaced
  by AI 2020s usages carrying check 1a's attention-mechanism markers (five draws per share, seed
  2026). Placebo: 40% replaced by AI 2020s usages with no marker. Found as in round X (J above the
  95th percentile of 200 shuffles and the 90th percentile of the control words' J). Pool: 291
  marked and 608 unmarked AI 2020s usages; 30 controls (90th percentile of J 0.0012).
- **Result:** P1 passed: found in 0 of 5 draws at 10% and 20%, 5 of 5 at 30%, 40% and 60%;
  smallest detected share 30%. P2 falsified: the placebo was found in 5 of 5 draws. Two readings
  were later corrected: the placebo was not clean (check 1a's reading shows the markers catch
  about half of the mechanism uses, so a 40% placebo plants roughly 17% mechanism usages), and
  the unmarked AI usages are mostly notice or consideration, computational selection and
  unmarked mechanism usages while psychology's are mostly cognitive attention, so the placebo
  moved the cell's mix of senses as well as its source (correction 9e01b82). The verdict stands
  as scored.
- **Downsides:** The placebo was not clean, so P2 on its own does not show a response to source
  with the sense held fixed; the conclusion that J follows context within a sense rests on check
  2b and round X's X5. Small samples: 150 usages per cell, five draws per share. Matching by
  `\bword(s?)\b` and the first mention per abstract. The sharp threshold between 20% and 30% is
  measured on one word in one field.
- **Status:** **holds**, as scored; its reading of P2 was corrected twice by its own source.
- **Retested by the rebuild:** experiment 0b (planted scenarios on the synthetic set); step 1b
  replaces `check_planted_pairs` (planted changes) as a live script.
- **Source and commits:** `docs/diagnostics.md`, "Check 1b", "Check 1b results", the exploratory
  note on the placebo and "Correction to check 1b's reading"; predictions 7f49200 (2026-11-10);
  results 3ba701e; correction 9e01b82.

### Check 2b′: does a second, independent judge agree with the author? (2026-11-10)

- **Line:** diagnostic check
- **Question and prediction(s):** None predicted. Reported: Spearman between the two judges over
  the 96 distinct pairs, exact agreement and agreement within one point, XL-LEXEME's Spearman
  with the second judge and with the mean of the two, and the second judge's consistency on the
  four repeated pairs. Nothing here changes check 2b's verdict.
- **Method and settings:** The same 100 items with contexts trimmed as the author saw them (30
  words either side of the target) and `docs/gold_guidelines.md` verbatim, in a file outside the
  repository. The second judge is a fresh instance of Claude (a subagent with none of the
  project's conversation), as Periti and Tahmasebi (2024) used GPT-4; it was given nothing else.
  Its 100 judgements were committed before scoring (1642c0e); it gave 29 fours, 28 threes, 33
  twos, 9 ones and one 0 (item 70, garbled text); 95 distinct pairs judged by both.
- **Result:** Author and second judge 0.616 (exact agreement 46%, within one point 81%).
  XL-LEXEME and the author 0.456; XL-LEXEME and the second judge 0.662; XL-LEXEME and the judges'
  mean 0.610. Both judges gave each of the four repeated pairs the same judgement twice. With
  the author's correction pass: XL-LEXEME and the author 0.494, author and second judge 0.657,
  XL-LEXEME and the judges' mean 0.622. The source reads check 2b's 0.431 as reflecting a
  difference in judging as much as the instrument (the author categorical, the second judge and
  XL-LEXEME in grades that move with context).
- **Downsides:** The second judge is a language model, not a person, and may share XL-LEXEME's
  sensitivity to context, so part of their agreement may be shared bias; a human second judge
  would settle it. The correction pass was made after the results were discussed, so it is a
  sensitivity check. Small samples: 95 pairs.
- **Status:** **holds**, because it records agreements and nothing later contradicts them; it has
  no verdict to withdraw.
- **Retested by the rebuild:** no.
- **Source and commits:** `docs/diagnostics.md`, "Check 2b′", "Check 2b′ results" and "Correction
  pass on check 2b"; predictions 06955f9 (2026-11-10); results 54dce3a; correction pass d9e83c0.

## Findings the README reports

Each README finding, with the entries it rests on and their statuses. Entries are named as in
the headings above; each entry's number-level detail is in its own section.

**Robust across seeds and methods**

- *Embodiment* means two different things (virtual-reality body ownership and presence in HCI;
  phenomenological corporeality in philosophy). Rests on: round 1 (P5 falsified on vocabulary;
  superseded), round 2 (R3, R4 passed; superseded), round 3 (T2, T3 passed; holds), round 4 grid
  (L2 passed in both seeds; holds), round M (no cell behind it is flagged; holds). Round 5
  (holds) qualifies it only as a positive finding from a cautious instrument. The README's
  remark that the word is about three times as frequent in HCI as in philosophy is not recorded
  in any of these entries.
- A subfield's model predicts its parent field better than the reverse (AI → computer science,
  cognitive psychology → psychology). Rests on: the first run (C1 falsified, C2 passed;
  superseded), the corrected run (D1 reversed, D2 passed; holds), round 4 whole-field (L10
  passed; holds).

**Depends on the instrument**

- HCI's abstracts changed between the 1990s and the 2020s; no direction is claimed. Rests on:
  round G (the GUI-engineering vocabulary receding and the two vocabularies coming in are its
  exploratory decomposition, and the theme-word share doubling is its observation; holds), round
  S (the classifier's 0.865 → 0.748 for HCI and the other fields' larger drifts, exploratory;
  holds), round V (the 29 technology terms; HCI's largest gain towards AI with them and psychology
  without them, exploratory; holds), round K′ (the change inside the nine venues HCI had in both
  decades, and the label widening from 915 to 3,601 sources recorded in round K; holds and
  untestable respectively). The README states that the two-part description of the vocabularies
  comes from exploratory readings (rounds G and V), not from scored predictions.
  - Language models, each decade's own (rounds J and K′): round J (holds), round K′
    (holds).
  - A classifier held at the 1990s (rounds S, V and N): round S (holds), round V (holds), round
    N (holds); the README reads N's computing result as possibly an artefact of its
    measure, per N's own correction.
  - Both instruments in one frame (round W): untestable as scored; the README calls the result
    exploratory.
- Fields converged between the 1980s and the 2010s under the language models and not under the
  classifier. Rests on: the corrected run (D5 reversed; holds), round 4 whole-field (L9 passed;
  holds), round S (S5 falsified; holds). The README's mix effect (+0.09 bits) and rate effect
  (−0.03) for field convergence are not recorded in any of the six source files (`docs/methods.md`
  has them), so no entry carries them.

**Robust in part**

- No change detected in *intentionality* in philosophy, 1980s to 2020s. Rests on: round 1 (P7
  passed; superseded), round 2 (R2; superseded), round 3 (T7; holds), round 4 grid (L3; holds).
  Round 5 (holds) is why the README reads the absence as no change detected, not as stability.
- *Representation* in AI moves towards learned representations; the symbolic 1980s holds in TWEC
  only. Rests on: round 1 (P3 passed; superseded), round 2 (R1; superseded), round 3 (T1
  falsified on the symbolic half, which is the "TWEC only" part; holds), round 4 grid (L4; holds).
- HCI's *affordance* is about experiential qualities in the 2010s in every run; the 2020s depend
  on the method. Rests on: round 1 (P4 falsified; superseded), round 2 (R7; superseded), round 3
  (T4 falsified in the 2020s; holds), round 4 grid (L5, 2010s only; holds).
- Philosophy's 2020s *consciousness* moves; the lemmatised neighbours carry both readings.
  Rests on: round 1 (P6 falsified; superseded), round 2 (R5 falsified; superseded), round 3 (T5
  falsified; holds), round 4 grid (L7 passed; holds).

**Open**

- *Information* converging between psychology and computer science: not robust. Rests on: round
  1 (P8 falsified in the opposite direction; superseded), round 2 (R6 passed; superseded), round 3
  (T6 falsified; holds), round 4 grid (L6 falsified, gap 13 against 15; holds).
- The share of HCI abstracts using philosophy's theme words doubled (observed, not predicted):
  round G's observation (22–24% to 48–52%); the entry holds.
- The direction of HCI's change: rounds J (holds), K′ (holds), S (holds), V (holds),
  N (holds) and W (untestable).

**Calibration of the sense-label instrument**

- It does not mistake a change of source for a change of meaning (the *years* placebo found in 0
  of 5 draws) and sees a planted change of a fifth. Rests on: round A (A4 and A5; holds).
- The usage-pair instrument needed a planted share of 30%, and its own placebo could not separate
  a change of source from one of sense. Rests on: check 1b (P1 passed, P2 falsified and its
  corrected reading; holds).
- Against the control words' drift it found two of four documented changes (*attention*,
  *platform*) and so stops before open questions; against shuffles alone it found all four and
  neither quiet transition, as two language models labelling the same usages did. Rests on:
  round A (A1 falsified, A2, A8; holds).

**The README's other claims that rest on entries**

- Predictions scored 3 of 9 in round 1, 7 of 9 in round 2, and 9 of 11 in round 4 (6 of 8 in
  the grid part, 3 of 3 in the whole-field part), one miss being a wrong guess about the
  instrument (neighbour pools shrank when plurals merged): round 1 (superseded), round 2
  (superseded), round 4 grid (holds; L1) and round 4 whole-field (holds).
- Retraining a cell moves an ordinary word's neighbours almost as much as a decade does (median
  Jaccard distance about 0.4 against 0.6), and control words are flagged in under 1% of
  measurements: round 2 (R9 and its reading; superseded), rounds 1, 2 and 4 grid (P1, R8, L0).
- Every held-out cell behind a robust finding generalises: round M (holds).
- Limitations about senses: clustering did not pass calibration (rounds C, C′ and C″; superseded,
  superseded, holds); the usage-pair instrument stopped at its second rung (round U pairs, round
  X, checks 1a, 1b, 2b and 2b′; all hold), with the figures "APD keeps 0.84 when the periods are
  shuffled" (round U pairs), "XL-LEXEME's similarities follow a word's context within one sense"
  (check 2b) and "a third of ordinary words' contexts drift between decades" (round X, X5); the
  sense-label classifier ranked DWUG's changes at 0.67 against 0.69 for Sonnet on the same
  usages (round E, 0.669 against 0.693 on half B; holds), with a labelling error on *prompt* and
  a change inside the 1980s on *network* (round A; holds).
- "The grid is cautious": round 5 (holds), with check 1a (holds) showing the missed changes were
  present in the cells.
- "Decade granularity is coarse" and "equal cells trade volume for comparability": round 5's
  exploratory note and check 1a (holds) for the first; round 1's P1 and round 2's R8 on *food*
  (superseded) for the control-word limit.
