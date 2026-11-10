# Predictions

Written and committed on 2026-11-09, after the field × decade models were trained and before
`drift grid` was run on them, so the predictions cannot be fitted to the results. The results
are scored against this file as written; every miss is reported.

## Disclosure

Before writing these, one pair had been inspected while testing `drift grid`: computer
science, 1950s → 1960s (*model* a candidate, *network* not). No prediction uses that pair.
Word frequencies per cell were known from the audit; the predictions are about meaning
(drift and neighbours), not about counts.

## Definitions

- **Candidate**: in the top 5% of its pair on both cosine drift and neighbourhood shift
  (`candidate` in the `drift grid` tables).
- **Cross-field rank**: a word's `cosine_rank` in `cross_field.parquet` for a pair of fields
  in one decade, a percentile (0-100) against the pair's noise floor; higher means the two
  fields use the word more differently than they use other words of its frequency.
- **Neighbours**: a word's 25 nearest words in one cell model, drawn from words with at
  least 50 occurrences in that cell.
- **Near a set**: at least 2 words of the set among the 25 neighbours. **Nearer A than B**:
  more words of A than of B among the 25 neighbours.
- **Untestable**: a prediction whose word misses 150 occurrences in a required cell is
  reported as untestable, not as passed.
- **Main grid** is `models/twec_grid/openalex`; **full cells** are
  `models/twec_grid/openalex_full`.

## Predictions

**P1 — Method sanity.** In each table, under 5% of control-word measurements are candidates,
and no control word is a candidate in more than 3 pairs. *Falsified* if either bound is
broken; the method is then wrong and no other prediction is interpreted.

**P2 — *network* in AI drifts at the connectionist and deep-learning turns** (main grid).
AI 1980s → 1990s or AI 2000s → 2010s is a candidate. *Falsified* if neither is.

**P3 — *representation* in AI moves from symbolic to learned** (main grid). Its AI 1980s
neighbours are near {knowledge, logic, rule, symbolic, reasoning, frame}, and its AI 2010s or
2020s neighbours are near {feature, learned, embedding, deep, latent, encoder}. *Falsified*
if either half fails.

**P4 — *affordance* means different things in HCI and philosophy** (full cells, 2010s and
2020s). In HCI, *affordance* is nearer DESIGN than ECOLOGICAL; in philosophy, nearer
ECOLOGICAL than DESIGN, in both decades.
DESIGN = {design, interface, user, usability, interaction, device, button}.
ECOLOGICAL = {perception, ecological, environment, action, perceive, organism, gibson}.
*Falsified* if HCI is not design-leaning or philosophy is not ecological-leaning in either
decade.

**P5 — *embodiment* differs between HCI and philosophy** (full cells, 2020s). In HCI it is
near {tangible, interaction, physical, wearable, design, gesture}; in philosophy, near
{phenomenology, mind, cognition, merleau-ponty, perception, enactive}. *Falsified* if either
side fails.

**P6 — *consciousness* in philosophy moves towards AI in the 2020s** (full cells).
Philosophy 2010s → 2020s is a candidate, and the 2020s neighbours contain at least 2 words of
{artificial, machine, ai, robot, computational, llm, chatbot, algorithm} that are not among
the 2010s neighbours. *Falsified* if either part fails.

**P7 — *intentionality* in philosophy is stable** (full cells; a null prediction). No
transition from the 1980s to the 2020s is a candidate. *Falsified* if any is.

**P8 — *information* diverges between psychology and computer science** (main grid). Its
mean cross-field rank for psychology × computer science over the 1960s and 1970s is lower
than over the 2000s and 2010s. *Falsified* if it is not lower.

**P9 — *network* converges between sociology and computer science** (main grid). Its
cross-field rank for sociology × computer science is lower in the 2010s than in the 1990s.
*Falsified* if it is not lower.

## Scoring

Each prediction is scored as passed, falsified or untestable, with the numbers behind it, in
a results section added below this line after `drift grid` runs. Predictions are not edited
after the run.

## Results (2026-11-09, `scripts/score_predictions.py`)

Scored against the definitions above, unchanged. 3 passed, 6 falsified.

| | Verdict | Numbers |
|---|---|---|
| P1 | **Falsified** (second criterion) | Control candidates 0.7% (within field, worst control *day* in 1 pair) and 0.9% (cross field), under 5%; but *food* is a candidate in 4 cross-field pairs, above the limit of 3 |
| P2 | **Falsified** | AI 1980s→1990s cosine rank 84, neighbour rank 90; 2000s→2010s 79 and 54; neither a candidate |
| P3 | **Passed** | 1980s neighbours include *knowledge, declarative, frame-based, formalism, reasoning*; 2020s include *embeddings, encoder, latent, contrastive, self-supervised, learned* (5 of the learned set) |
| P4 | **Falsified** | Philosophy leans ecological in both decades (2 vs 0: *environments, environment, perception, atmospheres, tactile*); HCI matches neither set (0 vs 0: *qualities, experiential, physicality, interactivity, metaphors*) |
| P5 | **Falsified** | HCI 2020s: *sense of embodiment, ownership, disembodiment, presence, illusion, self-location, immersion, agency* (0 of the set); philosophy 2020s: *embodied, corporeality, spatiality, temporality, affectivity, intersubjectivity* (0 of the set) |
| P6 | **Falsified** | 2010s→2020s is a candidate (ranks 98 and 99), but only 1 new word of the AI set (*computational*); 2020s neighbours include *computation, computational, intelligence, neural, substrate, information_processing* |
| P7 | **Passed** | No transition from the 1980s to the 2020s is a candidate |
| P8 | **Falsified** | Opposite direction: psychology × computer science rank 51 and 39 in the 1960s–70s, 18 and 15 in the 2000s–10s (the fields converged) |
| P9 | **Passed** (negligible) | 96.6 in the 1990s, 96.3 in the 2010s: lower, but *network* stays highly field-specific in both |

### Reading

- By P1's own rule, the results are not yet interpreted. The share of flagged controls is
  low; the breach is one control in the cross-field table. The limit of 3 pairs was set
  without computing what chance gives, and *food* is a weak cross-field control because
  fields study food differently (eating behaviour, food security, feeding circuits). The
  within-field table passes on both criteria. Cross-field controls need replacing with words
  that are stable across fields as well as across time before P8 and P9 are read.
- P4–P6 got the direction right (different senses per field; a 2020s shift in
  *consciousness*) and the vocabulary wrong: HCI's *embodiment* is virtual reality's body
  ownership and presence, philosophy's 2020s *consciousness* moved towards computational
  theories of mind. These remain falsified; any hypothesis drawn from the observed neighbours
  is new and must be tested on independent evidence (other seeds, PPMI-SVD), not on these
  results.

---

## Round 2 (committed 2026-11-09, before the seed-43 models and any of the evidence below
were inspected)

Round 1's surprises become predictions here, tested only on evidence not yet seen: the grid
retrained with seed 43 (`models/twec_grid_seed43`, same text, compass sample and settings;
only the training randomness differs), and decades or field pairs round 1 did not inspect.
The definitions of round 1 apply. A prediction passes only if **every** listed piece of
evidence agrees; any disagreement falsifies it.

**R1 — P3 replicates** (seed 43). *representation* in AI: 1980s near
{knowledge, logic, rule, symbolic, reasoning, frame}; 2010s or 2020s near
{feature, learned, embedding, deep, latent, encoder}.

**R2 — P7 replicates** (seed 43). No candidate for *intentionality* in philosophy (full
cells) in any transition from the 1980s to the 2020s.

**R3 — HCI's *embodiment* is the virtual-reality sense** (full cells; seed 43 2020s, and
seed 42 2010s). At least 3 of {presence, ownership, illusion, immersion, self-location,
avatar, virtual, agency} among its neighbours.

**R4 — philosophy's *embodiment* is phenomenological** (full cells; seed 43 2020s, and seed
42 2010s). At least 3 of {corporeality, spatiality, intersubjectivity, affectivity,
temporality, lived, flesh, embodied} among its neighbours.

**R5 — philosophy's 2020s *consciousness* turns computational** (seed 43 full cells, and seed
42 main grid). 2010s → 2020s is a candidate, and the 2020s neighbours contain at least 3 of
{computation, computational, neural, substrate, information_processing, informational,
intelligence}.

**R6 — *information* converges between psychology and computer science** (seed 43
psychology × computer science, and seed 42 cognitive psychology × computer science). The
mean cross-field rank over the 1960s–70s is at least 15 points higher than over the
2000s–10s.

**R7 — HCI's *affordance* is about experiential qualities** (seed 43 full cells, 2010s and
2020s). At least 3 of {experiential, qualities, physicality, interactivity, metaphors,
sensorial, materiality} among its neighbours in each decade.

**R8 — corrected cross-field sanity check** (seed 42 cross-field table). Cross-field controls
are the 15 words of `data/control_words.txt` whose per-million rate varies least across the
11 OpenAlex fields (lowest coefficient of variation, decades pooled), chosen from counts
only and written to `data/control_words_cross_field.txt`. Under 5% of their measurements
are candidates, and no control is a candidate in more pairs than the 99th percentile of a
binomial distribution with its number of measured pairs and the table's overall candidate
rate. *Falsified* if either bound is broken.

**R9 — drift exceeds training noise** (committed 2026-11-09 before the seed-43 grid finished
training). The two seeds have separate compasses, so vectors are not compared across seeds;
neighbour lists are. A word's *seed noise* in a cell is the neighbourhood shift (Jaccard over
25 neighbours drawn from words with at least 50 occurrences in that cell) between the seed-42
and seed-43 models of the same cell. For every target candidate in the seed-42 tables, its
neighbourhood shift between the two cells of its pair must exceed its seed noise in both
cells. At least 80% of the within-field target candidates and at least 80% of the
cross-field target candidates pass. *Falsified* if either share is below 80%.

## Round 2 results (2026-11-09, `scripts/score_predictions_round2.py`)

Scored against the round 2 definitions, unchanged. 7 passed, 2 falsified.

| | Verdict | Numbers |
|---|---|---|
| R1 | **Passed** | Seed 43 AI 1980s neighbours include *knowledge, declarative, frame-based, procedural, formalism*; 2020s include *embeddings, encoder, learned, contrastive, latent* (5 of the learned set) |
| R2 | **Passed** | Seed 43: no candidate for *intentionality* in philosophy (full cells) in any transition |
| R3 | **Passed** | 6 of the VR set in both the seed-43 2020s and the seed-42 2010s: *ownership, presence, illusion, self-location, agency, immersion* |
| R4 | **Passed** | 6 and 7 of the phenomenological set: *corporeality, temporality, affectivity, spatiality, intersubjectivity* |
| R5 | **Falsified** | Seed 43 full cells: a candidate with 7 computational neighbours (*computation, computational, substrate, neural, intelligence, informational*); seed 42 main grid: not a candidate (3 hits) |
| R6 | **Passed** | Seed 43 psychology × computer science: 1960s–70s mean 36, 2000s–10s mean 10; seed 42 cognitive psychology × computer science: 55 and 36 (gap 19) |
| R7 | **Passed** | Seed 43 HCI: 6 and 5 of the experiential set (*physicality, qualities, experiential, interactivity, sensorial, tangible*) |
| R8 | **Falsified** | Cross-field controls 0.96% candidates, but *food* exceeds its binomial bound (4 of 27 pairs, bound 3) |
| R9 | **Passed** | 54 of 54 within-field and 473 of 473 cross-field target candidates change more between their cells than between two trainings of the same cell |

### Reading

- Five observations from round 1 now hold on evidence that did not exist when they were
  predicted: *embodiment* has two senses (virtual-reality body ownership and presence in HCI,
  phenomenological corporeality in philosophy); HCI's *affordance* is about experiential
  qualities; *information* converged between psychology, cognitive psychology and computer
  science; *representation* in AI moved from symbolic to learned representations; and
  *intentionality* in philosophy is stable.
- Training noise is large: retraining a cell changes an ordinary word's 25 neighbours by a
  median Jaccard distance of about 0.4, against about 0.6 between consecutive decades. Raw
  neighbourhood shift is mostly noise; the top-5% candidates (median 0.89, minimum 0.72)
  clear it. Read neighbourhood results only through the noise floor.
- R5 marks the resolution limit: philosophy's computational turn in *consciousness* is robust
  in the full cells across seeds but does not clear the floor in 10,000-abstract cells. A
  full-cell version is a candidate for round 3.
- Cross-field controls remain the weak point: *food* differs across fields in meaning while
  being evenly frequent, so frequency-based control selection does not screen it out.

---

## Round 3 (committed 2026-11-09, before the PPMI-SVD grid was built)

Round 2's confirmed findings, tested with a second method: count-based PPMI-SVD vectors per
cell (`ppmi grid`, `models/ppmi_grid`), which share nothing with TWEC but the text. PPMI cells
are factorised separately and not aligned, so every test uses neighbour lists or
neighbourhood shift, never cosine drift. Neighbours and *near a set* are defined as in round
1. A PPMI *shift rank* is a word's `shift_rank` from `compare_cells` and `add_noise_floor`
run on two PPMI cells (the same functions as `drift grid`). A prediction passes only if every
listed piece of evidence agrees.

**Disclosure.** While testing `ppmi grid`, one PPMI cell was inspected: philosophy, 1990s
(main grid), neighbours of *consciousness*. No prediction below uses that cell.

**T0 — the two methods agree in general.** For the target words measured (at least 150
occurrences) in the full philosophy and full HCI cells of the 2010s and 2020s, the median
Jaccard similarity between their TWEC (seed 42) and PPMI neighbour lists is at least 0.2.

**T1 — *representation* in AI** (main grid). 1980s near
{knowledge, logic, rule, symbolic, reasoning, frame}; 2010s or 2020s near
{feature, learned, embedding, deep, latent, encoder}.

**T2 — HCI's *embodiment*** (full cells, 2010s and 2020s). At least 3 of {presence,
ownership, illusion, immersion, self-location, avatar, virtual, agency} in each decade.

**T3 — philosophy's *embodiment*** (full cells, 2010s and 2020s). At least 3 of
{corporeality, spatiality, intersubjectivity, affectivity, temporality, lived, flesh,
embodied} in each decade.

**T4 — HCI's *affordance*** (full cells, 2010s and 2020s). At least 3 of {experiential,
qualities, physicality, interactivity, metaphors, sensorial, materiality} in each decade.

**T5 — philosophy's *consciousness* turns computational** (full cells). PPMI shift rank for
2010s → 2020s above 95, and at least 3 of {computation, computational, neural, substrate,
information_processing, informational, intelligence} among the 2020s neighbours.

**T6 — *information* converges** (main grid, psychology × computer science). Mean PPMI
shift rank between the two fields over the 1960s–70s at least 15 points higher than over the
2000s–10s.

**T7 — *intentionality* stable** (full cells, philosophy). No transition from the 1980s to
the 2020s has a PPMI shift rank above 95.

## Round 3 results (2026-11-09, `scripts/score_predictions_round3.py`)

Scored against the round 3 definitions, unchanged. 4 passed, 4 falsified.

| | Verdict | Numbers |
|---|---|---|
| T0 | **Passed** | Median Jaccard between TWEC and PPMI neighbour lists 0.32 over 144 word-cells |
| T1 | **Falsified** | 1980s: 1 of the symbolic set (*knowledge*; the list also has *declarative, procedural, frame-based*, outside the set); 2010s and 2020s: 3 and 5 of the learned set (*embeddings, latent, contrastive, learned, encoder, feature*) |
| T2 | **Passed** | 4 of the VR set in each decade (*ownership, illusion, self-location, agency*) |
| T3 | **Passed** | 7 and 6 of the phenomenological set (*corporeality, intersubjectivity, temporality, spatiality, affectivity*) |
| T4 | **Falsified** | 2010s: 6 of the experiential set; 2020s: 2 (*experiential, qualities*; the list leans *interactional, relational, affective, theories*) |
| T5 | **Falsified** | Shift rank 91, below 95; 1 of the computational set; 2020s neighbours *qualia, orch-or, workspace, gwt, iit, integrated, substrates* |
| T6 | **Falsified** | PPMI shift rank, psychology × computer science: 76 and 58 (1960s–70s), 63 and 85 (2000s–10s): no convergence |
| T7 | **Passed** | Shift ranks 60, 79, 82, 76; none above 95 |

### Status of the findings after three rounds

Evidence: TWEC seed 42 (round 1), TWEC seed 43 and unseen decades or field pairs (round 2),
PPMI-SVD (round 3).

- **Robust across seeds and methods.** *embodiment* has two senses: virtual-reality body
  ownership and presence in HCI, phenomenological corporeality in philosophy (passed every
  test, rounds 2 and 3). *intentionality* in philosophy is stable from the 1980s to the 2020s
  (P7, R2, T7).
- **Robust in part.** *representation* in AI moves towards learned representations in every
  run; its symbolic 1980s passes in TWEC but not under the committed word set in PPMI. HCI's
  experiential *affordance* holds in the 2010s in every run; the 2020s depend on the method.
- **Not robust.** *information* converging between psychology and computer science appears in
  both TWEC seeds but not in PPMI-SVD. Philosophy's 2020s *consciousness* changes in both
  methods but is described differently by each (computational vocabulary in TWEC, named
  theories of consciousness in PPMI) and misses the committed thresholds in PPMI.

---

## Round 4 (committed 2026-11-10, before the lemmatised corpus was built)

The cells are rebuilt from a lemmatised corpus: regular plurals collapse onto their singular
and British spellings onto American ones (`corpus lemmas learn|apply`, rules in
`src/interpretant/corpus/lemmas.py`, overrides in `data/lemma_overrides.tsv`); verbs,
participles and *-ing* nominalisations are left as they are. Phrases are relearned on the
lemmatised text, then the grid is retrained with seeds 42 and 43, PPMI-SVD is rebuilt, and
`drift grid` and `drift perplexity` are rerun. The lemmatised grid takes the default paths
(`models/twec_grid`, `data/drift/*.parquet`); the unlemmatised artefacts rounds 1–3 were
scored on stay under `*_unlemmatised` paths, and the scorers of those rounds point there.

The definitions of round 1 apply (candidate, neighbours from words with at least 50
occurrences, *near a set* = at least 2 of the set among the 25 neighbours; the R-round sets
ask for at least 3). The word sets are those of rounds 1–3, unchanged; they contain no plural
or British form, so lemmatisation does not favour them.

**Disclosure.** The lemma map was reviewed before these predictions: its 200 most frequent
mappings, the spelling merges, and the share of each unlemmatised cell's vocabulary it
touches (4.6% and 5.6% of the full philosophy and HCI 2020s cells, 9.9% to 12.6% of the
main-grid and books cells inspected). No model, drift table or cross-perplexity table of the
lemmatised corpus existed. The audit's reliable-transition count will not change by design:
the target words were already mapped onto one lemma by `TARGET_VARIANTS` at preprocessing,
so the lemmatisation changes the neighbourhoods and the noise floor, not the targets' counts.

**L0 — method sanity.** In each lemmatised table, under 5% of control-word measurements are
candidates, and no control word is a candidate in more than 3 pairs (the round-1 rule; *food*
exceeded it in rounds 1 and 2 and is expected to again, and is reported as such). *Falsified*
if the 5% bound is broken by any control other than *food*, or if any control other than
*food* exceeds 3 pairs.

**L1 — lemmatisation did its job.** (a) In every cell of the seed-42 lemmatised grid the
vocabulary (the `.kv` keys) is at least 4% smaller than in the same unlemmatised cell, and at
least 8% smaller in every main-grid and books cell. (b) In each of the full philosophy and HCI
cells of the 2010s and 2020s, the neighbour pool (words with at least 50 occurrences) is at
least 3% larger than in the unlemmatised cell. *Falsified* if any cell misses (a) or any of
the four cells misses (b).

**L2 — *embodiment*'s two senses survive** (full cells, 2020s, both seeds). HCI: at least 3
of {presence, ownership, illusion, immersion, self-location, avatar, virtual, agency}.
Philosophy: at least 3 of {corporeality, spatiality, intersubjectivity, affectivity,
temporality, lived, flesh, embodied}. *Falsified* if either side fails in either seed.

**L3 — *intentionality* stable** (full cells, philosophy, both seeds). No candidate
transition from the 1980s to the 2020s. *Falsified* if any transition is a candidate in
either seed.

**L4 — *representation* in AI** (main grid, seed 42). 1980s near {knowledge, logic, rule,
symbolic, reasoning, frame}; 2010s or 2020s near {feature, learned, embedding, deep, latent,
encoder}. *Falsified* if either half fails.

**L5 — HCI's *affordance* is experiential** (full cells, seed 42, 2010s). At least 3 of
{experiential, qualities, physicality, interactivity, metaphors, sensorial, materiality}.
*Falsified* otherwise.

**L6 — *information* converges between psychology and computer science** (main grid, seed
42; stated as a prediction for the first time, after appearing in both unlemmatised TWEC
seeds and not in PPMI). The mean cross-field cosine rank over the 1960s–70s is at least 15
points higher than over the 2000s–10s. *Falsified* if the gap is under 15.

**L7 — philosophy's *consciousness* moves in the 2020s** (full cells, seed 42). 2010s → 2020s
is a candidate. The direction of the move (computational vocabulary or named theories of
consciousness) is not predicted: the two unlemmatised methods disagreed on it, and the 2020s
neighbours are reported for the record. *Falsified* if the transition is not a candidate.

Lemmatised PPMI-SVD cells are built for the record and for later rounds; no prediction in
this round uses them.

## Scoring

`scripts/score_predictions_round4.py`, committed before the lemmatised corpus exists, scores
L0–L7 and prints the evidence for each. Results are appended below after the run; the
predictions above are not edited.

## Round 4 results (2026-11-09, `scripts/score_predictions_round4.py`)

Scored as written on the lemmatised grid (seeds 42 and 43, `data/drift/*.parquet`,
`data/phrased/audit.json`). 6 passed, 2 falsified; L8–L10 are scored in
`predictions_cross_perplexity.md`.

| | Verdict | Numbers |
|---|---|---|
| L0 | **Passed** | Control candidates 0.4% (within field; worst *forest*, 1 pair) and 0.3% (cross field; worst *home*, 3 pairs); *food* in 1 cross-field pair |
| L1 | **Falsified** | Vocabulary shrink: median 8.4%, minimum 4.1%; 37 main-grid and books cells between 6.0% and 8.0%, under the 8% floor (none under 4%). Neighbour pools (words with at least 50 occurrences) shrank instead of growing: −11.4% and −11.7% (philosophy 2010s, 2020s), −12.5% and −12.6% (HCI) |
| L2 | **Passed** | Both seeds: HCI 2020s 5 of the VR set (*illusion, immersion, ownership, presence, self-location*); philosophy 2020s 6 of the phenomenological set (*affectivity, corporeality, embodied, intersubjectivity, spatiality, temporality*) |
| L3 | **Passed** | No candidate transition in either seed; highest cosine rank 93 (seed 42, 2000s → 2010s) with shift rank 5 |
| L4 | **Passed** | 1980s 2 of the symbolic set (*knowledge, reasoning*); 2010s 3 and 2020s 5 of the learned set (*embedding, encoder, feature, latent, learned*) |
| L5 | **Passed** | 5 of the experiential set (*experiential, interactivity, materiality, physicality, sensorial*) |
| L6 | **Falsified** | Cross-field cosine rank, psychology × computer science: 38 and 23 (1960s, 1970s), 12 and 24 (2000s, 2010s); early mean 30, late mean 18, gap 13, under 15 |
| L7 | **Passed** | Philosophy 2010s → 2020s cosine rank 99, shift rank 96, a candidate; 2020s neighbours *qualia, conscious, cognition, mind, computational, computation, reality, substrate, information_processing, neural, substrate-independent, intelligence, informational, nonlocal, sentience* |

### Reading

- **Lemmatisation changed the instrument less than expected, and in one respect the other way.**
  Vocabularies shrank by 4–10%, not the 8%+ predicted for the main grid, because many plural
  types had no singular in the cell to merge into. The neighbour pools shrank by about 12%
  rather than growing: when both *structure* and *structures* clear 50 occurrences, merging them
  removes a pool member, and that outweighs the rare words the merge lifted over the threshold.
  L1's reasoning was wrong, and the prediction is reported as falsified; the mapping itself did
  what it was built to do (8.3% of tokens, no wrong merge found in review).
- **The findings survive.** *Embodiment*'s two senses, *intentionality*'s stability,
  *representation*'s move from symbolic to learned and HCI's experiential *affordance* all pass
  on the lemmatised grid, the first two in both seeds. Philosophy's 2020s *consciousness* is a
  candidate again, and its lemmatised neighbours carry both earlier readings at once (*qualia,
  substrate, computational, information_processing, sentience*).
- ***Information*'s convergence is not robust.** It appeared in both unlemmatised TWEC seeds and
  not in PPMI-SVD; on the lemmatised grid the gap is 13 points, under the committed 15. It moves
  to the not-robust list.

---

## Round 5: does the grid find changes whose history is known? (committed 2026-11-10, before any drift rank below was inspected)

Every robust finding rests on the TWEC grid, but the grid has been checked only against its own
noise (controls, seeds, a second method), never against a change whose history is documented.
The one such test it took failed: P2, *network* at AI's connectionist turn. So it is not known
how sensitive the grid is, and therefore what its nulls mean (*intentionality*'s stability).
This round asks the grid to find documented changes, within and across fields, and to stay
quiet where nothing documented happened. The sense rounds (`predictions_senses.md`) showed
that an instrument can look sound against noise and still fail on known answers.

### Method (fixed before any rank is inspected)

- **Grid.** The lemmatised main grid (`models/twec_grid/openalex`, seed 42, and
  `models/twec_grid_seed43/openalex`). Seed 42's ranks are read from
  `data/drift/within_field.parquet` and `cross_field.parquet`; seed 43's are computed with the
  same functions (`compare_cells`, `add_noise_floor`) on the seed-43 cells, as round 4 did.
- **Candidate**, as in round 1: in the top 5% of its pair on both cosine drift and
  neighbourhood shift.
- **Known changes**, each measurable (at least 150 occurrences in both cells):
  - *attention* in AI, 2010s → 2020s (379 and 1,121 occurrences): the attention mechanism of
    neural networks (Bahdanau et al. 2014; Vaswani et al. 2017);
  - *platform* in communication, 2000s → 2010s (173 and 822): digital and social-media
    platforms (Gillespie 2010);
  - *network* in AI, 1980s → 1990s (2,119 and 3,924): the revival of neural networks after
    backpropagation (Rumelhart, Hinton and Williams 1986);
  - *alignment* in AI, 2010s → 2020s (201 and 533): aligning language models with human
    intent (Ouyang et al. 2022), a weaker case because the 2020s also use cross-modal
    alignment.
- **Quiet transitions**, where no change is documented: *attention* in AI 1990s → 2000s, and
  *attention* in psychology 2010s → 2020s.
- **Cross-field.** *attention*'s cross-field cosine rank between AI and each of psychology,
  cognitive psychology and neuroscience, in the 1990s and the 2020s.
- Already seen: which words and transitions are measurable and their counts; round C and C′'s
  readings of *attention*'s senses (the mechanism holds about 55% of AI's 2020s occurrences);
  P2's unlemmatised result for *network* (AI 1980s → 1990s cosine rank 84, neighbour rank 90,
  not a candidate). Not inspected: any lemmatised drift rank for these words, in either seed.

### Predictions

**E1 — *attention* in AI, 2010s → 2020s, is a candidate in both seeds.** *Falsified* if not a
candidate in either.

**E2 — *platform* in communication, 2000s → 2010s, is a candidate in both seeds.** *Falsified*
if not a candidate in either.

**E3 — *network* in AI, 1980s → 1990s, is a candidate in both seeds** (P2 retested on the
lemmatised grid). *Falsified* if not a candidate in either.

**E4 — *alignment* in AI, 2010s → 2020s, is a candidate in both seeds.** *Falsified* if not a
candidate in either.

**E5 — The grid stays quiet where nothing documented changed.** *attention* in AI 1990s →
2000s and in psychology 2010s → 2020s is not a candidate, in either seed. *Falsified* if either
is a candidate in either seed.

**E6 — Across fields, the new sense shows.** In both seeds, *attention*'s cross-field cosine
rank between AI and each of psychology, cognitive psychology and neuroscience is higher in the
2020s than in the 1990s, and at least 90 in the 2020s. *Falsified* if any of the three pairs
fails either condition in either seed.

Reading rule:

- **At least two of E1–E3 pass, and E5 passes:** the grid finds documented changes and does not
  flag quiet ones. The robust findings stand with a calibrated instrument, and a pre-registered
  survey of field-specific meanings on the grid may be designed.
- **Fewer than two of E1–E3 pass:** the grid misses documented changes. Its candidates remain
  evidence (the controls stay quiet), but its nulls are not evidence of stability: the README
  qualifies *intentionality*'s stability accordingly, and no survey is built on the grid.
- **E5 fails:** the grid flags transitions where nothing documented happened, and its
  candidates are read with that rate in mind; the README says so.
- E4 is the weak case and does not count towards the rule. E6 decides whether cross-field
  ranks (the measure behind P8, P9, R6, T6 and L6) can register a new sense; it does not affect
  the within-field reading.
- Whatever the outcome, no round follows to repair the grid.

Scored by `scripts/score_grid_calibration.py`, committed before it is run.

## Round 5 results (2026-11-10, `scripts/score_grid_calibration.py`)

Scored as written. 2 passed, 4 falsified.

| | Verdict | Numbers (cosine rank, neighbour rank) |
|---|---|---|
| E1 | **Falsified** | *attention* in AI, 2010s → 2020s: seed 42 92 and 98, seed 43 90 and 97; not a candidate in either |
| E2 | **Falsified** | *platform* in communication, 2000s → 2010s: 61 and 13, 61 and 31 |
| E3 | **Falsified** | *network* in AI, 1980s → 1990s: 86 and 23, 82 and 47 |
| E4 | **Passed** | *alignment* in AI, 2010s → 2020s: 100 and 100, 100 and 98; a candidate in both |
| E5 | **Passed** | Quiet transitions, none a candidate: *attention* in AI 1990s → 2000s 17 and 41, 47 and 69; in psychology 2010s → 2020s 17 and 42, 27 and 42 |
| E6 | **Falsified** | *attention*'s cross-field cosine rank, 1990s → 2020s, rises in all six comparisons but stays under 90: AI × psychology 2 → 53 and 2 → 73, AI × cognitive psychology 32 → 61 and 32 → 75, AI × neuroscience 24 → 29 and 33 → 51 |

### Reading

Fewer than two of E1–E3 passed, so by the reading rule the grid misses documented changes. Its
candidates remain evidence: E5 passed, and the control words have stayed quiet in every round.
Its nulls are not evidence of stability, so *intentionality*'s "stability" (P7, R2, T7, L3) is
restated in the README as no change detected. No survey of field-specific meanings is built on
the grid, and, as committed, no round follows to repair it.

The grid is conservative rather than noisy: it flagged one documented change, the case judged
weakest beforehand, and none of the quiet transitions. E6 shows the cross-field ranks moving
the right way for a new sense without reaching a high rank, which bears on the cross-field
results of rounds 1–4 (P8, P9, R6, T6, L6): a modest cross-field rank does not mean two fields
use a word alike. Round 5 tested the main grid's 10,000-abstract cells; the full cells behind
*embodiment*, *intentionality* and *affordance* are larger and may be more sensitive, which is
untested.

### Exploratory, after scoring: near misses and decade boundaries

Not predicted, and not a reason to revise any verdict.

- *attention* in AI was a near miss: its neighbourhood changed more than 97–98% of words of
  its frequency, but its cosine drift ranked 90–92, under the 95 a candidate needs on both.
- *platform* and *network* did not register at all. Two explanations come to mind after the
  fact, and both are untested: communication's 2000s cell (2000–2009) may already hold the
  digital-platform sense, and AI's 1980s cell (1980–1989) straddles the revival of neural
  networks in 1986. Decade cells can place a change inside a cell instead of between two.
