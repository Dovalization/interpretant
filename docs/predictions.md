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
