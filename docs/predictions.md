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
