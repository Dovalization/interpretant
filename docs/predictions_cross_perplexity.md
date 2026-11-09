# Predictions: cross-perplexity between fields

Committed on 2026-11-09, before any cross-perplexity code was written or run. The results are
scored against this file as written; every miss is reported.

## Method (fixed before results)

- **Cells**: the main OpenAlex grid only (`data/phrased/openalex/texts`), no books (another
  genre) and no full cells (another size).
- **Language model**: one bigram model per cell with interpolated Kneser-Ney smoothing,
  discount 0.75.
- **Training size**: within a decade, every qualifying field trains on the same number of
  tokens, sampled from its abstracts with a fixed seed: the smallest qualifying field's
  training text, capped at 1,000,000 tokens. A field qualifies in a decade if its training
  text (all abstracts but the held-out ones) has at least 200,000 tokens.
- **Held-out text**: 1,000 abstracts per cell, chosen with a fixed seed, never used for
  training.
- **Vocabulary**: one per decade, the words with at least 5 occurrences across that decade's
  training samples; every other word is one unknown token.
- **Excess cross-entropy** E(A → B): the cross-entropy, in bits per token, of B's held-out
  text under A's model minus its cross-entropy under B's own model. Low E(A → B) means B's
  language is predictable from A's (Vilhena et al. 2014, "cultural holes").

## Predictions

**C0 — Own model best.** For at least 95% of qualifying cells, the field's own model gives
its held-out text the lowest cross-entropy of all qualifying fields' models in that decade.
*Falsified* if under 95%.

**C1 — Broad fields predict their subfields better than the reverse.**
E(computer science → AI) < E(AI → computer science), and
E(psychology → cognitive psychology) < E(cognitive psychology → psychology), each in at least
6 of the decades where both fields qualify (or in every such decade, if fewer than 6).
*Falsified* if either falls short.

**C2 — Subfields sit closest to their parent field.** For AI's held-out text, computer
science has the lowest excess cross-entropy among the other qualifying fields; for cognitive
psychology's, psychology does; each in at least 6 of the decades where they qualify (or in
every such decade, if fewer than 6). *Falsified* if either falls short.

**C3 — Philosophy absorbs AI's language in the 2020s.** E(AI → philosophy) is lower in the
2020s than in the 2010s. *Falsified* if it is not lower.

**C4 — HCI moves towards philosophy.** E(philosophy → HCI) is lower in the 2020s than in the
1990s. *Falsified* if it is not lower.

**C5 — Fields grow more specialised.** The mean E(A → B) over all ordered pairs of fields
qualifying in both decades is higher in the 2010s than in the 1980s. *Falsified* if it is
not higher. (The literature argues both for specialisation and for convergence; this is a
genuine open question.)

## Results

To be added below this line after the run. Predictions are not edited after the run.

### Results (2026-11-09, `scripts/score_cross_perplexity.py`)

Scored as written. 5 passed, 1 falsified; two design flaws found after the run limit what
several of them mean.

| | Verdict | Numbers |
|---|---|---|
| C0 | **Passed** | Own model best in 76 of 79 cells (96%); the 3 misses are AI and computer science in the 2010s–2020s |
| C1 | **Falsified** | computer science → AI below AI → computer science in 5 of 6 decades; psychology → cognitive psychology below the reverse in 2 of 7 |
| C2 | **Passed** | Computer science closest to AI in 6 of 6 decades; psychology closest to cognitive psychology in 7 of 7 |
| C3 | **Passed** | E(AI → philosophy) 1.61 bits (2010s), 1.56 (2020s) |
| C4 | **Passed** | E(philosophy → HCI) 2.12 bits (1990s), 1.94 (2020s) |
| C5 | **Passed** | Mean excess over 11 fields 1.08 bits (1980s), 1.20 (2010s) |

### Design flaws found after the run

- **Parent and subfield cells share abstracts.** OpenAlex files AI under computer science and
  cognitive psychology under psychology, so 18–29% of AI's abstracts are also in computer
  science's cell and 10–20% of cognitive psychology's in psychology's. A parent's model can
  have trained on the subfield's held-out abstracts, which produced negative excess values
  (computer science predicting AI's 2020s text better than AI's own model) and C0's three
  misses, and favours the parent in C1 and C2. C2's pass is therefore weak evidence.
- **Training size was equal within decades, not across them.** C3 compares two decades
  trained on 1,000,000 tokens each and is unaffected; C4 compares 858,655 with 1,000,000;
  C5 compares 205,731 (1980s) with 1,000,000 (2010s). A model trained on five times more text
  captures more field-specific detail, which raises excess on its own, so C5 may reflect
  training size as much as specialisation.
- **No uncertainty estimate.** C3's change (0.05 bits, 3%) could be sampling noise; C4's
  (0.18 bits, 9%) is larger but equally untested.

### Reading

C4 — HCI's language becoming more predictable from philosophy's between the 1990s and the
2020s, the whole-language counterpart of the *embodiment* finding — is the cleanest result.
C3 points the right way with a small effect; C5 is not interpretable as run. A corrected
follow-up (subfield abstracts removed from parent cells, one training size across compared
decades, repeated samples for uncertainty) must be committed as new predictions, not edits.

---

## Corrected run (committed 2026-11-09, before the corrected code exists)

Fixes the three flaws above. Everything not listed here is as in the first run (bigram
Kneser-Ney, discount 0.75, 1,000 held-out abstracts per cell, one vocabulary per decade,
excess cross-entropy).

- **No shared abstracts.** Any abstract that appears in more than one cell of a decade is
  removed from every cell of that decade before splitting, so no model can have trained on
  another cell's held-out text.
- **One training size across decades.** Every qualifying field in every decade trains on
  exactly 200,000 tokens; a field qualifies if its training pool, after the removal above, has
  at least 200,000 tokens.
- **Repeated samples.** The run is repeated with seeds 1 to 5; each seed draws its own
  held-out abstracts and training sample. A comparison between decades *holds* only if it
  goes the predicted way in all 5 seeds.

**D0 — Own model best.** In every seed, for at least 95% of qualifying cells, the field's own
model gives its held-out text the lowest cross-entropy.

**D1 — Broad fields predict their subfields better than the reverse.** Using the mean excess
over the 5 seeds, E(computer science → AI) < E(AI → computer science) and
E(psychology → cognitive psychology) < E(cognitive psychology → psychology), each in at least
6 of the decades where both qualify (or in every such decade, if fewer than 6).

**D2 — Subfields sit closest to their parent field.** Using the mean excess over the 5
seeds, computer science has the lowest excess for AI's text and psychology for cognitive
psychology's, each in at least 6 of the decades where they qualify (or in every such decade,
if fewer than 6).

**D3 — Philosophy absorbs AI's language in the 2020s.** E(AI → philosophy) is lower in the
2020s than in the 2010s in all 5 seeds.

**D4 — HCI moves towards philosophy.** E(philosophy → HCI) is lower in the 2020s than in the
1990s in all 5 seeds.

**D5 — Fields grow more specialised.** The mean E(A → B) over all ordered pairs of fields
qualifying in both decades is higher in the 2010s than in the 1980s in all 5 seeds.

Each prediction is *falsified* if its condition fails.

### Corrected run results (2026-11-09, `scripts/score_cross_perplexity_corrected.py`)

Scored as written. 3 passed, 3 falsified (two of them in the opposite direction).

| | Verdict | Numbers (seeds 1–5) |
|---|---|---|
| D0 | **Passed** | Own model best in 100% of cells in every seed |
| D1 | **Falsified** (reversed) | Parent → subfield lower in 0 of 6 decades for both pairs: without shared abstracts, the subfield's model predicts the parent's text better than the reverse |
| D2 | **Passed** | Computer science closest to AI and psychology closest to cognitive psychology in 6 of 6 decades |
| D3 | **Falsified** | E(AI → philosophy), 2020s minus 2010s: −0.055, −0.053, −0.041, +0.024, +0.005 |
| D4 | **Passed** | E(philosophy → HCI), 2020s minus 1990s: −0.072, −0.103, −0.065, −0.056, −0.096 |
| D5 | **Falsified** (reversed) | Mean excess over 10 fields, 2010s minus 1980s: −0.042, −0.046, −0.040, −0.061, −0.056 |

### Reading

- **HCI's language moved towards philosophy's** between the 1990s and the 2020s, robust to
  shared abstracts, training size and sampling (all 5 seeds). It is the whole-language
  counterpart of the *embodiment* finding: the field that took up *embodiment* and
  *affordance* became more predictable from philosophy overall.
- **Philosophy absorbing AI's language is not established**: the first run's 0.05-bit drop is
  within seed-to-seed noise.
- **The first run's specialisation was a training-size artefact.** With equal training, the
  fields grew closer from the 1980s to the 2010s in every seed. This convergence and D1's
  reversed asymmetry are observations from these data; each needs its own prediction and
  independent evidence before it counts as a finding.

---

## Follow-up: HCI and philosophy (committed 2026-11-09, before the evidence below was
inspected)

Uses the corrected run (`data/drift/cross_perplexity_corrected.parquet`: shared abstracts
removed, 200,000 training tokens, seeds 1–5). Only E(philosophy → HCI) for the 1990s and 2020s
had been inspected (D4); every value below other than those two is unseen. "Mean" is the mean
over the 5 seeds.

**F1 — HCI's move towards philosophy is concentrated after 2000** (the decade of embodied and
phenomenological approaches in HCI). Mean E(philosophy → HCI) falls more from the 2000s to the
2020s than from the 1990s to the 2000s. *Falsified* if the later fall is not larger.

**F2 — The movement is one-sided: HCI moves towards philosophy more than philosophy moves
towards HCI.** In every seed, E(HCI → philosophy) changes less between the 1990s and the
2020s, in absolute value, than E(philosophy → HCI) does. *Falsified* if any seed shows the
opposite.

**F3 — The words behind it are philosophy's words.** Per-word excess is the bits per
occurrence that HCI's own model saves over philosophy's model on HCI's held-out text,
pooled over the 5 seeds (both models retrained exactly as in the corrected run). Ranking the
words occurring in both decades' HCI held-out text by their fall in per-word excess from the
1990s to the 2020s, weighted by their 2020s occurrences, at least 2 of {embodiment, embodied,
affordance, experience, experiential, phenomenological, phenomenology, body, bodily, meaning,
practice, perception} are among the top 50. *Falsified* if fewer than 2 are.

### Follow-up results (2026-11-09, `scripts/score_hci_philosophy.py`)

Scored as written. 1 passed, 2 falsified.

| | Verdict | Numbers |
|---|---|---|
| F1 | **Falsified** | Mean E(philosophy → HCI): 1.700 (1990s), 1.652 (2000s), 1.521 (2010s), 1.622 (2020s); fall 1990s→2000s 0.048, 2000s→2020s 0.031. Closest in the 2010s, partly apart again in the 2020s |
| F2 | **Passed** | Per seed, change 1990s→2020s in E(philosophy → HCI) vs E(HCI → philosophy): (−0.072, +0.018), (−0.103, +0.002), (−0.065, −0.040), (−0.056, +0.003), (−0.096, −0.006) |
| F3 | **Falsified** | None of the 12 words in the top 50, which is led by *the, user, design, system, of, interface, research, virtual, digital, interaction, environment* |

### Reinterpretation of D4 and F2

E(A → B) falls when B's language moves towards A's **or** when A's language broadens to cover
B's. F3 points to the second: the words whose excess fell most are HCI's own vocabulary (*user,
design, interface, virtual, digital, interaction*), which philosophy's model learned to predict,
not philosophy's vocabulary. An exploratory count, made after scoring and not a test, agrees:
per million tokens, philosophy's abstracts use *digital* 6 → 126, *interface* 10 → 49,
*virtual* 10 → 40, *user* 8 → 29 and *design* 47 → 134 between the 1990s and the 2020s.

So the robust result is that **philosophy's language model came to predict HCI's abstracts
better, while HCI's model did not come to predict philosophy's** — consistent with philosophy
turning towards digital technology rather than with HCI borrowing philosophy's language. The
earlier description, "HCI's language moved towards philosophy's", was wrong and has been
corrected in the README and ROADMAP. Philosophy's turn to technology is a hypothesis from
these data and needs its own test. F3's ranking, weighted by occurrences, lets very frequent
words dominate; a per-occurrence ranking would be the better design for such a test.

---

## Philosophy's turn towards technology (committed 2026-11-09, before the evidence below was
inspected)

Tests the hypothesis from the follow-up: philosophy's language broadened into technology. All
evidence is unseen: philosophy's pairs with computer science and AI in the corrected run, the
full philosophy cells with a technology lexicon that excludes every word of the exploratory
count (*digital, virtual, design, user, interface, technology, algorithm*), and a word ranking
per occurrence rather than weighted by frequency.

**H1 — Philosophy's model comes to predict computing's text better.** In all 5 seeds,
E(philosophy → computer science) and E(philosophy → AI) are each lower in the 2020s than in
the 1990s.

**H2 — The change is one-sided.** In all 5 seeds, E(computer science → philosophy) changes
less between the 1990s and the 2020s, in absolute value, than E(philosophy → computer
science) does.

**H3 — Philosophy writes more about technology, decade by decade.** In the full philosophy
cells, the share of abstracts containing at least one word of {internet, online, software,
computer, computers, computational, robot, robots, artificial_intelligence, algorithmic, data,
machine, machines, web, cyber, automation, platform, platforms} rises at every step from the
1980s to the 2020s.

**H4 — Philosophy reaches computing through ethics and society.** On computer science's
held-out text, per-word excess (bits per occurrence that computer science's own model saves
over philosophy's, pooled over the 5 seeds) is compared between the 1990s and the 2020s for
words with at least 100 pooled occurrences in each decade. Ranked by their fall per
occurrence, at least 3 of {ethics, ethical, social, society, human, humans, privacy, trust,
moral, responsibility, values, fairness} are in the top 50.

Each prediction is *falsified* if its condition fails.

### Technology-turn results (2026-11-09, `scripts/score_technology_turn.py`)

Scored as written. 1 passed (by the letter only), 3 falsified.

| | Verdict | Numbers (seeds 1–5) |
|---|---|---|
| H1 | **Falsified** | Change 1990s→2020s, E(philosophy → computer science): −0.025, +0.016, +0.079, +0.103, +0.109; E(philosophy → AI): +0.060, +0.031, +0.035, +0.003, −0.013 |
| H2 | **Passed** (letter only) | E(computer science → philosophy) changed less in every seed (+0.001, +0.010, −0.048, −0.014, +0.015), but E(philosophy → computer science) mostly rose, so the pass does not support the hypothesis |
| H3 | **Falsified** | Share of full-cell philosophy abstracts with a technology word: 3.1% (1980s), 3.3%, 2.9%, 3.6%, 6.5% (2020s) |
| H4 | **Falsified** | None of the 12 ethics and society words in the top 50, led by *digital, signal, compression, memory, interface, implementation* |

### Reading

The hypothesis drawn from the exploratory count fails on fresh evidence. Philosophy's model got
worse, not better, at predicting computer science and AI abstracts from the 1990s to the 2020s,
so philosophy did not broadly turn towards computing's language; its technology vocabulary
jumps only in the 2020s, while philosophy and HCI were closest in the 2010s. The robust result
stands as measured — philosophy's model predicts HCI's abstracts better over time, and not the
reverse — but it is specific to HCI, and neither explanation tested so far (HCI borrowing
philosophy's language, philosophy broadening into technology) accounts for it. Its cause is
open.

---

## Round 4: the lemmatised corpus (committed 2026-11-10, before the lemmatised corpus was
built)

`drift perplexity` is rerun on the lemmatised cells with the corrected design (shared abstracts
removed, 200,000 training tokens, 1,000 held-out abstracts, seeds 1–5; output
`data/drift/cross_perplexity_corrected.parquet`, the unlemmatised table having moved to
`data/drift/unlemmatised/`). Lemmatisation shrinks every vocabulary, so absolute excess values
will differ from the unlemmatised run and are not compared across the two corpora; only
directions within the lemmatised run are predicted. "Mean" is the mean over the 5 seeds.

**L8 — philosophy's model predicts HCI better over time, one-sidedly.** E(philosophy → HCI)
is lower in the 2020s than in the 1990s in all 5 seeds, and in every seed E(HCI → philosophy)
changes less between the 1990s and the 2020s, in absolute value, than E(philosophy → HCI)
does (D4 and F2 on the lemmatised corpus). *Falsified* if either part fails in any seed.

**L9 — fields converge.** The mean E(A → B) over all ordered pairs of fields qualifying in
both decades is lower in the 2010s than in the 1980s in all 5 seeds (the reversal observed
in D5, now predicted). *Falsified* if any seed shows the opposite.

**L10 — the subfield asymmetry holds.** Using the mean excess over the 5 seeds,
E(AI → computer science) < E(computer science → AI) and E(cognitive psychology → psychology)
< E(psychology → cognitive psychology) in at least 5 of the 6 decades where both qualify
(the reversal observed in D1, now predicted). *Falsified* if either pair falls short.

Scored by `scripts/score_predictions_round4.py`. Results are appended below after the run.

### Round 4 results (2026-11-09, `scripts/score_predictions_round4.py`)

Scored as written on the lemmatised corpus. 3 passed.

| | Verdict | Numbers (seeds 1–5) |
|---|---|---|
| L8 | **Passed** | Change 1990s → 2020s in E(philosophy → HCI): −0.071, −0.103, −0.066, −0.057, −0.097; in E(HCI → philosophy): +0.014, +0.001, −0.045, +0.002, −0.005, smaller in absolute value in every seed |
| L9 | **Passed** | Mean excess over 10 fields, 2010s minus 1980s: −0.050, −0.049, −0.040, −0.059, −0.056 |
| L10 | **Passed** | AI → computer science below the reverse in 6 of 6 decades; cognitive psychology → psychology below the reverse in 6 of 6 |

### Reading

The one-sided philosophy → HCI effect, the convergence of fields between the 1980s and the
2010s, and the subfield → parent asymmetry all hold on the lemmatised corpus with values within
a few thousandths of a bit of the unlemmatised run. The convergence and the asymmetry, first seen
as reversals of D5 and D1, are now predicted results. The cause of the philosophy → HCI effect
remains open.

---

## Round G: where the philosophy → HCI gain sits (committed 2026-11-09, before any
per-abstract number existed)

E(philosophy → HCI) fell from the 1990s to the 2020s in every seed, on both corpora (D4, L8),
and HCI → philosophy did not move (F2). Two whole-field explanations failed: HCI borrowing
philosophy's words (F3) and philosophy turning towards technology (H1–H4). This round asks
*where* in HCI's text the gain sits: in a subset of abstracts (topics the two fields came to
share) or across all of them (a register philosophy's model came to cover).

### Method (fixed before results)

- Lemmatised cells (`data/phrased/openalex/texts`), the corrected design (shared abstracts
  removed, 200,000 training tokens, 1,000 held-out abstracts, seeds 1–5), decades 1990 and
  2020, rebuilt with `build_decade` exactly as L8 was scored.
- **Per-abstract excess** of an HCI held-out abstract: its bits per token under philosophy's
  model minus under HCI's own model (end token included). Its mean over the 1,000 abstracts is
  E(philosophy → HCI) up to length weighting.
- The **fall** of a statistic is its 1990s value minus its 2020s value.
- A **philosophy-flavoured** HCI abstract contains at least one of {ethics, ethical, moral,
  phenomenology, phenomenological, embodiment, embodied, ontology, ontological, epistemic,
  epistemological, hermeneutic, agency, autonomy, meaning, experience, value, values, critical,
  theory, theoretical, reflection, reflexive, normative}. The list is fixed here.

### Predictions

**G1 — The fall is spread across abstracts, not carried by a subset.** In every seed, the
median per-abstract excess falls by at least 70% of the mean's fall. *Falsified* if any seed
is under 70%.

**G2 — The lower tail does not do the work.** In every seed, the fall of the 25th percentile
is at most 1.5 times the fall of the 75th percentile. *Falsified* if any seed exceeds 1.5.

**G3 — The gain reaches abstracts with none of philosophy's themes.** In at least 4 of 5
seeds, the fall in mean excess over the non-flavoured HCI abstracts is at least 60% of the
fall over the flavoured ones. *Falsified* otherwise; if the flavoured fall is the larger one,
the effect is topical after all.

**G4 — The vocabulary doing it is small.** Pooling the five seeds, per-word excess as in F3
(bits HCI's model saves over philosophy's, per occurrence), scored per word as the fall in
per-occurrence excess times 2020s occurrences: the 50 highest-scoring words account for at
least 40% of the sum of all positive scores. *Falsified* if under 40%.

**G5 — One-sided at the document level.** In every seed, the median per-abstract excess of
philosophy's held-out abstracts under HCI's model changes between the 1990s and the 2020s by
less, in absolute value, than half the fall of HCI's median under philosophy's model.
*Falsified* if any seed fails.

**Reading rule, written now.** G1–G3 passing together means a register effect: philosophy's
model learned to cover how HCI writes, everywhere in HCI. G1 or G3 failing with the lower tail
or the flavoured partition carrying the fall means topics. G4 says whether the register is a
few words or many. Exploratory and unscored: the ten HCI 2020s held-out abstracts with the
lowest excess and the ten with the highest, first 25 words each, printed for reading.

Scored by `scripts/score_decomposition.py`, committed before it is run. Results are appended
below after the run.

### Round G results (2026-11-09, `scripts/score_decomposition.py`)

Scored as written. 3 passed, 2 falsified.

| | Verdict | Numbers (seeds 1–5) |
|---|---|---|
| G1 | **Passed** | Median fall / mean fall: 0.84, 1.39, 1.21, 1.53, 0.84 |
| G2 | **Falsified** | q25 fall / q75 fall: 1.00, 1.63, 0.55, 2.16, 0.82; over 1.5 in seeds 2 and 4 |
| G3 | **Passed** | Fall on plain abstracts vs flavoured ones (bits): (0.144, −0.026), (0.092, 0.072), (0.030, 0.070), (0.069, −0.005), (0.082, 0.032); plain at or above 60% of flavoured in 4 of 5 seeds, and larger than it in 4 |
| G4 | **Falsified** | The 50 highest-scoring words hold 38.8% of the positive score (needed 40%); top 20: *the, user, design, system, of, interface, research, interaction, a, environment, digital, are, virtual, information, factor, present, on, device, development, is* |
| G5 | **Passed** | (HCI median fall under philosophy, change of philosophy's median under HCI): (0.120, +0.031), (0.190, +0.022), (0.109, −0.007), (0.163, +0.026), (0.093, −0.003) |

Per-seed means and medians of HCI's per-abstract excess under philosophy's model, 1990s →
2020s: means 1.81 → 1.66, 1.73 → 1.60, 1.70 → 1.61, 1.77 → 1.66, 1.75 → 1.64; medians
1.80 → 1.68, 1.77 → 1.58, 1.74 → 1.63, 1.78 → 1.62, 1.75 → 1.66.

### Reading

By the rule written before the run, this is a **register effect, not a topical one**. The
whole distribution of HCI abstracts moved: the median fell at least as much as the mean, and
the abstracts that gained most were the ones with none of philosophy's theme words, while the
flavoured ones barely moved. Philosophy's model did not learn a philosophical corner of HCI; it
learned to predict the ordinary HCI abstract. G2's failure is a noisy statistic (the two
quartiles' falls swing from 0.55 to 2.16 across seeds) and does not point either way; G4 missed
by 1.2 points, so the gain is carried by somewhat more than fifty word types, led by function
words and HCI's core nouns.

Two observations from the run, not predicted and not findings:

- **HCI's use of philosophy's theme words doubled.** The share of HCI held-out abstracts
  containing at least one of the fixed theme words (*experience, value, meaning, agency,
  embodied, ethics, ...*) rose from 22–24% in the 1990s to 48–52% in the 2020s in every seed.
  The themed vocabulary spread through HCI; the gain in predictability did not come from those
  abstracts.
- **The register the model learned is HCI's expository one.** The HCI 2020s abstracts cheapest
  under philosophy's model are reflective and first-person (a position paper on prototypes as
  instruments of knowledge, personal reflections on thirty years of VR, an essay on software
  not doing what you want); the costliest are machine-learning system papers (sign-language
  recognition, gesture recognition, traffic-police gestures).

Together with F3 (the words whose excess fell were *user, design, interface, virtual, digital,
interaction*) and the exploratory counts in the follow-up (philosophy's use of *digital*,
*interface*, *user* and *design* rising several-fold per million tokens), the most economical
account is that **philosophy acquired HCI's objects, not computing's methods**: by the 2020s
philosophy writes about users, interfaces, design and the digital often enough that its model
predicts the ordinary HCI abstract better, while it predicts computer science and AI, whose
core vocabulary is algorithmic, no better than before (H1). HCI's register did not move towards
philosophy at all (F2, G5). This is a reading built from three rounds of evidence, not a scored
prediction; the test that would settle it is a pre-registered contrast, in the full philosophy
cells, between HCI's object nouns and computing's method nouns.

### Exploratory, after scoring: function words, and a mix/rate decomposition

Prompted by the question whether *the* and *of* should count as words in G4. Not predicted, not
scored; the numbers are from the same models.

- **Function words are 15% of G4's positive score**, content words 85%; the content-word part
  is broad (top 50 words 37%, top 100 48%), led by *user, design, system, interface, research,
  interaction, environment, digital, virtual, information, present, device, this_paper,
  analysis, experiment*. G4's verdict does not depend on function words.
- **Per-word rates got worse on net.** Words whose per-occurrence excess rose outweigh those
  whose excess fell (112% of the positive score), led by *language, sign, dataset, learning,
  hand, app, accuracy, gesture, deep, network*: the machine-learning vocabulary that entered
  HCI in the 2020s.
- **The fall of E(philosophy → HCI) is mostly a change in HCI's word mix, not in philosophy's
  skill at HCI's words.** Splitting the fall into a mix effect (1990s per-word rates applied to
  the 2020s word frequencies) and a rate effect (the rest), per seed: fall 0.071, 0.103, 0.066,
  0.057, 0.097 bits; mix effect 0.039, 0.064, 0.068, 0.086, 0.091; rate effect 0.032, 0.039,
  −0.001, −0.029, 0.005. The mix effect dominates in every seed; the rate effect is positive in
  two, about zero in two, negative in one.
- **The mix effect is HCI shedding its 1990s interface jargon.** The words whose falling share
  lowered the excess most (seed 1): *interface, user_interface, design, computer, system, user,
  software, task, designer, information, describe, graphical, display, technique, tool,
  virtual_environment, graphic, menu, device, input, development, operator, database, hci*.
  Words whose rising share raised it: *gesture, research, immersive, participant, sign,
  virtual_reality, ux, metaverse, study, accuracy, digital, technology, vr*.

### Corrected reading

The reading above, "philosophy acquired HCI's objects", is at best the smaller half. The
larger and more consistent half is that **HCI's own vocabulary moved away from the
GUI-engineering lexicon of the 1990s** (menus, displays, graphical interfaces, software,
operators, databases), which philosophy's model finds most alien, towards a more generic
empirical register (*research, study, participant*) and the vocabulary of experience and
technology; philosophy's model finds the result less foreign without having learned much. This
is still consistent with every scored result: the change is spread across HCI's abstracts (G1,
G3), it is one-sided because philosophy's mix did not move (F2, G5), and philosophy's model did
not improve on computer science or AI (H1), whose jargon HCI partly adopted (*dataset, accuracy,
deep, network*), which is also why the per-word rates got worse on net.

Two things follow, and neither is scored yet. If the mechanism is HCI's mix, then *every*
field's model should predict HCI's 2020s abstracts better than its 1990s ones, with the same
mix effect dominating; the corrected table already holds those cells, uninspected. And the
earlier reading's test, HCI's object nouns against computing's method nouns in philosophy's
cells, is now secondary. The next round, if run, is the first of these.

---

## Round J: is it HCI's mix? (committed 2026-11-09, before the cells below were inspected)

The exploratory decomposition after round G attributed most of the fall in
E(philosophy → HCI) to HCI's word mix moving away from its 1990s interface jargon. If that is
the mechanism, it is not about philosophy at all: every field's model should predict HCI's
2020s abstracts better than its 1990s ones, and the same mix effect should dominate for them.
The corrected table (`data/drift/cross_perplexity_corrected.parquet`, lemmatised corpus, seeds
1–5) holds E(X → HCI) for every other field X; none of those cells has been inspected beyond
philosophy's. Fields converged on average between the 1980s and the 2010s (D5, L9), so a fall
alone is not enough; the fall on HCI must stand out against each model's falls on other texts.

Definitions: the nine *other* models are AI, cognitive psychology, communication, computer
science, history and philosophy of science, linguistics, neuroscience, psychology and
sociology. "Mean" is the mean over the five seeds. The *fall* of E(X → Y) is its 1990s value
minus its 2020s value. The mix and rate effects are computed exactly as in the exploratory
decomposition (1990s per-word rates at the 2020s word frequencies; words with no 1990s rate
keep their 2020s rate).

**J1 — Every field's model gains on HCI.** For at least 8 of the 9 other models, mean
E(X → HCI) is lower in the 2020s than in the 1990s. *Falsified* if fewer than 8.

**J2 — In every seed.** For at least 6 of the 9 other models, E(X → HCI) is lower in the 2020s
than in the 1990s in all 5 seeds. *Falsified* if fewer than 6.

**J3 — The mix effect dominates for other models too.** For psychology's model and for
sociology's model (chosen now: neither computing nor philosophy), the mix effect exceeds the
rate effect in at least 4 of 5 seeds each. *Falsified* if either falls short.

**J4 — HCI's fall stands out.** For at least 7 of the 9 other models, the fall of mean
E(X → HCI) exceeds the median fall of mean E(X → Y) over the other text fields Y (Y ≠ X,
Y ≠ HCI) that qualify in both decades. *Falsified* if fewer than 7.

Reading rule: J1, J2 and J4 passing means the effect was HCI's text changing, and the
"philosophy → HCI" result of D4 and L8 is one instance of a general gain on HCI; J3 passing
means the change is the word mix. J4 failing with J1 passing means a general convergence that
HCI merely shares. Scored by `scripts/score_mix.py`, committed before it is run.
