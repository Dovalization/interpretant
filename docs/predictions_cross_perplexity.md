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

### Round J results (2026-11-09, `scripts/score_mix.py`)

Scored as written. 3 passed, 1 falsified.

| | Verdict | Numbers |
|---|---|---|
| J1 | **Falsified** | 7 of 9 models gain on HCI (needed 8). Falls of mean E(X → HCI), 1990s → 2020s: neuroscience 0.640, sociology 0.256, psychology 0.252, communication 0.110, cognitive psychology 0.082, linguistics 0.048, history and philosophy of science 0.015; **AI −0.033, computer science −0.004** |
| J2 | **Passed** | 6 of 9 fall in all 5 seeds: cognitive psychology, communication, linguistics, neuroscience, psychology, sociology |
| J3 | **Passed** | Mix effect exceeds rate effect in 5 of 5 seeds for both: psychology (fall, mix, rate) ≈ (0.25, 0.25, 0.01); sociology ≈ (0.26, 0.18, 0.07) |
| J4 | **Passed** | 7 of 9 models fall more on HCI than their median fall on other texts; the two exceptions are AI (−0.033 vs 0.079) and computer science (−0.004 vs 0.105) |

For comparison, philosophy's fall on HCI (L8) is 0.071–0.103 bits: among the smaller gains.

### Reading

The mechanism is confirmed and it has a shape the earlier rounds could not see. **HCI's
language moved out of computing's register and towards the behavioural and social sciences'.**
Every non-computing field's model predicts HCI's 2020s abstracts better than its 1990s ones, in
every seed, and by more than it gained on other texts; the gain is almost entirely a change in
HCI's word mix (for psychology's model the per-word rates did not change at all). The two
computing models, AI and computer science, are the exceptions J1 needed and did not get: they
did not gain on HCI, and relative to their gains elsewhere they lost ground. HCI stopped writing
GUI-engineering prose (*menu, display, graphical, user_interface, software, operator*) and
started writing *research, study, participant, experience*, which the social sciences' models
already knew and computing's did not.

Seen from here, the result of rounds D, F, H, L and G, "philosophy's model came to predict HCI
better", was true and misleading: philosophy was one of the smaller beneficiaries of a change
in HCI that favoured psychology, sociology and neuroscience far more. The one-sidedness (F2, G5)
follows too: HCI moved; nobody moved towards HCI. What the embedding grid showed for one word,
*embodiment* becoming HCI's own virtual-reality term, the language models show for HCI as a
whole, from the other side: the field's prose became a behavioural science's.

---

## Round K: the field's move, or the label's? (committed 2026-11-09, before any venue was
fetched)

Round J found that every non-computing field's model predicts HCI's 2020s abstracts better
than its 1990s ones, through HCI's own word mix. A field here is OpenAlex's primary-topic
subfield applied retrospectively; if what OpenAlex files under HCI broadened over those
decades, the move could be the label's scope rather than the field's prose. The check holds
the venues fixed.

### Method (fixed before any venue was fetched)

- **Venue lookup**: for every work in the HCI 1990s and 2020s cells, its
  `primary_location.source` id and name from OpenAlex, saved to
  `data/raw/openalex/hci/sources_<decade>.jsonl`.
- **Shared venues**: the sources with at least 100 cleaned documents in *both* decades' HCI
  cells. The **venue-stable cell** HCI_v of a decade is the cleaned HCI documents whose source
  is a shared venue; works without a source are outside it.
- **Models**: the corrected design (lemmatised text, shared abstracts removed across the
  decade's cells, seeds 1–5), except that every model in this round trains on **100,000
  tokens** and every text holds out **500 abstracts**, because HCI_v is smaller than a cell.
  HCI_v and the whole HCI cell (HCI_all) are scored in the same run under the same settings.
  HCI_v qualifies only if its pool after the held-out split has at least 100,000 tokens in
  both decades; otherwise the round is reported as untestable.
- Mix and rate effects as in rounds G and J.

### Predictions

**K1 — The fall survives inside fixed venues.** E(psychology → HCI_v) and
E(sociology → HCI_v) are lower in the 2020s than in the 1990s in all 5 seeds. *Falsified* if
either fails in any seed.

**K2 — And it is still the word mix.** For both models, the mix effect exceeds the rate effect
on HCI_v in at least 4 of 5 seeds. *Falsified* otherwise.

**K3 — Most of the effect is the field's, not the label's.** For both models, the mean fall on
HCI_v is at least half the mean fall on HCI_all measured in the same run. *Falsified* if
either ratio is under 0.5.

**K4 — Computing still does not gain.** The mean fall of E(AI → HCI_v) is less than a third of
the mean fall of E(psychology → HCI_v). *Falsified* otherwise.

Reported, not scored: the shared venues and their document counts per decade; the share of
each decade's cleaned HCI cell that lies in shared venues; the ten sources that grew most as a
share of the cell.

**Reading rule.** K1–K3 passing means round J's finding is the field's: HCI's prose moved even
inside the venues it always had. K3 failing with K1 passing means both happened, the field
moved and the label broadened, in the proportion K3 reports. K1 failing means round J measured
OpenAlex's classification, and the README's headline is withdrawn to the embedding finding.
Scored by `scripts/score_venue.py`, committed before it is run.

### Round K results (2026-11-09, `scripts/score_venue.py`)

**Untestable as written.** With the rule of at least 100 cleaned documents in both decades,
the shared venues are CHI (305 / 218 documents) and CHI Extended Abstracts (258 / 155) only;
HCI_v is 563 documents in the 1990s and 373 in the 2020s, and the 1990s training pool after
holding out 500 abstracts is 5,913 tokens. No excess was computed.

Reported numbers: the 1990s cleaned HCI cell (7,294 documents) spans 915 distinct sources,
with 1,271 documents lacking one; the 2020s cell (9,665 documents) spans 3,601 sources, 177
without one. The shared-venue share of the cell fell from 7.7% to 3.9%. The sources that grew
most as a share of the cell: IEEE VR workshops (VRW), Zenodo, AHFE International, Routledge
eBooks, Proceedings of the ACM on Human-Computer Interaction, *Sensors*, IEEE TVCG, Designing
Interactive Systems, IEEE Access, *Frontiers in Virtual Reality*. The label OpenAlex applies as
HCI is far more diffuse in the 2020s than in the 1990s, whatever the prose did.

---

## Round K′: the same test at feasible sizes (committed 2026-11-09, before any excess was
computed)

Identical to round K except for three sizes, chosen from the pool sizes above (inputs, not
evidence): shared venues need **at least 20** cleaned documents in both decades (9 venues:
CHI, CHI EA, UIST, VRST, IJHCI, HFES proceedings, CRC Press eBooks and two more, 1,471 / 856
documents); every text holds out **150** abstracts; every model trains on **90,000 tokens**
(HCI_v's smallest pool over seeds is 166,316 / 101,999 tokens). K1–K4 and the reading rule
are as in round K. Scored by `scripts/score_venue.py` with these constants.

### Round K′ results (2026-11-09, `scripts/score_venue.py`)

Scored as written. 3 passed, 1 falsified.

| | Verdict | Numbers (seeds 1–5) |
|---|---|---|
| K1 | **Passed** | Falls on HCI_v: psychology 0.054, 0.163, 0.109, 0.025, 0.108; sociology 0.243, 0.346, 0.290, 0.322, 0.231 |
| K2 | **Passed** | Mix > rate in 5 of 5 for both; psychology (fall, mix, rate) ≈ (0.09, 0.17, −0.07), sociology ≈ (0.29, 0.22, 0.06) |
| K3 | **Falsified** | Mean fall on HCI_v over mean fall on HCI_all: psychology 0.092 / 0.184, just under 0.50; sociology 0.286 / 0.234 = 1.22 |
| K4 | **Passed** | AI's mean fall on HCI_v −0.053 against psychology's 0.092 |

Shared venues at the K′ rule: CHI (305 / 218 documents), CHI EA (258 / 155), HFES proceedings
(440 / 35), *Behaviour and Information Technology* (134 / 23), UIST (100 / 52), IJHCI
(76 / 99), CRC Press eBooks (97 / 115), VRST (33 / 52), DIS (28 / 107). They hold 20.2% of the
1990s cleaned cell and 8.9% of the 2020s one.

### Reading

By the rule written before round K: **both happened.** The field moved: inside the venues HCI
had in both decades, psychology's and sociology's models predict its 2020s abstracts better in
every seed, through the word mix, while AI's model does not. And the label broadened: the
OpenAlex HCI cell went from 915 sources to 3,601, the venue-stable share fell from a fifth to
under a tenth, and the sources that grew are virtual-reality workshops, repositories, publisher
eBook series and sensor journals. How much of round J's measured size each accounts for depends
on the model doing the measuring: for sociology's the effect is larger inside fixed venues than
on the whole cell (nothing of it is the label); for psychology's, half of it. K3 failed on
psychology's ratio by less than a thousandth of a bit, and the verdict stands as written.

So the headline keeps its substance with a qualification: HCI's prose moved out of computing's
register into the behavioural and social sciences' inside its own venues, and OpenAlex's HCI
label also widened to take in work from adjacent fields, which made the whole-cell measurement
larger than the field's own move. Both are findings about the corpus; the second is also a
warning about any claim that rests on a subfield label held fixed across seventy years.

## Round S: can a classifier tell the fields apart? (committed 2026-11-09, before any classifier was trained)

Every result about HCI so far comes from one family of instrument: bigram and unigram language
models scoring each other's text. Round S asks the same questions of a method that shares
nothing with them but the corpus: a classifier that guesses an abstract's field from its words.
If two fields' language converges, the classifier finds them harder to tell apart; if HCI's
language moved into the behavioural and social sciences' register (J, K′), HCI's abstracts get
harder to tell from theirs and no harder to tell from computing's, and a classifier trained on
the 1990s sends more of HCI's 2020s abstracts to those fields. A classifier is symmetric
between two fields, so the direction of the move is tested only through S4's cross-decade
design. Nothing below has been computed; the classifier code does not exist yet.

### Method (fixed before any classifier is trained)

- **Corpus.** The lemmatised, phrased main grid (`data/phrased/openalex/texts`). In each decade,
  abstracts that appear in more than one cell are removed first, as in the corrected
  cross-perplexity design. Decades 1980s–2020s, where all eleven fields keep at least 2,000
  abstracts.
- **Samples.** For each seed 1–5 and each decade, 2,000 abstracts per field, drawn at random.
- **Features and model.** TF-IDF over the abstracts' tokens (unigrams, sublinear term frequency,
  words in at least 2 training abstracts), then logistic regression (L2, C = 1). Fitted on the
  training part only.
- **Pairwise separability.** For every pair of fields in a decade, the mean accuracy of stratified
  5-fold cross-validation on the two samples (chance is 0.5).
- **Multiclass assignments.** For each decade, one classifier over all eleven fields, trained on
  1,600 abstracts per field and applied to the other 400 of each field; for each field, the share
  of its held-out abstracts assigned to every field.
- **Cross-decade assignments.** The 1990s multiclass classifier is also applied to 400 abstracts
  from each field's 2020s sample.
- **Groups.** The *behavioural and social* fields are psychology, cognitive psychology,
  neuroscience, sociology and communication; the *computing* fields are computer science and AI.
  "Mean" is the mean over the five seeds.

### Predictions

**S0 — The classifier sees the fields.** In both the 1990s and the 2020s, every pair's mean
accuracy is at least 0.75, and the three parent–subfield pairs (computer science–AI, computer
science–HCI, psychology–cognitive psychology) are among the five least separable pairs of the
decade. *Falsified* if any pair falls below 0.75 or any parent–subfield pair is outside the
bottom five in either decade.

**S1 — HCI becomes harder to tell from the behavioural and social sciences.** For at least 4 of
the 5 behavioural and social fields X, the HCI–X accuracy is lower in the 2020s than in the
1990s in every seed comparison (all five 2020s values below all five 1990s values). *Falsified*
if fewer than 4.

**S2 — but not from computing.** For both computer science and AI, the mean HCI–X accuracy in the
2020s is at least the 1990s mean minus 0.01. *Falsified* if either falls by more than 0.01.

**S3 — Where each decade's classifier sends HCI's abstracts.** Comparing the 1990s and the 2020s
multiclass classifiers, the share of HCI's held-out abstracts assigned to the behavioural and
social fields rises in at least 4 of 5 seeds, and the share assigned to the computing fields
falls in at least 4 of 5 seeds. *Falsified* if either part fails.

**S4 — HCI moved; nobody moved towards HCI.** The 1990s classifier assigns a larger share of
HCI's 2020s abstracts than of its 1990s held-out abstracts to the behavioural and social fields,
in at least 4 of 5 seeds; and it assigns no larger share of psychology's 2020s abstracts to HCI
than of psychology's 1990s held-out abstracts, in at least 4 of 5 seeds, and the same for
sociology. *Falsified* if any of the three parts fails.

**S5 — Fields converged.** Over all field pairs, the mean pairwise accuracy is lower in the 2010s
than in the 1980s in all 5 seeds (the classifier's version of D5 and L9). *Falsified* otherwise.

Reading rule: S1–S4 passing replicates round J's finding with an independent instrument, and S4
gives the direction without the language models' asymmetry. S1 passing with S4 failing would
mean HCI and those fields converged without HCI being the one that moved. S0 failing would make
the classifier too blunt for the rest to mean much. Scored by `scripts/score_separability.py`,
committed before it is run.

### Round S results (2026-11-09, `scripts/score_separability.py`)

Scored as written. 1 passed, 5 falsified.

| | Verdict | Numbers |
|---|---|---|
| S0 | **Falsified** | Every pair is separable (lowest mean accuracy 0.848 in the 1990s, 0.836 in the 2020s), and computer science–AI and psychology–cognitive psychology are in the bottom five; computer science–HCI is not (0.929 and 0.932), so the second clause fails |
| S1 | **Falsified** | 3 of 5 (needed 4). Harder to tell apart in every seed: psychology 0.961–0.971 → 0.953–0.957, neuroscience 0.967–0.970 → 0.958–0.964, sociology 0.973–0.975 → 0.965–0.968; easier: cognitive psychology 0.947–0.955 → 0.956–0.962, communication 0.972–0.977 → 0.975–0.981 |
| S2 | **Passed** | HCI–computer science 0.929 → 0.932; HCI–AI 0.949 → 0.956 |
| S3 | **Falsified** | Share of HCI's held-out abstracts sent to the behavioural and social fields rises in 4 of 5 seeds (about 0.06 → 0.07), but the share sent to computing falls in only 2 of 5 |
| S4 | **Falsified** | HCI's part passes in 5 of 5 seeds (the 1990s classifier sends 0.055–0.070 of HCI's 1990s abstracts to the behavioural and social fields, 0.110–0.123 of its 2020s ones); the other two fail in every seed: psychology's 2020s abstracts go to HCI more often (0.003–0.013 → 0.018–0.028), and so do sociology's (0.000–0.007 → 0.010–0.022) |
| S5 | **Falsified** | Mean pairwise accuracy did not fall from the 1980s to the 2010s: 0.9488 → 0.9487 in seed 1, a small rise in seeds 2–5 (largest 0.9471 → 0.9503) |

### Reading

The classifier agrees with the language models on one thing and not on the rest. **It agrees
that HCI's prose moved away from its 1990s self towards the behavioural and social sciences**:
a classifier trained on the 1990s sends about twice as many of HCI's 2020s abstracts to those
fields as of its 1990s ones, in every seed, and HCI became harder to tell from psychology,
neuroscience and sociology while staying as easy to tell from computing (S2). But the effect is
small against a ceiling (pairwise accuracies move by about 0.01 around 0.95–0.97), it misses two
of the five fields, and it is **not one-sided**: psychology's and sociology's 2020s abstracts
also look more like 1990s HCI to the 1990s classifier (S4's second and third parts, in every
seed). Round J's "nobody moved towards HCI" was a statement about the language models'
asymmetry; the classifier does not reproduce it.

S5 contradicts D5 and L9 at face value: whatever convergence the language models measure (L9
showed it is a convergence of word mixes), it does not make fields harder for a classifier to
tell apart. The two readings can both hold: average word frequencies can draw together while
each field keeps enough marker words for a classifier to separate it at 95%. S0's surprise is
that HCI is not one of computer science's hardest neighbours: its vocabulary is distinctive
enough to keep it apart from the field it is filed under.

### Exploratory, after scoring: who else drifts under the 1990s classifier

Not predicted; read from `data/drift/separability_assignments.parquet` after the scorer ran.
Means over the five seeds.

- **Every field drifts.** The 1990s classifier recognises most fields' 2020s abstracts less well
  than their 1990s ones (computer science 0.796 → 0.621, neuroscience 0.841 → 0.634, AI 0.772 →
  0.624, HCI 0.865 → 0.748); only psychology, cognitive psychology and linguistics hold steady.
  Thirty years of vocabulary change blurs any fixed classifier.
- **Almost every field drifts towards HCI.** The share of a field's 2020s abstracts sent to HCI
  rises for ten of the eleven fields (computer science 0.046 → 0.073, neuroscience 0.005 →
  0.025, psychology 0.008 → 0.022, sociology 0.003 → 0.015; only cognitive psychology falls).
  The likeliest reading, untested here, is that technology vocabulary the 1990s classifier tied
  to HCI spread to every field by the 2020s.
- **HCI's 2020s abstracts leak in every direction.** The 1990s classifier sends 0.111 of them to
  AI and computer science together, about as much as to the five behavioural and social fields
  together (0.117), and up from 0.06 for each group in the 1990s.

Taken together, the classifier sees HCI's language spreading away from its 1990s core, partly
towards the behavioural and social sciences and partly towards computing, while technology
words spread into everyone else's abstracts. That is a weaker and more symmetric picture than
round J's, and a claim that HCI moved "out of computing's register" should be stated as the
language models' finding, not as one both instruments confirm. A next round could test the
technology-vocabulary reading directly by removing a fixed list of technology terms before
classifying.

## Round V: is the drift towards HCI technology vocabulary? (committed 2026-11-09, before any classifier ran without it)

Round S's exploratory note found that, under a classifier trained on the 1990s, ten of the
eleven fields' 2020s abstracts look more like 1990s HCI, and read that as technology vocabulary
(the words that marked HCI in the 1990s) spreading into every field. Round V tests that reading
by removing a fixed list of technology terms and repeating round S's classifiers. If the reading
is right, the drift towards HCI disappears without those words, while HCI's own move towards the
behavioural and social sciences, if it is more than technology words, survives.

### Method (fixed before any classifier runs without the terms)

- **Exactly round S's method and samples** (same seeds, same 2,000 abstracts per field), for the
  1990s and the 2020s only, with one change: before any classifier sees them, every token that
  is a listed term, or a phrase with a listed term as a component, is removed from every
  abstract.
- **The terms** are the 29 in `data/technology_terms.txt`, committed with these predictions:
  technology, technological, digital, online, internet, web, website, computer, computing,
  computational, software, hardware, device, mobile, smartphone, app, interface, user, algorithm,
  algorithmic, automation, automated, robot, robotic, virtual, platform, machine, ai, artificial.
  They name computing artefacts and the digital medium; general research words (*data, network,
  model, system*) and other fields' core words (*media*) stay in.
- Groups, shares and "mean" as in round S. A field's *drift towards HCI* is the share of its
  2020s held-out abstracts that the 1990s classifier sends to HCI, minus the share of its 1990s
  held-out abstracts it sends there.

### Predictions

**V0 — Still a working classifier.** Without the terms, every pair's mean accuracy is still at
least 0.75 in both the 1990s and the 2020s. *Falsified* if any pair falls below 0.75.

**V1 — Psychology and sociology stop drifting towards HCI.** Without the terms, the drift
towards HCI is zero or negative in at least 4 of 5 seeds for psychology, and the same for
sociology (round S's S4 parts two and three, which failed in every seed with the terms).
*Falsified* if either field falls short.

**V2 — The drift towards HCI shrinks across the board.** Without the terms, at most 5 of the 10
non-HCI fields have a positive mean drift towards HCI (9 of 10 had one in round S, all but
cognitive psychology). *Falsified* if 6 or more do.

Correction to round S's exploratory note: it says the share sent to HCI "rises for ten of the
eleven fields"; by its own table it rises for 9 of the 10 non-HCI fields (HCI's share of its own
abstracts fell, and cognitive psychology's share fell).

**V3 — HCI's move survives without the technology words.** Without the terms, the 1990s
classifier sends a larger share of HCI's 2020s held-out abstracts than of its 1990s ones to the
behavioural and social fields, in at least 4 of 5 seeds. *Falsified* otherwise.

Reading rule: V1 and V2 passing mean round S's symmetric picture was technology vocabulary
spreading into every field, and with V3 passing, HCI's move towards the behavioural and social
sciences is real and one-sided once those words are set aside: round S's S4 would have passed.
V1 or V2 failing means other fields really did come to write more like 1990s HCI, and the
one-sidedness of round J belongs to the language models alone. V3 failing means HCI's move, as
a classifier sees it, was technology vocabulary too. Scored by `scripts/score_technology_terms.py`,
committed before it is run.

### Round V results (2026-11-09, `scripts/score_technology_terms.py`)

Scored as written. 2 passed, 2 falsified.

| | Verdict | Numbers |
|---|---|---|
| V0 | **Passed** | Lowest mean pair accuracy without the terms: 0.848 in the 1990s (cognitive psychology–psychology), 0.826 in the 2020s (AI–computer science) |
| V1 | **Falsified** | Without the terms, the drift towards HCI is still positive in every seed: psychology +0.013 to +0.035, sociology +0.002 to +0.022 |
| V2 | **Falsified** | 9 of 10 non-HCI fields still drift towards HCI (needed at most 5); the largest are computer science +0.026, psychology +0.024 and history and philosophy of science +0.015; only cognitive psychology does not (−0.001) |
| V3 | **Passed** | Without the terms, the 1990s classifier sends more of HCI's 2020s abstracts than of its 1990s ones to the behavioural and social fields in 5 of 5 seeds: 0.055–0.082 → 0.120–0.158 |

### Reading

Round S's exploratory reading was wrong. **Technology vocabulary does not explain why other
fields' 2020s abstracts look more like 1990s HCI**: removing 29 technology terms changes the
drift towards HCI very little (psychology's mean drift is +0.024 without them, against +0.014
with them; sociology's +0.010 against +0.012). **HCI's own move towards the behavioural and
social sciences is not technology vocabulary either**: it survives the removal in every seed and
grows slightly. So the classifier's picture from round S stands, now without the easy
explanation: HCI moved a long way towards the behavioural and social sciences, and some of those
fields, psychology above all, moved a shorter way towards 1990s HCI. The movement is mutual and
lopsided in size (HCI's shares change by 0.06–0.10, psychology's by about 0.02), not one-sided.
The language models' asymmetry (round J) stays what it was: a statement about which models gained
on which texts, which a classifier does not reproduce.

### Exploratory, after scoring: where drifted abstracts land

Not predicted; read from both rounds' assignment tables after the scorer ran. For every text
field, the gain in the share of its abstracts the 1990s classifier sends to each other field
(2020s minus 1990s, means over five seeds).

- **HCI is not a general sink.** Summed over the other ten fields, the drifted share received is
  largest for psychology (0.26 without the terms), then sociology (0.19), AI (0.16) and computer
  science (0.13); HCI is fifth (0.11). Philosophy and history and philosophy of science lose
  share. 2020s prose of most fields lands, when it leaves home, in psychology's, sociology's and
  AI's registers.
- **Psychology's largest gain is towards HCI**, with and without the terms (+0.014 and +0.023),
  the only field whose top destination is HCI.
- **HCI's largest gain depends on the technology words**: towards AI with them (+0.038), towards
  psychology without them (+0.039). HCI's technology vocabulary pulls its 2020s abstracts towards
  AI; the rest of its prose pulls them towards psychology.

The pair that converged, by a classifier's account, is HCI and psychology, from both sides; what
round J saw as HCI leaving computing for the behavioural sciences is, here, HCI and psychology
meeting, with HCI travelling further.

## Round M: does every cell's model generalise? (committed 2026-11-09, before any held-out score exists)

A cell built from a few hundred abstracts can yield a model that has memorised its text: its
neighbour lists and its scores then describe those abstracts, not the field. The drift and
perplexity rounds guard against small cells with occurrence thresholds, noise floors and fixed
training budgets, but no round has measured, cell by cell, how well a model trained on a cell
predicts text from that cell it never saw. Round M does, and asks whether any robust finding
rests on a cell that fails.

### Method (fixed before any score is computed)

- **Cells.** Every cell of the lemmatised, phrased corpus: books, the main OpenAlex grid and
  the full cells (`data/phrased/<layer>/texts/<cell>.txt`).
- **Split.** A cell's stored documents (one per line) are shuffled with a fixed seed; 10% are
  held out (at least one), the rest are the training part, capped at 5,000,000 tokens (taken
  in shuffled order). Seeds 1–3; "mean" is the mean over them.
- **Model and measure.** A Kneser-Ney bigram model (`KneserNeyBigram`, vocabulary = training
  words with at least 5 occurrences) is fitted on the training part. Its *gap* is the
  cross-entropy of the held-out documents minus the cross-entropy of a random sample of the
  training documents of the same number of documents, in bits per token. A model that
  generalises has a small gap; one that memorised a small cell has a large one.
- **Flag.** A cell is *flagged* when its mean gap exceeds 1.5 times the median mean gap of the
  main-grid cells of the 2000s, 2010s and 2020s (33 cells of about 10,000 abstracts, the
  reference size).

### Predictions

**M0 — Size drives the gap.** Over all cells, the Spearman correlation between a cell's
training tokens and its mean gap is −0.8 or stronger. *Falsified* if weaker than −0.8.

**M1 — The check has teeth.** HCI's main-grid cells of the 1950s, 1960s and 1970s (18, 152 and
459 abstracts) and AI's of the 1950s (234) are all flagged. *Falsified* if any is not.

**M2 — No robust finding rests on a flagged cell.** None of these cells is flagged: philosophy's
full cells of the 1980s–2020s and HCI's of the 2000s–2020s (*embodiment*, *intentionality*,
*consciousness*); AI's main-grid cells of the 1980s, 2000s, 2010s and 2020s (*representation*);
and every main-grid cell of the 1980s, 1990s, 2010s and 2020s (the HCI rounds and the
convergence of fields). *Falsified* if any is flagged.

Reading rule: M0 and M1 passing mean the measure sees what it should; M2 passing means the
README's robust findings rest on cells that generalise. A flagged cell under M2 names a finding
whose evidence needs a second look, and every flagged cell is a cell whose neighbour lists the
explorer should mark. Scored by `scripts/score_heldout.py`, committed before it is run.

### Method amendment (2026-11-09, before any corpus score was computed)

The synthetic tests written for the scorer showed the committed measure fails on its own
terms: with the vocabulary restricted to training words with at least 5 occurrences, a small
cell maps most of its words to the unknown token, which makes both its training and held-out
text cheap, so its gap comes out *smaller* than a large cell's (on uniform synthetic text:
−0.40 bits for 20 documents, 0.75 for 4,000; on Zipf-distributed text: 0.28 for 20, 1.23 for
20,000). With every training word in the vocabulary, held-out words the model never saw cost
what they should and the gap falls steadily with size (uniform: 5.10, 4.20, 1.61, 0.75; Zipf:
4.41, 3.78, 2.65, 1.23). The measure is amended accordingly: **the vocabulary is every word of
the training part**. Nothing else changes, the predictions stand as written, and no corpus cell
had been scored when this was found.
