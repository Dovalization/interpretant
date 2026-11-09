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
