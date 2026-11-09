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
