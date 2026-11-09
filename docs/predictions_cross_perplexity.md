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
