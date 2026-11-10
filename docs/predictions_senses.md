# Predictions: senses by field and decade

Pre-registered predictions for the sense-level rounds (ROADMAP item 8). As in
`predictions.md` and `predictions_cross_perplexity.md`, a round's predictions are committed
before any code that tests them is run on the corpus; results are appended below them, and
predictions and verdicts are never edited afterwards. Corrections go in new sections.

## Round C: does the instrument find senses whose history is known? (committed 2026-11-10, before any corpus occurrence was encoded)

The direction rounds taught that a measure with no external answer can only be tested against
another measure. This round tests the sense instrument on two changes whose history is known
before it is pointed at open questions. In AI, *attention* gained a new sense, the attention
mechanism of neural networks (Bahdanau et al. 2014; Vaswani et al. 2017); psychology's
*attention* kept its cognitive sense. And AI's sense of *intelligence* spread into other fields'
abstracts in the 2020s, with the arrival of generative AI. Two encoders are tested, and only
those that pass are used on open questions.

### Method (fixed before any corpus run)

- **Occurrences.** From the cased abstracts behind each cleaned OpenAlex cell, with abstracts
  found in more than one field's cell of the same decade removed: for each word, cell and seed
  (1–5), up to 500 abstracts that contain the word, in a seeded random order, and the first
  occurrence in each. The word matches in any case, also inside a hyphenated compound
  (*self-attention*). The encoder reads the occurrence's sentence, cut to 100 words around it
  when longer.
- **Encoders.** SciBERT (`allenai/scibert_scivocab_uncased`) and ModernBERT
  (`answerdotai/ModernBERT-base`). An occurrence's vector is the mean of the last four hidden
  layers over the subword tokens that overlap it.
- **Senses.** For each word, encoder and seed, every occurrence from every cell of the run is
  clustered together: k-means on length-normalised vectors for k from 2 to 10, keeping the k
  with the best cosine silhouette (on a seeded sample of at most 5,000). A cell's *sense
  distribution* is the share of its occurrences in each sense; the change between two cells is
  the Jensen-Shannon divergence of their distributions, in bits (0 to 1).
- **Runs.** `interpretant drift senses`, once per encoder for each design:
  - *attention* and the 30 control words in `data/control_words_senses.txt`, on AI's and
    psychology's 1990s and 2020s cells;
  - *intelligence* on the 2000s and 2020s cells of AI, communication, history and philosophy
    of science, sociology, philosophy and cognitive psychology.
- **Controls.** The 30 control words were drawn at random (seed 2026) from the 570 words in at
  least 100 abstracts in each of AI's and psychology's 1990s and 2020s cells, stopwords and
  words under four letters left out (`scripts/select_sense_controls.py`). They are a sample of
  typical words, not words assumed stable: some may have changed in AI too.
- **Marked uses.** An occurrence is *marked* as the attention mechanism when its context
  contains, in any case, one of: *self-attention, self attention, cross-attention,
  transformer, multi-head, multihead, attention mechanism, attention layer, attention
  weight, attention head, attention module*. The marks only identify the mechanism sense;
  the clustering never sees them. The *mechanism sense*, per encoder and seed, is the sense
  holding the most marked uses among AI's 2020s occurrences.
- **The AI sense of *intelligence*,** per encoder and seed, is the sense holding the largest
  share of AI's 2020s occurrences.
- **Present** is round N's rule: the same sign in every seed and a mean at least twice the
  standard deviation. Change measures are means over the five seeds.
- **Sizes.** A cell with fewer than 40 occurrences makes the predictions that use it
  untestable. C1 is untestable for an encoder if AI's 2020s cell has fewer than 20 marked
  uses.
- Already seen: the number of abstracts containing each candidate word per field and decade
  (in the processed texts), the control words' counts, and three hand-written sentences
  encoded to check that both encoders load. Not computed: any encoding of a corpus occurrence,
  any clustering.

### Predictions

Every prediction is scored separately for each encoder.

**C1 — The instrument isolates the new sense.** In every seed, the mechanism sense holds at
least 80% of the marked uses among AI's 2020s occurrences, holds at most 5% of AI's 1990s
occurrences, and holds at least 30% of AI's 2020s occurrences. *Falsified* if any of the three
fails in any seed.

**C2 — Psychology's *attention* did not change.** The mechanism sense holds less than 5% of
psychology's occurrences in both decades in every seed, and psychology's change in *attention*
from the 1990s to the 2020s is no larger than the median of the 30 control words' changes in
psychology. *Falsified* if either fails.

**C3 — AI's *attention* changed more than typical words.** AI's change in *attention* from the
1990s to the 2020s is larger than that of at least 27 of the 30 control words in AI.
*Falsified* otherwise.

**C4 — AI's *intelligence* spread into other fields.** In communication, history and philosophy
of science, sociology and philosophy, the AI sense's share of the field's occurrences rises
from the 2000s to the 2020s, present in each of the four. *Falsified* if absent in any.

**C5 — Not into psychology's *intelligence*.** In cognitive psychology, where *intelligence* is
mostly the psychometric kind, the AI sense's rise from the 2000s to the 2020s is not present.
*Falsified* if present.

Reading rule:

- An encoder that passes C1–C3 can find a sense appearing inside one field and is used on the
  open questions (round D). One that also passes C4 and C5 can find a sense moving between
  fields, and only such an encoder is used on round D's questions about borrowing.
- If both encoders qualify, a round D finding must hold under both; their disagreements are
  reported, not resolved.
- If neither passes C1, the sense instrument is not used on open questions. A redesign is
  calibrated on *attention* again, in a new round, before anything else.
- Each encoder's mechanism sense and AI sense of *intelligence* are printed with their top
  context words and nearest examples (seed 1), so the senses can be read, not just counted.

Scored by `scripts/score_senses.py`, committed before the runs.
