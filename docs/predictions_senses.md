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

### Round C results (2026-11-10, `scripts/score_senses.py`)

Scored as written. For each encoder 2 passed and 3 falsified, the same predictions under both.

The first run stopped before any scoring: the SciBERT *intelligence* run met a mostly Chinese
abstract in AI's 2020s cell, which has no spaces between words, so its 100-word context ran
past the encoder's 512 tokens and the occurrence was cut off. An occurrence past the limit is
now encoded again from 200 characters on either side (commit `39c7f97`); every occurrence that
fits is encoded as before, so the SciBERT *attention* table, finished before the fix and never
past the limit, was kept. The three remaining runs were made with the fix; the predictions and
the scorer were not changed.

| | SciBERT | ModernBERT |
|---|---|---|
| C1 | **Falsified.** 144–169 marked uses per seed, 0.99 of them in one sense; its share of AI's occurrences 0.147 in the 1990s (above 0.05), 0.580–0.622 in the 2020s | **Falsified.** 0.99 of marked uses in one sense; its share 0.136–0.141 in the 1990s, 0.564–0.614 in the 2020s |
| C2 | **Falsified.** The mechanism sense holds 0.366 of psychology's 1990s occurrences and 0.374–0.402 of its 2020s ones (above 0.05). Psychology's change in *attention* 0.0005, the controls' median 0.0147 | **Falsified.** The mechanism sense holds 0.327–0.329 and 0.364–0.392 of psychology's occurrences. Change 0.0018, the controls' median 0.0114 |
| C3 | **Passed.** AI's change in *attention* 0.164, larger than all 30 controls' (largest: *report* 0.083) | **Passed.** 0.165, larger than all 30 (largest: *increasing* 0.066) |
| C4 | **Passed.** The AI sense rises by 0.554 in communication, 0.385 in history and philosophy of science, 0.535 in sociology, 0.376 in philosophy, each present | **Passed.** 0.505, 0.340, 0.546, 0.395, each present |
| C5 | **Falsified.** Cognitive psychology's rise +0.292 (sd 0.000), present | **Falsified.** +0.281 (sd 0.003), present |

### Reading

Neither encoder passed C1, so by the reading rule the sense instrument is not used on open
questions, and a redesign is calibrated on *attention* again, in a new round, first.

C1 and C2 failed on one cause. Both encoders found two senses of *attention* in every seed,
and the two are not the ones the predictions named. One is attention as something that
operates: the transformer mechanism, but also cognitive selective attention, joint attention
and attention deficit disorder. The other is the stock phrase of academic prose (*particular
attention is given*, *has received less attention*). The marked uses fall almost entirely into
the first sense (0.99), and its share of AI's occurrences rises from about 0.14 to about 0.59,
but it already held AI's 1990s cognitive models of attention and a third of psychology's
occurrences. The instrument found the change in AI and found psychology's distribution
unchanged; it did not separate the mechanism from cognitive attention, which C1 and C2 required.
C3 passed under both encoders by a wide margin: AI's change in *attention* is about twice the
largest of 30 random controls'.

C4 passed: the AI sense of *intelligence* is "artificial intelligence" under both encoders,
and it rose in all four fields. C5 was a wrong bet, not a failure of the instrument: cognitive
psychology's 2020s abstracts write about artificial intelligence too (see below).

### Exploratory, after scoring: what the senses are, and what limits the instrument

Not predicted; read from the tables and the scorer's examples.

- **The number of senses is the limit.** The silhouette chose two senses for every *attention*
  and *intelligence* run, and in about two-thirds of all word and seed runs (81 of 155 under
  SciBERT's *attention* design, 110 of 165 under ModernBERT's runs). It favours the coarsest
  split. A redesign should find senses at a finer grain: split each sense again, or fix a
  larger number in advance and merge senses whose examples read the same.
- **The encoders agree.** Same numbers of senses, the same splits, shares within a few
  hundredths. The choice of encoder is not what limits the instrument.
- **Cognitive psychology's *intelligence*.** Under SciBERT, 96 of cognitive psychology's 289
  occurrences in the 2020s fall in the AI sense, and every one of them is "artificial
  intelligence": ChatGPT's creativity measured against people's, explainable AI, generative
  AI in creative work. Some are mislabelled engineering (microsleep detection in vehicles), a
  reminder that OpenAlex's field labels are noisy.
- **Small cells make the seeds nearly identical.** Most non-AI *intelligence* cells hold fewer
  than 500 occurrences, so every seed draws all of them and only the clustering's start
  differs; standard deviations near zero make the presence rule almost automatic there. The
  redesign should draw occurrences with replacement, so seeds vary in small cells too.

## Round C′: a finer instrument, tested on a known answer it has not seen (committed 2026-11-10, before any *hallucination* occurrence was encoded)

Round C's reading rule requires a redesign calibrated on *attention* again before the sense
instrument is used on open questions. *attention* is no longer unseen: the redesign was chosen
on its round C data, so passing on it again shows only that the fix does what it was chosen to
do. The test is a second known answer the instrument has not met. *hallucination* in
psychology and neuroscience is a perception without a stimulus; in AI's 2020s abstracts it is
a language model's fabricated output, a sense that did not exist before about 2020.

### Development on *attention* (round C's data, after scoring)

Re-encoding round C's seed-1 *attention* sample with SciBERT and clustering it into a fixed
number of senses k, the senses holding 80% of the marked uses (taken greedily, largest first)
hold this share of each cell's occurrences:

| k | AI 1990s | AI 2020s | Psychology 1990s | Psychology 2020s |
|---|---|---|---|---|
| 2–4 | 0.147 | 0.586–0.588 | 0.303–0.370 | 0.322–0.384 |
| 6–16 | 0.027–0.033 | 0.554–0.558 | 0.004–0.014 | 0.012–0.022 |
| 20 | 0.027 | 0.396 | 0.014 | 0.026 |

From six senses up, the attention mechanism separates from cognitive attention, and the
result holds from 6 to 16. The redesign fixes k at 8, the middle of that range, instead of
choosing it by silhouette, which chose two in round C. A sense is then a fine-grained usage
type, and a meaning such as "the attention mechanism" can span several of them, so a meaning
is measured as a *sense set*: the fewest senses that hold 80% of its marked uses.

### Method (fixed before any *hallucination* occurrence is encoded)

As round C, with these changes:

- **Senses.** k-means with k = 8 for every word, on length-normalised vectors.
- **Encoder.** SciBERT only. In round C the two encoders agreed to within a few hundredths on
  every measure.
- **Sampling with replacement.** For each word, cell and seed (1–5), as many occurrences as
  the cell has, up to 500, drawn with replacement from the abstracts containing the word, so
  the seeds differ in cells under the cap too.
- **Quality rules.** Abstracts with more than 20% of their letters outside the Latin script
  are skipped, and URLs and DOIs are removed before the context is taken. A word matches with
  its regular plural (*hallucination, hallucinations*).
- **Nested fields.** Two cells are compared as different communities only if neither field
  contains the other in OpenAlex's hierarchy (AI and HCI lie inside computer science,
  cognitive psychology inside psychology). Computer science is therefore left out of this
  round: AI carries the new sense.
- **Runs.** *attention* on AI's and psychology's 1990s and 2020s grid cells, as in round C.
  *hallucination* on the word-targeted cells (`data/raw/openalex_words/hallucination/`) of AI
  in the 2020s and of psychology and neuroscience in the 2000s and the 2020s (AI before the
  2020s has 1 and 16 works).
- **Marked uses.** *attention*: round C's list. *hallucination*, marked as the language-model
  sense when the context contains one of: *language model, LLM, ChatGPT, GPT, generative AI,
  chatbot, factual, fabricat*. The marks identify the sense; the clustering never sees them.
  The *mechanism set* and the *language-model set*, per seed, are the fewest senses that hold
  80% of the marked uses among AI's 2020s occurrences, taken greedily by marked count.
- **Sizes.** A cell with fewer than 40 occurrences makes the predictions that use it
  untestable, and so do fewer than 20 marked uses in AI's 2020s cell.
- Already seen: round C's tables and the development above (*attention*, seed 1, SciBERT);
  the *hallucination* cells' sizes, how many of their abstracts contain the exact word, and
  the label-noise shares (`scripts/measure_label_noise.py`). Not computed: any encoding of a
  *hallucination* occurrence.

### Predictions

**C′1 — The fix works where it was chosen.** In every seed, the mechanism set holds at most 5%
of AI's 1990s occurrences, at least 30% of AI's 2020s occurrences, and less than 5% of
psychology's occurrences in each decade. *Falsified* if any fails in any seed.

**C′2 — The finer instrument finds a sense it has not seen.** In every seed, the language-model
set holds at least 50% of AI's 2020s occurrences of *hallucination* and less than 5% of
psychology's and of neuroscience's 2000s occurrences. *Falsified* if any fails in any seed.

**C′3 — The perceptual sense still dominates where it lives.** In psychology and in
neuroscience, the language-model set holds less than 10% of the 2020s occurrences (mean over
seeds). *Falsified* if either holds 10% or more.

**C′4 — But the new sense has reached psychology.** The language-model set's share of
psychology's occurrences rises from the 2000s to the 2020s, present. *Falsified* if absent.
This one is not a known answer; it is the round's first open question, asked only because the
cell already exists.

Reading rule:

- C′1 and C′2 passing: the redesigned instrument is used on round D's open questions, with
  its method unchanged.
- C′2 failing: the fix was fitted to *attention*. The sense instrument is not used on open
  questions, and the project records sense-level drift as not achieved with this design.
- C′1 failing: the fix does not hold even where it was chosen, and the instrument is not used
  on open questions either.
- C′3 and C′4 describe *hallucination*; they do not decide the instrument's use.
- Every sense set is printed with its top context words and nearest examples (seed 1).

Scored by `scripts/score_senses_finer.py`, committed before the runs.

### Round C′ results (2026-11-10, `scripts/score_senses_finer.py`)

Scored as written. 2 passed, 2 falsified. The first launch of the run script failed before
encoding anything (a shell variable named `options`, which zsh reserves); it was renamed and the
runs were made once. *attention* had 1,681 occurrences per seed and *hallucination* 2,323,
eight senses each.

| | Verdict | Numbers |
|---|---|---|
| C′1 | **Passed** | One mechanism sense in every seed (159–183 marked uses); its share of AI's 1990s occurrences 0.005–0.048, of AI's 2020s 0.550–0.608, of psychology's 1990s 0.000–0.008 and 2020s 0.006–0.020 |
| C′2 | **Falsified** | The language-model set holds 0.810–0.882 of AI's 2020s occurrences in every seed. Psychology's and neuroscience's 2000s shares are 0.003–0.028 and 0.002–0.006 in seeds 2–4, but 0.115 and 0.058 in seed 1, 0.211 and 0.150 in seed 5 |
| C′3 | **Falsified** | The language-model set's mean share of psychology's 2020s occurrences 0.137 (seeds 0.038–0.266), neuroscience's 0.060 |
| C′4 | **Passed** | Its share of psychology's occurrences rises from the 2000s to the 2020s by +0.062 (sd 0.020), positive in every seed (+0.035 to +0.089), present |

### Reading

C′2 failed, so by the reading rule the fix is taken to have been fitted to *attention*: the
sense instrument is not used on open questions, and sense-level drift is recorded as not
achieved with this design. C′1 shows the finer instrument separating the attention mechanism
from cognitive attention in every seed; C′3 and C′4 describe *hallucination* and decide
nothing about the instrument.

### Exploratory, after scoring: what failed in seeds 1 and 5

Not predicted; read from the table, sense by sense.

In every seed the instrument found senses that are the language-model meaning and nothing
else: together they hold 0.66–0.85 of AI's 2020s occurrences and 0.000–0.012 of psychology's
2000s ones, with context words such as *retrieval-augmented, reasoning, prompting* and examples
about language models prone to hallucination. The perceptual senses separated too (auditory
verbal hallucinations, sleep paralysis, psychoanalytic readings). In seeds 1 and 5 one further
sense mixed the two: it held 15–17% of AI's 2020s occurrences and about 12% of the marked uses,
but also 10–22% of psychology's 2000s occurrences. The sense-set rule (the fewest senses holding
80% of the marked uses) needed it to reach 80%, and took it. In seeds 2–4 the pure senses
reached 80% alone. The failure is in the rule that groups senses into a meaning, which admits a
sense without regard to what else it holds, not in the number of senses.

## Round C″: the sense-set rule repaired, tested on a third known answer (committed 2026-11-10, before any *prompt* occurrence was encoded)

Round C′ located its failure in the rule that groups senses into a meaning. This round repairs
that rule on the data already seen and tests it on *prompt*, which no sense run has touched. In
psychology a prompt is a cue or a prompting intervention; in AI's 2020s abstracts it is the
instruction given to a language model, a sense that spread from about 2021. **This is the last
calibration round: if it fails, sense-level drift ends here.**

### Development on round C′'s tables (seen data, no new encoding)

The *meaning set* is now every sense whose share of marked uses, counted over all of its
occurrences in every cell, is at least half that of the sense with the highest share. A mixed
sense holds many unmarked occurrences from other cells, so its share is low whatever it holds in
AI. On round C′'s tables:

| | Marked share of senses that join | Of senses left out | Meaning set's share of AI's 2020s | Of the reference cells |
|---|---|---|---|---|
| *attention* | 0.54–0.58 (one sense) | at most 0.02 | 0.550–0.608 | AI 1990s 0.005–0.048, psychology at most 0.020 |
| *hallucination* | 0.44–0.93 (two senses) | at most 0.15 | 0.662–0.882 | psychology 2000s at most 0.028, neuroscience 2000s at most 0.008 |

With this rule both words meet round C′'s C′1 and C′2 thresholds in every seed. That is a
property of the rule on the data it was chosen on, not a result.

### Method (fixed before any *prompt* occurrence is encoded)

Round C′'s method (SciBERT, eight senses, sampling with replacement, plurals, the quality rules,
the nested-field rule), with the meaning set above in place of round C′'s sense sets.

- **Run.** *prompt* on its word-targeted cells (`data/raw/openalex_words/prompt/`): AI and
  psychology in the 2000s and the 2020s. Abstracts containing *prompt* or *prompts*: AI 298 and
  756, psychology 466 and 554. AI and psychology are not nested.
- **Marked uses**, as the language-model sense: the context contains one of *language model,
  large language, LLM, ChatGPT, GPT, generative AI, chatbot, few-shot, zero-shot, in-context,
  prompt engineering, prompt tuning, prompt learning*.
- **Sizes.** A cell with fewer than 40 occurrences, or fewer than 20 marked uses in AI's 2020s
  cell, makes the predictions untestable.
- Already seen: the four cells' sizes and how many abstracts contain the word. Not computed:
  any encoding of a *prompt* occurrence.

### Predictions

**C″1 — The repaired instrument finds a third known sense.** In every seed, the meaning set
holds at least 50% of AI's 2020s occurrences of *prompt*, at most 5% of AI's 2000s occurrences
and less than 5% of psychology's 2000s occurrences. *Falsified* if any fails in any seed.

**C″2 — The language-model prompt has reached psychology.** The meaning set's share of
psychology's occurrences rises from the 2000s to the 2020s, present. *Falsified* if absent. Not a
known answer: an open question asked because the cells exist.

Reading rule:

- **C″1 passes:** the instrument is used on round D's open questions with this method unchanged
  (round C′'s method and this meaning set). Round D's predictions are written before any of its
  words is encoded, and each finding is also computed without the abstracts flagged by
  `scripts/measure_label_noise.py`.
- **C″1 fails:** sense-level drift ends. Rounds C, C′ and C″ stand as scored, every sense result
  is reported as exploratory only, round D is not run, and the README says the instrument did
  not pass calibration.
- C″2 describes *prompt*; it does not decide the instrument's use.
- The meaning set is printed with its top context words and its nearest distinct examples
  (seed 1).

Scored by `scripts/score_senses_final.py`, committed before the run.
