# Diagnostics: where the problem lies

Checks from the diagnostic plan in `docs/assumptions.md`. Like predictions, each check's rules
are committed before anything it counts is looked at; results are appended below and not edited.

## Check 1a: are the known changes present in the cells? (committed 2026-11-10, before any marker was counted or any sampled usage was read)

Rounds 5 and X judge an instrument by whether it finds changes documented in a field's history.
That assumes the changes are present in the cells the instrument reads (assumption A5). This check
measures their presence directly, so a missed change can be put down to the instrument or to the
data.

### Method

- **Usages.** Exactly the usages round X's scorer reads: for each case, the word with its regular
  plural, one occurrence per abstract, at most 150 abstracts per cell, seed 1, round C′'s quality
  rules, abstracts in both cells of a transition removed (`scripts/score_pairs_known.py`).
- **Cases and the sense whose presence is measured.**

  | Case | Cells | Sense |
  |---|---|---|
  | *attention*, AI | 2010s, 2020s | the attention mechanism of neural networks |
  | *prompt*, AI (word-targeted cells) | 2000s, 2020s | an instruction or input given to a language model |
  | *platform*, communication | 2000s, 2010s | a digital or social-media platform |
  | *network*, AI | 1980s, 1990s | an artificial neural network |
  | *alignment*, AI | 2010s, 2020s | aligning an AI system with human intentions, preferences or values |
  | *attention*, AI (quiet) | 1990s, 2000s | the attention mechanism |
  | *attention*, psychology (quiet) | 2010s, 2020s | the attention mechanism |

- **Markers.** A usage's context is marked for the sense when it matches, as whole words and in
  any case, one of:
  - attention mechanism: *self-attention, self attention, cross-attention, transformer(s),
    multi-head, multihead, attention mechanism(s), attention layer(s), attention weight(s),
    attention head(s), attention module(s)* (round C's list);
  - prompt: *language model(s), large language, LLM(s), ChatGPT, GPT, generative AI, chatbot(s),
    few-shot, zero-shot, in-context, prompt engineering, prompt tuning, prompt learning* (round
    C″'s list);
  - platform: *social media, social network(s), social networking, Facebook, Twitter, YouTube,
    Instagram, TikTok, WeChat, Weibo, Reddit, digital platform(s), online platform(s),
    platform economy, platform companies, platformisation, platformization*;
  - neural network: *neural network(s), neural net(s), connectionist, connectionism,
    backpropagation, back-propagation, perceptron(s), hidden unit(s), hidden layer(s), Hopfield,
    Boltzmann*;
  - alignment: *language model(s), LLM(s), RLHF, human feedback, human preference(s), human
    values, value alignment, AI alignment, AI safety*.
- **Blind reading.** Twenty usages per cell, drawn at random (seed 2026) from those above, are
  pooled across all fourteen cells, shuffled, and labelled one by one without the cell, the
  marker match or the case's other usages in view: *yes* if the context uses the word in the
  case's sense, *no* otherwise, *unclear* if it cannot be told. The labels are made by Claude,
  recorded in `data/diagnostics/check_1a_labels.tsv`, and open to the author's correction; a
  correction is recorded as such.
- **Present.** A documented change is *present* in its cells when the marked share and the read
  share (yes over yes plus no) each rise by at least 0.10 from the earlier cell to the later. A
  quiet transition is *quiet* when the marked share changes by less than 0.05 and the read share
  by less than 0.10.

### What each outcome crosses out

- A documented change **present** but missed by an instrument: the instrument (or its bars),
  not the data, failed on that case.
- A documented change **absent**: the case is not a known answer for these cells; any verdict
  that used it (round 5, round X) says nothing about the instrument for that case, and the next
  step is the cells (A1, A4), not the instrument.
- The two shares **disagree** (one rises, one does not): the markers are unreliable for that
  sense, and the reading decides, reported as such.
- A quiet transition **not quiet**: something changed where nothing was documented; reported,
  and the transition is not used as a quiet case again.

Scored by `scripts/check_known_presence.py`, committed before it is run.

### Check 1a results (2026-11-10, `scripts/check_known_presence.py`)

Scored as written. All five documented changes are present; both quiet transitions are quiet.
The 280 labels were recorded and committed before the key was opened (commit 2f4eca4).

| Case | Marked share, earlier → later | Read share, earlier → later (yes / no / unclear) | Verdict |
|---|---|---|---|
| *attention*, AI, 2010s → 2020s | 0.13 → 0.31 | 0.25 → 0.60 (5/15/0, 12/8/0) | **Present** |
| *prompt*, AI, 2000s → 2020s | 0.01 → 0.51 | 0.00 → 0.95 (0/20/0, 18/1/1) | **Present** |
| *platform*, communication, 2000s → 2010s | 0.08 → 0.47 | 0.62 → 0.89 (10/6/4, 17/2/1) | **Present** |
| *network*, AI, 1980s → 1990s | 0.45 → 0.71 | 0.65 → 0.95 (13/7/0, 19/1/0) | **Present** |
| *alignment*, AI, 2010s → 2020s | 0.01 → 0.13 | 0.00 → 0.15 (0/20/0, 3/17/0) | **Present** |
| *attention*, AI, 1990s → 2000s | 0.01 → 0.01 | 0.00 → 0.00 (0/19/1, 0/20/0) | **Quiet** |
| *attention*, psychology, 2010s → 2020s | 0.00 → 0.00 | 0.00 → 0.00 (0/20/0, 0/20/0) | **Quiet** |

### Reading

Assumption A5 is crossed out for these cases: every documented change is visibly in the cells
round X reads, and nothing changed in the quiet ones. An instrument that misses one of these
changes has failed on that case, not the data. That applies to round 5 too, whose cells are the
same field and decade samples: the grid missed *attention*, *platform* and *network*, all present.

Not predicted, read from the table: the changes differ greatly in size. *prompt* changed almost
completely. *network* and *platform* changed from an already high start: about two thirds of the
1980s *network* uses and of the 2000s *platform* uses are already the later sense, so the decade
cells catch the end of each change, not the change itself (A4). *alignment*, the one case round 5's
grid flagged, is the smallest change: only about one in seven of its 2020s uses is about aligning
AI systems. The grid's single hit is therefore unlikely to be the documented change, and more
likely a broader shift in *alignment*'s contexts.

## Check 2b: does XL-LEXEME read this corpus's sentences as a person does? (committed 2026-11-10, before the gold pairs were drawn or judged)

The usage-pair instrument was validated on DWUG EN, sentences from historical fiction and
news (round U). This corpus is research abstracts, a formulaic genre (assumption A6). This check
compares XL-LEXEME's judgement of a word's meaning in two sentences with the author's judgement
of the same pairs, on this corpus.

### Method

- **Pairs.** From the usages round X reads (`scripts/score_pairs_known.py` sampling), for six
  words: *attention* (AI, 2010s and 2020s), *prompt* (AI, 2000s and 2020s, word-targeted cells),
  *platform* (communication, 2000s and 2010s), *network* (AI, 1980s and 1990s), and two control
  words drawn at random (seed 2026) from round C's list, *component* and *having* (AI, 2010s and
  2020s). Per word, 16 pairs: 5 within the earlier cell, 5 within the later, 6 across, drawn at
  random (seed 2026). Four pairs, one per known word, are repeated with the sentences swapped,
  for 100 pairs in all, shuffled.
- **Judgements.** The author judges each pair, blind to field, decade and cell, on the DURel
  scale: 4 identical, 3 closely related, 2 distantly related, 1 unrelated, 0 cannot decide
  (guidelines in `docs/gold_guidelines.md`). The judgements are committed before any XL-LEXEME
  score of the pairs is computed.
- **Instrument.** XL-LEXEME as frozen in round U; a pair's score is the cosine similarity of its
  two usage vectors.
- **Agreement.** Spearman's ρ between the score and the judgement over the 96 distinct pairs,
  pairs judged 0 left out. The four repeated pairs give the author's own consistency (the
  difference between the two judgements), reported, not predicted.

### Prediction

**G1 — XL-LEXEME agrees with the author on this corpus.** Agreement is at least 0.5. Published
WiC agreement for XL-LEXEME against human judgements on DWUG EN is about 0.63 (Periti and
Tahmasebi 2024, computational-annotation setting). *Falsified* if under 0.5.

What it crosses out: if G1 passes, XL-LEXEME's reading of abstract sentences is not the problem,
and any failure of the usage-pair instrument on this corpus lies in the cells or the measure. If
it fails, the instrument misreads this genre (A6, A7), whatever the cells.

Scored by `scripts/gold_pairs.py score`, committed with the sampler before the pairs are drawn.

## Check 1b: how large a change can the usage-pair instrument see in this corpus? (committed 2026-11-10, before any planted usage was encoded)

Check 1a showed the documented changes present in round X's cells but very different in size:
the read share of the new sense rose by +0.95 for *prompt*, +0.35 for *attention*, +0.30 for
*network*, +0.27 for *platform* and +0.15 for *alignment*. This check plants changes of known size
into real abstracts where nothing changed, and finds the smallest the instrument detects.

### Method

- **Base.** *attention* in psychology, 2010s and 2020s: round X's quiet transition, in which
  check 1a read no attention-mechanism use in either cell. Round X's usages, 150 per cell.
- **Plant.** A share of the 2020s cell's usages is replaced by usages of *attention* from AI's
  2020s cell that carry check 1a's attention-mechanism markers (drawn from the whole cell, not
  only round X's 150): 10%, 20%, 30%, 40% and 60%, five random draws per share (seed 2026).
- **Placebo.** The same replacement at 40% with AI 2020s usages that carry no marker, which are
  mostly the stock phrase and cognitive senses psychology already has: a change of source, not
  of sense.
- **Found**, as in round X: J above the 95th percentile of 200 shuffles of the planted pair's
  usages, and above the 90th percentile of the control words' J on psychology's 2010s → 2020s
  transition (round X's controls, cached).
- A share is *detected* when at least four of its five draws are found.

### Predictions

**P1 — The instrument sees a planted change of a third or less.** The smallest detected share is
at most 30%, and every larger share is detected. *Falsified* otherwise.

**P2 — It does not see a change of source without a change of sense.** The placebo is found in
at most one of its five draws. *Falsified* if in two or more.

What it crosses out: with P1 and P2 passing, a change the size of *attention*'s (+0.35),
*network*'s (+0.30) or *platform*'s (+0.27) is within the instrument's reach, so a miss on those
in round X points to the cells (A1, A2, A4) rather than to the instrument's sensitivity. If P1
fails, those changes may simply be too small for it. If P2 fails, the instrument responds to
where usages come from, not only to what they mean.

Scored by `scripts/check_planted_pairs.py`, committed before it is run.

### Check 2b results (2026-11-10, `scripts/gold_pairs.py score`)

Scored as written. G1 falsified. The judgements were made interactively, one pair at a time,
with contexts longer than 30 words either side of the marked word shown trimmed; they were
committed before any XL-LEXEME score of the pairs was computed (commit f2e4260).

| | Verdict | Numbers |
|---|---|---|
| G1 | **Falsified** | Spearman 0.431 between XL-LEXEME's similarity and the author's judgement over 96 pairs (needed 0.5) |

The author's four repeated pairs received the same judgement both times. Mean judgement and mean
similarity by kind of pair: within the earlier cell 2.63 and 0.726, within the later 3.07 and
0.772, across 2.67 and 0.737.

### Reading

XL-LEXEME reads this corpus's sentences less as the author does than its published agreement on
DWUG EN (about 0.63) suggests. By the check's own terms, the instrument misreads this genre to a
degree (A6, A7), whatever the cells.

### Exploratory, after scoring: where the agreement sits

Not predicted; per word, Spearman over 16 pairs:

| Word | Agreement | The author's judgements |
|---|---|---|
| *attention* | 0.70 | ten 1s, six 4s |
| *prompt* | 0.66 | spread across 1–4 |
| *having* | 0.54 | five 1s, eleven 4s |
| *platform* | 0.43 | seven 1s, nine 3s and 4s |
| *network* | 0.10 | mostly 4s |
| *component* | 0.00 | mostly 4s |

Where a word's uses fall into clearly different senses (*attention*, *prompt*), XL-LEXEME agrees
with the author well. Where the author judged one meaning throughout (*network*, *component*), its
similarity still spreads from about 0.45 to 0.97: it responds to the context around a word as well
as to its meaning, the confound Kutuzov et al. (2022) describe. Part of the disagreement is sense
granularity: the author judged telephone, computer and neural networks as one meaning of *network*
in several pairs, where XL-LEXEME separates them.

## Check 2b′: a second, independent judge (committed 2026-11-10, before the second judge saw any pair)

Check 2b compared XL-LEXEME with one judge, so its agreement mixes the model's errors with the
judge's mistakes and sense boundaries. A second judge on the same pairs gives the agreement
between judges, the ceiling for any instrument, and makes XL-LEXEME's agreement with each judge
interpretable. No person was available; the second judge is a fresh instance of Claude (a
subagent with none of this project's conversation), as Periti and Tahmasebi (2024) used GPT-4.
It is another instrument, not gold, and is reported as such.

### Method

- **Input.** The same 100 items, with the contexts trimmed exactly as the author saw them (30
  words either side of the marked word), and `docs/gold_guidelines.md` verbatim, in a file outside
  the repository. The second judge is given nothing else: not the author's judgements, not
  XL-LEXEME's scores, not the key.
- **Output.** One DURel judgement per item (4, 3, 2, 1, or 0 for cannot decide).
- **Reported, not predicted:** Spearman between the two judges over the 96 distinct pairs (pairs
  either judge marked 0 left out); the share of pairs on which they agree exactly and within one
  point; XL-LEXEME's Spearman with the second judge and with the mean of the two; the second
  judge's consistency on the four repeated pairs.

Nothing here changes check 2b's verdict. Scored by `scripts/gold_pairs.py` (the `second` step),
committed before the second judge runs.

### Check 1b results (2026-11-10, `scripts/check_planted_pairs.py`)

Scored as written. P1 passed, P2 falsified. Base: 150 usages per cell; 30 controls (90th
percentile of J 0.0012); a pool of 291 marked and 608 unmarked AI 2020s usages.

| | Verdict | Numbers |
|---|---|---|
| P1 | **Passed** | Found in 0 of 5 draws at 10% and 20%, 5 of 5 at 30%, 40% and 60%: smallest detected share 30% |
| P2 | **Falsified** | The placebo (unmarked AI usages at 40%) was found in 5 of 5 draws |

### Reading

The instrument detects a planted change of a third of a word's usages and not a fifth, with a
sharp threshold between them. It also detects a change of source without a change of sense: by
the check's own terms, it responds to where usages come from, not only to what they mean.

### Exploratory, after scoring: how clean the placebo was

Not predicted. The placebo was meant to carry no mechanism uses, but check 1a's reading of AI's
2020s cell shows the markers catch about half of them (marked share 0.31, read share 0.60), so
about two in five unmarked usages are still the mechanism, and a 40% placebo plants roughly 17%
mechanism usages. A 20% plant of marked mechanism usages was found in none of five draws, so that
contamination alone is unlikely to account for the placebo being found in all five. Together with
check 2b (XL-LEXEME's similarity follows context within a sense) and round X's X5 (a third of
control words' contexts drift between decades), the result says J measures a change in the
contexts a word is used in, of which a change of sense is one source and a change of topic,
field mix or phrasing another (assumption A7). Round X's hits on *platform* and *network*, whose
read changes (+0.27, +0.30) sit at the 30% threshold, and its false alarm are open to that
reading.
