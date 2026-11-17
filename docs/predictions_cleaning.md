# Predictions: does cleaning move the measurement?

Pre-registered checks of whether a cleaning step changes what the static measurement finds,
the thin end-to-end path asked for in `docs/pipeline_review.md` (scope item 5). As in the other
predictions files, a round's method, rule and bets are committed before any code that tests them
runs on its evidence; results are appended below, and nothing above them is edited afterwards.

## Round S1: does the largest identifiable junk dose move static drift? (committed 2026-11-16, before any static view of the rebuilt corpus was built)

After cleaning runs 9–17, a provisional review by calibrated agent judges found 16 failures in
200 kept records (8%, Wilson 95% interval 5–13%; agenda 4b), against a 2% target that comes from
audit practice, not from any effect on the measurement. The author chose (2026-11-16) to test
whether junk moves the measurement before more cleaning and before the 4b sign-off, which changes
agenda 9.1's order. The round is a cleaning diagnostic: no target's result is printed or read,
since those are what later predictions will be about.

### Method (fixed before any static view is built)

- **Static view, thin.** From the cached spaCy documents of the run machine: lowercase tokens,
  punctuation and whitespace dropped, tokens with no letter dropped, accents folded (Unicode
  NFKD, combining marks removed), the en dash written as a hyphen, and spaCy's in-context lemma
  for plural nouns (`NNS`) and third-person singular verbs (`VBZ`); each token keeps its
  character offsets, so a build's removals apply to it later. No phrases and no word tables
  (agenda stage 6 adds them). A removed range breaks the line, so no window crosses it.
- **Cells.** One field in one decade, as in the corpus (11 fields × 8 decades). A work counts in
  each of its fields' cells. Only kept works from the regular source (`dropped_reason` null,
  `source == "regular"`) enter a cell.
- **Base P.** The kept works of the build of record `2e20238`, before the cleaning batch, each with
  that build's removals. Every version below is a removal from P, so no version adds text.
- **Treatment C.** P minus two things:
  - the cleaning batch's changes, taken from the two builds' tables (the works the build of
    record `75f36a82` drops and P keeps, and the ranges it removes that P keeps), with each work's
    `text` asserted identical in both builds;
  - the works of P with no finite verb (`VerbForm=Fin` in the cached morphology) outside P's
    removals: about 23,355, about 92% junk on the calibrated judges' 129-drop sample (agenda 2.3,
    run 13).

  That is about 4.7% of works, the largest junk dose that can be identified now. P minus the batch
  alone and P minus the verbless works alone are measured only if C moves the measurement.
- **Placebos, 19.** In each cell, C's removal is matched outside C's own works. Each work C drops
  is replaced by a random work of the same cell from the same tenth of token lengths, drawn
  without replacement. Each passage C removes is replaced by a span of the same token length, at
  a random position in a random work. Seeds are fixed per placebo and cell (base seed 20261116).
- **Noise, 9 bootstrap samples.** Each cell's works of P drawn with replacement, the same number
  of works (Antoniak and Mimno 2018), seeded the same way.
- **Model.** PPMI factorised by SVD, one model per cell and version, with the settings of
  `embedding/ppmi.py`: window 5, 100 dimensions, minimum count 20, context smoothing 0.75,
  eigenvalue weight 0.5, seed 42. It is deterministic, unlike TWEC with gensim's workers; the
  choice of static model (agenda 6b) stays open.
- **Measure.** For the 77 within-field comparisons of consecutive decades: the neighbour shift of
  `drift/grid.py`'s `compare_cells` (a word measured at 150 uses or more in both cells, neighbours
  drawn from words at 50 or more, the 25 nearest), its percentile rank within frequency bins (the
  bins of `add_noise_floor`, on neighbour shift alone, since cosine drift is invalid for unaligned
  PPMI cells), and the flags, the words in the top 5% of their bin.
- **Primary statistic.** For a version V, M(V) is the mean, over the comparisons with at least
  200 measured words in P, of the flag turnover: 1 − the Jaccard index of V's flags and P's flags,
  over the words measured in either. A word that leaves the measured set counts as turnover.

### Decision rule

C moves the measurement if both hold:

1. M(C) is above M of all 19 placebos (a rank test, p = (1 + 0)/(19 + 1) = 0.05);
2. M(C) is at least half the median M of the 9 bootstrap samples.

The second part is a size condition. Random removal differs from junk removal almost always,
since junk has its own words, so the first part alone would count a tiny effect. An effect under
half of what resampling the same corpus does is below the measurement's resolution.

Reported but not decisive:
- 1 − Spearman's ρ of the within-bin ranks over the words measured in both versions;
- the number of words entering and leaving the measured set;
- the 32 single-word targets' combined absolute rank change, with no target named;
- each target's share of tokens inside C's removal, a corpus count.

### What each outcome means

- **C does not move the measurement.** The cleaning batch and the verbless junk do not matter for
  static drift under PPMI-SVD, and the 2% target can be revisited for that kind of junk (the
  author decides). Junk that has verbs (lists of reviewed books, archive descriptions, obituaries)
  stays untested by this round.
- **C moves the measurement.** P minus the batch and P minus the verbless works are measured
  separately, and cleaning continues, aimed first at the comparisons that move.

### Bets (Claude's, before any run)

- **S1a.** M(C) is above all 19 placebos: junk is not random text, so removing it changes some
  flags that random removal does not. Confidence about 70%.
- **S1b.** M(C) is below half the median bootstrap M, so C does not move the measurement. The
  measured words need 150 uses in both cells, and junk is short, and made of its own words
  (*pp.*, *vol.*, *profile*, names). So it changes few of those words' 25 nearest neighbours,
  while a bootstrap sample changes about a third of a cell's works. Confidence about 65%.
- **S1c.** The comparisons that involve the 1950s and 1960s move most under C, since much of the
  junk is there: 6,885 of the 23,355 verbless works and 1,378 of the 2,891 works the citation
  lines changed are from those two decades.

### Amendment: a null verdict is checked on bootstrap-averaged drift (2026-11-16, before any code ran)

A search of the literature after the round was committed found no study that measures how a
share of non-abstract documents moves semantic-change results, but it found a reason a null
verdict here could mislead. Antoniak and Mimno (2018, TACL 6) resample documents with
replacement and find that nearest-neighbour lists change substantially with the documents
present, more on small corpora. Of the four methods they test, PPMI's top-N lists were the least
stable, and they advise averaging over bootstrap samples rather than trusting one model.
Wendlandt et al. (2018) find words with 100–200 occurrences often unstable, and the measured
words here start at 150 uses. The rule above compares junk with the noise of one model per cell.
If the final static measurement averages bootstrap models, its noise falls and junk's relative
effect grows, so "C does not move the measurement" would hold only for single models.

So, if C does not move the measurement:

- **Round S1b, bootstrap-averaged drift.** A version's score for a word is its neighbour shift
  averaged over 9 bootstrap models per cell (each cell's works drawn with replacement, seeded).
  - averaged P reuses the round's 9 bootstrap samples of P;
  - averaged C draws 9 bootstrap samples of C's works;
  - averaged P′ draws a second, independent 9 of P, to give the noise of an average.

  The flags and the turnover are computed as above, on the averaged scores.
- **Rule.** Let D(junk) be the turnover between averaged P and averaged C, and D(noise) the
  turnover between averaged P and averaged P′. C moves the averaged measurement if
  D(junk) ≥ 1.5 × D(noise): junk adds at least half the averaged noise on top of it.
  (D(junk) ≈ D(noise) is what no effect looks like, since both are differences between
  averages.)
- **Outcomes.**
  - C moves the averaged measurement: the junk matters for any measurement that averages models,
    and cleaning continues.
  - C moves neither measurement: the 2% target can be revisited for this kind of junk.
- **Cost.** 18 × 88 more cell models, run only if the single-model verdict is null.

If C moves the single-model measurement, nothing changes: junk that matters at one model's noise
matters at lower noise too.

### Round S1 results (2026-11-16, `scripts/score_cleaning_sensitivity.py`)

The full run on the run machine (26 min 51 s, 2,640 cell models; static view of record `362befe0`,
code `7d87b439`): 68 of the 77 comparisons have at least 200 measured words in P. C removes
833,803 of P's 102,488,387 tokens (0.81%): 26,339 works dropped, 23,891 of them for having no
finite verb, plus the batch's passages. Junk is short, so 4% of the works is under 1% of the
text.

| Version | M (flag turnover) | Rank disagreement | Words leaving the measured set | Targets' combined rank change |
|---|---|---|---|---|
| C | 0.4592 | 0.0642 | 572 | 7,153 |
| 19 placebos | 0.4423–0.4606 | 0.0550–0.0570 | 605–681 | 6,506–7,067 |
| 9 bootstrap samples | 0.7782–0.7953 (median 0.7913) | 0.2487–0.2537 | 2,404–2,603 | 15,556–16,442 |

- **S1a: falsified.** M(C) = 0.4592 is below one placebo (0.4606); rank test p = 0.25.
- **S1b: falsified.** M(C) is above half the median bootstrap M (0.3957). But every placebo is
  too, so the size condition holds for random removal as well.
- **S1c: passed.** C's mean flag turnover is 0.5065 in the comparisons that start in the 1950s
  or 1960s, against 0.4469 in the others.
- **Verdict: C does not move the measurement** (the rule's first part fails). The amendment
  therefore applies: round S1b runs on bootstrap-averaged drift.

The targets' tokens inside C's removal: 8,396 of 1,522,689 (0.55%); the largest share for one
target is 2.22%.

### Reading

The flag statistic is dominated by the instrument's own instability. Removing 0.8% of the tokens
at random changes about 45% of the top-5% flags, and a bootstrap sample changes about 79%. So a
single PPMI-SVD model per cell cannot carry a flag decision: many words sit close to the 95th
percentile, and small perturbations of the matrix reorder them. That is Antoniak and Mimno's
(2018) warning, quantified on this corpus.

Against that noise, junk removal is indistinguishable from random removal on the primary
statistic. The secondary measures, which are continuous and not thresholded, do separate the two:
C's rank disagreement (0.0642) and its targets' combined rank change (7,153) are both above every
placebo. Junk has a small systematic effect, about a quarter of the bootstrap noise on the rank
measure, and the threshold noise of the flags hides it.

Round S1b, on bootstrap-averaged drift, decides whether that effect matters once the
measurement's own noise is reduced. Separately, the single-model instability bears on agenda 6b:
a static grid built on one model per cell would need bootstrap averaging (or another stabilised
estimate) before any flag means anything.
