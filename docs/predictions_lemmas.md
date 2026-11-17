# Predictions: do the lemma exceptions move the measurement?

Pre-registered checks of whether the static view's lemmatisation changes what the static
measurement finds. As in the other predictions files, a round's method, rule and bets are committed
before any code that tests them runs on its evidence; results are appended below, and nothing
above them is edited afterwards.

## Round L1: the lemma exceptions against bootstrap noise (committed 2026-11-17, before either view below was built)

The lemma review (agenda 3.3, `docs/lemma_review.md`) produced `curated/lemma_exceptions.tsv`:
3,297 forms whose lemma differs from spaCy's in-context lemma, among them the override file's
lexicalised plurals (*data*, *media*, *means*), the adjective rule (*others*, *professionals*,
*characteristics* keep their form), the author's and the agents' corrections and the hyphenated
rule. Step 3.6 applies them to the cached spaCy documents and rebuilds the static view. The author
asked whether they matter for the measurement. The round is a diagnostic: the view with the
exceptions becomes the view of record whatever it finds, so the verdict cannot choose the view;
it decides whether further lemma work (the verb-type plurals still judged case by case, plurals
such as *logos*, the spellings of 3.4) is worth doing for static drift. No target's drift result
is printed or read.

### Method (fixed before either view is built)

- **View A.** The static view rebuilt from the cache with auxiliaries left as they are (commit
  1d9f6fc7) and no exceptions: `dvc repro build_static_view` at the commit after this file,
  recorded by its own lock commit (R_A) and copied aside.
- **View B.** The same cache with the exceptions applied by spaCy's attribute-ruler interface
  before the static tokens are written (R_B, the view of record). If the check of the target
  words below finds a target's own spelling written as another word, exceptions for those
  (form, tag) pairs are added before view B is built, and the change is recorded here first.
- **Alignment.** An exception changes a token's string, never whether the token is kept (every
  exception lemma contains a letter), so A and B hold the same works, tokens per work and
  character offsets. Any work whose offsets differ voids the run.
- **Cells and works.** One field in one decade, as in the corpus (11 fields × 8 decades). The kept
  works from the regular source of the build of record `75f36a82`, each with that build's
  removals, counted in each of its fields' cells. Nothing is dropped.
- **Versions.** Per cell, 9 bootstrap samples (works drawn with replacement, Antoniak and Mimno
  2018) for each of averaged A (seeds offset 1000), averaged A′ (a second, independent 9 of A,
  offset 2000) and averaged B (offset 3000), base seed 20261116, seeded from the cell's full name
  (115e1eba).
- **Model and measure.** As in rounds S1 and S1b (`docs/predictions_cleaning.md`): PPMI-SVD per
  cell (window 5, 100 dimensions, minimum count 20, context smoothing 0.75, eigenvalue weight 0.5,
  seed 42); for the 77 within-field comparisons of consecutive decades, the neighbour shift of
  `compare_cells` (measured at 150 uses, neighbours at 50, the 25 nearest), averaged over the 9
  replicates, a word entering an averaged table only if it is measured in all 9; its percentile
  rank within frequency bins.
- **Primary statistic.** For each comparison with at least 200 words in averaged A, r(V) is
  1 − Spearman's ρ of the within-bin ranks of averaged A and averaged V, over the words measured
  in both. A renamed word (*datum* in A, *data* in B; *professionals* separate only in B) is
  measured in one version only, so it cannot count as movement. D(V) is the mean of r(V).
  Round S1b found the top-5% flags themselves unreproducible (two averages of the same corpus
  disagree on half of them), so the rank, not the flag, carries the decision.

### Decision rule

The exceptions move the averaged measurement if D(B) ≥ 1.5 × D(A′): they add at least half the
noise of an average on top of it. This is round S1b's multiple; on this statistic its junk removal
gave 1.07 (0.0503 against 0.0471).

Reported but not decisive:
- the flag turnover of B and A′ against A, as in S1b;
- the number of words entering and leaving the measured set, no word named;
- the 32 single-word targets' combined absolute rank change for B and for A′, with the number of
  targets compared, no target named;
- the dose: tokens that differ between A and B, out of the kept tokens;
- each single-word target's token count in A and in B, a corpus count. *medium* is expected to
  lose the 57,608 uses of *media* that spaCy merged into it, by the override file's decision.

### Check of the target words (before view B is built)

Over the whole cache, with the exceptions applied: for each single-word target, the tokens of its
own spelling that view B writes as another word (expected none), and the forms that A and B write
as the target, as corpus counts.

### What each outcome means

- **B does not move the averaged measurement.** For static drift, the lemma exceptions are below
  the instrument's noise, and the remaining case-by-case lemma questions are not worth more work
  for the static measurement. The exceptions stay, as corrections.
- **B moves it.** Lemmatisation matters at this resolution: the remaining judgement calls and the
  spellings (3.4) come before the static experiments.

One limit, as in S1b: there is no placebo, so "moves" cannot alone separate these exceptions from
any relabelling of as many tokens.

### Bets (Claude's, before any run)

- **L1-a.** D(B) < 1.5 × D(A′): the exceptions do not move the averaged measurement. They touch
  about 4.5% of plural-noun and third-person-verb tokens, a large share of them renames (*data*,
  *media*) that leave the rank comparison, and the 9-model average still carries noise of the
  size S1b found. Confidence about 65%.
- **L1-b.** D(B) / D(A′) > 1.0: the exceptions add something above the noise, even if not 1.5
  times it, since the adjective rule changes frequent words' contexts (*characteristics*,
  *individuals*). Confidence about 70%.
- **L1-c.** B's combined target rank change is larger than A′'s, because *medium* loses the uses
  of *media*. Confidence about 75%.
