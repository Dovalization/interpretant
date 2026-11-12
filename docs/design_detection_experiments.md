# Design draft: detection experiments

*Draft, 2026-11-12, for review. It turns the detection analysis of 2026-11-11 into experiments on
the rebuilt store (`docs/design_store_rebuild.md`). The jobs, traps and past results are in
`docs/methodology_comparison.md`; nothing here is a commitment rule.*

## Where detection stands

- **Across minus within (J)** found all four documented changes and the weak case in round X,
  with one narrow false alarm (0.0019 against a bar of 0.0015) on the smallest cells, 186 and 174
  usages. It also moved with the context it was given (round X's check 2b): the open problem is
  topic change, not the measure's arithmetic.
- **Average distance (APD)** beat its own shuffles on every documented change but measures
  variety as much as change, and failed the control-word bar.
- **The grid** missed changes because of its double-threshold rule (SemEval's graded change was
  0.29 while 1 of 19 changes was flagged), its decade windows (*platform* and *network* were about
  two-thirds changed within the earlier decade) and its static ceiling.
- **Sense shares** (round A) found 4 of 4 changes against shuffles and 2 of 4 against controls,
  missing *prompt* (the labeller) and *network* (the window).

## The checks every experiment reads

| Check | Cases |
|---|---|
| Documented changes | *alignment*, AI, 2010s→2020s; *attention*, AI, 2010s→2020s; *network*, AI, 1980s→1990s; *platform*, communication, 2000s→2010s |
| Quiet transitions | *attention*, AI, 1990s→2000s; *attention*, psychology, 2010s→2020s |
| Planted changes | Usages of one sense moved into a period in known shares, chosen without cue words |
| Placebo | A word whose meaning is fixed but whose field's topics moved (sociology's *years* in psychology) |
| Controls and shuffles | The 30 control words; periods shuffled within each word |

## Experiments

1. **J on every usage, with resampled intervals.** Today's J reads a sample of usages; with every
   usage in the store and a bootstrap over abstracts, each change gets an interval, and a change
   counts only when the whole interval clears the controls' upper bound. Aimed at the false alarm
   on small cells. Cost: encoding is done once in the store; the bootstrap is cheap.
2. **Controls from the same abstracts.** For each target usage, other words from the same
   abstracts form the control set, so a field's change of topics or sources moves the controls
   as much as the target. Aimed at topic change, the open problem. It needs the store's
   annotated documents to pick comparable words (same part of speech, similar frequency).
3. **Matched control words.** Each target paired with control words of the same frequency band
   and part of speech in the same cells, instead of one shared list. A sharper bar than the
   current 30 words, whose frequencies differ from the targets'.
4. **One measure for the grid.** The grid flags a word only when both cosine drift and
   neighbourhood shift are in the top 5% of its band; a single measure (either one, or their
   mean rank) with the bar set on the quiet transitions tests whether the double threshold, not
   the embeddings, caused the misses.
5. **Half-decade windows.** The documented changes of *platform* and *network* sit inside the
   earlier decade; half-decades place them, at the cost of smaller cells (see the data
   acquisition draft for the usages they need).
6. **Removing the era signal.** A direction in the vectors that encodes which decade a sentence
   sounds like, learned from words whose meaning did not change, removed before measuring
   (concept erasure, LEACE). Aimed at topic change for every vector measure; the risk is removing
   real meaning, which the documented changes would show.

## Order

Experiment 1 first: it needs nothing new beyond the rebuilt store and answers whether J's false
alarm was noise. Then 2 and 3, the two answers to topic change and to the control bar. 4 runs on
the grid as it is, at any time. 5 waits for the downloads. 6 last, as it changes every vector.

## Decisions needed

1. Whether results are judged against rules written before each run (as the earlier rounds
   were), or read openly and then confirmed on held-out cases.
2. Which held-out known answers, if any, are kept back from development.
3. The interval and bar in experiment 1 (for example a 95% interval against the controls'
   95th percentile).
