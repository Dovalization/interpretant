# Predictions: meaning by sense labels

Pre-registered predictions for the sense-label instrument (ROADMAP item 8d,
[`plan_sense_labels.md`](plan_sense_labels.md)). As in the other predictions files, a round's
predictions are committed before any code that tests them is run; results are appended below
them, and predictions and verdicts are never edited afterwards. Corrections go in new sections.

Two labellers run as separate instruments: Claude Sonnet 5.5 at low effort through the Message
Batches API, and JEV (TypeSafe AI) pinned at `jev-1.14.0`. Every prediction is scored for each
labeller on its own. A labeller that fails a round stops there, whatever the other does, and a
result counts only for the labeller that produced it.

## Round Y: do the labels rank a human benchmark's changes? (committed 2026-11-10, before any benchmark usage was labelled)

Rung 1. DWUG EN 2.0.1 gives, for 46 English words, the usages from two periods of the Corpus of
Historical American English (1810–1860 and 1960–2010), the clusters of meaning human annotators
formed from usage-pair judgements, and a graded change score per word: the Jensen-Shannon
divergence between the two periods' cluster distributions. The labels' own measure is the same
divergence over sense labels, so this round asks directly whether labels stand in for the human
clusters.

### Method (fixed before any benchmark usage is labelled)

- **Usages.** Every usage in `data/<lemma>/uses.csv` except the 320 that DWUG's cluster files
  mark `-1` (usages its annotators set aside; its change scores leave them out too): 8,787
  usages. Each is shown as its `context`, cut to 100 words around the target, the target marked
  with ⟦ and ⟧ (`interpretant.drift.labels.shown`).
- **Sense lists.** `data/sense_lists/dwug_en.toml`, committed in 199b273 before this file: 2 to 4
  senses per word, written from general knowledge, not reviewed by the author.
- **Labelling.** One request per usage, as `docs/plan_sense_labels.md` describes: Sonnet with a
  JSON schema allowing one label from the list (`BatchLabeller`), JEV with a Choice among the
  listed senses and a companion yes-or-no question on whether any listed sense fits (top choice
  above 0.7, *other* below 0.3, *unclear* between). 10% of the usages, drawn with seed 2026, are
  labelled a second time as separate requests.
- **Measures.** For each word, the divergence (in bits) between the two periods' distributions
  over the listed senses and *other*, with *unclear*, *failed* and *refused* left out; original
  requests only. The adjusted Rand index between a word's labels (every label a category) and
  DWUG's clusters, averaged over the words with at least two clusters.
- **Shuffled periods.** 20 shuffles (seeds 0–19) of each word's labelled usages between the two
  periods, keeping each period's size; the divergence and its Spearman with DWUG's graded change
  recomputed for each.
- Already seen: DWUG's graded change of *afternoon* (0) and *attack* (0.235), one usage of
  *plane*, the SemEval-2020 results generally (the benchmark's targets and some of their answers
  are public), round U's results, and the pilot. Not computed: any label of a benchmark usage.

### Predictions

**Y1 — The labels rank the benchmark's changes.** Spearman between the labels' divergence and
DWUG's graded change over the 46 words is at least 0.70. Between BERT's 0.563 and XL-LEXEME's
0.864 (round U) on the same benchmark. *Falsified* below 0.70.

**Y2 — The labels group usages as the annotators did.** The mean adjusted Rand index is at least
0.273, XL-LEXEME's published figure for clustering this benchmark (Periti and Tahmasebi 2024;
GPT-4 0.340, human 0.633). *Falsified* below 0.273. The lists cap a word at four senses while
DWUG's clusters are not capped, so this is the harder test.

**Y3 — The labels measure change, not variety.** With the periods shuffled, the mean Spearman
over the 20 shuffles is at most 0.20. Round U found APD keeps 0.842 when shuffled, because it
measures how varied a word's uses are; labels should not. *Falsified* above 0.20.

**Y4 — The labels are consistent.** At least 85% of the repeated usages get the same label both
times. *Falsified* below 85%.

**Y5 — The two labellers agree.** They give the same label to at least 80% of the usages (the
pilot: 89%). Descriptive; it decides nothing.

Reading rule, per labeller:

- **Y1, Y3 and Y4 pass:** the labeller passes rung 1 and goes on to rung 2.
- **Any of them fails:** the labeller stops at rung 1.
- Y2 is reported and decides nothing: a coarse list can rank changes well and still split usages
  differently from the annotators. The shares of *other*, *unclear*, *failed* and *refused* are
  reported; a labeller with more than 5% *failed* or *refused* is reported as not measured on this
  round, and stops.

Scored by `scripts/score_labels_benchmark.py`, committed before it is run.

### Round Y results for JEV (2026-11-10, `scripts/score_labels_benchmark.py`)

Scored as written, before Sonnet's labels were back. JEV labelled all 9,666 requests (8,787
usages and 879 repeats) with none failed or refused; *other* 1.5%, *unclear* 7.1%.

| | Verdict | Numbers |
|---|---|---|
| Y1 | **Passed** | Spearman 0.821 over 46 words (needed 0.70; BERT 0.563, XL-LEXEME 0.864) |
| Y2 | **Falsified** | mean ARI 0.267 over 42 words with two or more clusters (needed 0.273) |
| Y3 | **Falsified** | mean Spearman with periods shuffled 0.363 (at most 0.20; range 0.160 to 0.632) |
| Y4 | **Passed** | 96.8% of 879 repeats labelled the same (needed 85%) |

By the reading rule, Y3's failure stops JEV at rung 1.

### Correction: Y3's bar fails the gold standard (2026-11-10, after scoring JEV, before scoring Sonnet)

Y3 was meant to show that the labels measure change, not a word's variety, as round U showed APD
does. The bar was not checked against DWUG's own human clusters before it was committed. Checked
now, by the scorer's shuffle procedure applied to the clusters themselves: the clusters'
divergence matches DWUG's graded change exactly (Spearman 1.0, as it is the same computation),
and with the periods shuffled it keeps a mean Spearman of 0.855 (range 0.812 to 0.885) over the
same 20 shuffles. The gold standard fails Y3 by more than JEV does.

The reason: shuffling a word whose senses changed mixes two different distributions, so the
shuffled divergence is the sampling noise of a varied pooled distribution, which is larger the
more senses the word has in use; a word with one sense gives almost none. Words that changed
have, pooled, more senses in use, so the shuffled divergence correlates with the change for any
labeller that reads senses correctly, and more strongly the better it reads them. Y3 as written
cannot separate change from variety; it penalises accuracy. Round U's version of the check worked
for J because J subtracts the within-period distances; the divergence of label distributions does
not.

What this changes: nothing in the verdicts above, which stand as scored, including JEV's stop at
rung 1 under the rule as written. What it means: Y3's falsification is not evidence that JEV
measures variety, and Sonnet's Y3, when scored, will be read the same way. Whether a labeller
stopped only by Y3 may continue is the author's decision, and if it does, it continues under a
new pre-registered round with a variety check calibrated on the gold first, not by reinterpreting
round Y.

## Round Z: round Y with sense lists from WordNet (committed 2026-11-10, before any usage was labelled with these lists, and before Sonnet's round Y was scored)

A robustness round. Round Y's sense lists were written by Claude, who knew the SemEval-2020
results in general; a labeller passing round Y could owe part of its result to how the lists
were drawn. Round Z repeats round Y with lists no one wrote: for each word, the first four
WordNet 3.0 senses of its part of speech that contain the word itself, in WordNet's order (most
frequent first), each with WordNet's own definition and example, instance synsets (named people
and places) left out (`scripts/wordnet_sense_lists.py`, lists committed in 30c4bc4). The
instance rule was added after a first generation showed two person names (Sir Hiram Maxim under
*maxim*, Sir Herbert Beerbohm Tree under *tree*), before any usage was labelled with the lists.

### Method

As round Y in everything else: the same 8,787 usages, the same labellers and requests, the same
10% of repeats (seed 2026), the same divergence and adjusted Rand index. Five words get a single
WordNet sense (*chairman, chef, lass, maxim, quilt*), so their distribution is that sense against
*other*. WordNet's order follows modern sense-tagged text, and the cap can drop a sense that
mattered in the 19th century (*plane* keeps the aeroplane and loses the plane tree); that is the
price of lists no one chose. No variety check is made: round Y's Y3 was shown unable to separate
change from variety, and its successor waits for a design calibrated on the gold first.

- Already seen: everything round Y lists, JEV's round Y results, the Y3 correction, and the
  generated lists. Not seen: Sonnet's round Y scores, or any label made with these lists.

### Predictions (for each labeller)

**Z1 — The WordNet labels rank the benchmark's changes.** Spearman between the labels'
divergence and DWUG's graded change over the 46 words is at least 0.70. *Falsified* below.

**Z2 — The lists do not drive round Y's result.** The labeller's Spearman in round Z is within
0.10 of its Spearman in round Y. *Falsified* if they differ by more than 0.10.

**Z3 — The labels are consistent.** At least 85% of the repeated usages get the same label both
times. *Falsified* below.

**Z4 — Grouping, compared.** The mean adjusted Rand index with DWUG's clusters, reported beside
round Y's; descriptive.

Reading rule: round Z decides nothing about whether a labeller continues. If Z1 or Z2 fails for a
labeller, every citation of that labeller's round Y result says that it depends on the sense
lists. If both pass, round Y's result is reported as not depending on who wrote the lists.

Scored by `scripts/score_labels_wordnet.py`, committed before it is run.
