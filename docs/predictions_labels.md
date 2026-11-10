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
