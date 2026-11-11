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

### Round Y results for Sonnet, and Y5 (2026-11-10, `scripts/score_labels_benchmark.py`)

Scored as written, after the Y3 correction above was committed and after round Z was
pre-registered. Sonnet labelled all 9,666 requests with none failed or refused; *other* 8.4%,
*unclear* 0.5%.

| | Verdict | Numbers |
|---|---|---|
| Y1 | **Passed** | Spearman 0.767 over 46 words (needed 0.70; BERT 0.563, XL-LEXEME 0.864) |
| Y2 | **Passed** | mean ARI 0.287 over 42 words with two or more clusters (needed 0.273; GPT-4 0.340) |
| Y3 | **Falsified** | mean Spearman with periods shuffled 0.265 (at most 0.20; range −0.079 to 0.489) |
| Y4 | **Passed** | 98.9% of 879 repeats labelled the same (needed 85%) |
| Y5 | **Passed** | the labellers gave the same label to 84.5% of 8,787 usages (at least 80%; descriptive) |

By the reading rule, Y3's failure stops Sonnet at rung 1 too. Read with the correction above:
both labellers rank the benchmark's changes well (JEV 0.821, Sonnet 0.767, between BERT's 0.563
and XL-LEXEME's 0.864), both are consistent, and Sonnet groups usages slightly better than
XL-LEXEME's clustering (0.287 against 0.273, below GPT-4's 0.340). Neither has been shown to
measure variety, because the test meant to show it fails the gold standard itself (0.855). Both
stopped on that test alone.

### Round Z results for JEV (2026-11-10, `scripts/score_labels_wordnet.py`)

Scored as written, before Sonnet's round Z labels were back. JEV labelled all 9,666 requests
with none failed.

| | Verdict | Numbers |
|---|---|---|
| Z1 | **Falsified** | Spearman 0.635 over 46 words (needed 0.70) |
| Z2 | **Falsified** | round Z 0.635, round Y 0.821, difference −0.187 (at most 0.10) |
| Z3 | **Passed** | 96.6% of 879 repeats labelled the same (needed 85%) |
| Z4 | Descriptive | mean ARI 0.220 with WordNet lists, 0.267 with round Y's |

By the reading rule, every citation of JEV's round Y result says that it depends on the sense
lists.

Exploratory, after scoring: with WordNet's lists JEV found no listed sense that fitted (*other*
or *unclear*) for 19.2% of the earlier period's usages and 19.4% of the later's, against 8.3%
and 8.8% with round Y's lists. The WordNet lists fit this corpus worse in both periods alike,
not only its 19th-century usages. The drop is therefore consistent with two readings this round
cannot separate: round Y's lists were better because their writer knew which words changed, or
they were better because they were written for these words and WordNet's first four senses miss
many of the uses (the cap, modern frequency order and single-sense words). Either way, the
result depends on the list, which for this corpus is the reason the author's review of the lists
matters.

### Round Z results for Sonnet (2026-11-10, `scripts/score_labels_wordnet.py`)

Scored as written. Sonnet labelled all 9,666 requests with none failed or refused.

| | Verdict | Numbers |
|---|---|---|
| Z1 | **Falsified** | Spearman 0.588 over 46 words (needed 0.70) |
| Z2 | **Falsified** | round Z 0.588, round Y 0.767, difference −0.180 (at most 0.10) |
| Z3 | **Passed** | 98.1% of 879 repeats labelled the same (needed 85%) |
| Z4 | Descriptive | mean ARI 0.235 with WordNet lists, 0.287 with round Y's |

By the reading rule, every citation of either labeller's round Y result says that it depends on
the sense lists. The two labellers dropped by almost the same amount (−0.180 and −0.187), so the
dependence belongs to the lists, not to one labeller.

Exploratory, after scoring: Sonnet found no fitting listed sense for 19.4% and 17.4% of the two
periods' usages with WordNet's lists, against 8.8% and 9.0% with round Y's, the same pattern as
JEV's. The two readings recorded for JEV stand, and this round cannot separate them.

## Round Q: sense lists induced from the texts, three labellers (committed 2026-11-10, before any list was induced or any usage of half B labelled)

Rounds Y and Z showed that the labels rank DWUG's changes well, and that the result depends on the
sense lists: with WordNet's first four senses, both labellers dropped by about 0.18, and found no
fitting sense for about one usage in five. Round Q builds the lists from the texts instead: from
one half of each word's usages, with evidence from methods that are not generative language
models, consolidated by a language model that may only edit what the evidence shows, then frozen
and measured on the other half by three labellers, one of them not a generative model. If it
reaches round Y's level, the method needs neither hand-written lists nor any one model.

It also replaces round Y's Y3 with a variety check calibrated on the gold first (below), so a
labeller that passes round Q passes rung 1, with induced lists, despite round Y's stop.

### Method (fixed before any list is induced)

- **Halves.** For each word and period, DWUG's kept usages (round Y's 8,787), sorted by key and
  shuffled with one `random.Random(2026)` taken over the words in order and periods 1 then 2; the
  first half (rounded down) of each is half B, the rest half A. Half B: 4,377 usages. Lists are
  induced from half A only, and every measure is on half B only.
- **Step A, evidence from half A, no generative model.**
  1. *Substitutes.* ModernBERT-large (`answerdotai/ModernBERT-large`, revision 45bb4654) reads
     each usage's context, cut to 100 words around the target, with " (or even [MASK])" after
     the target, the pattern of Amrami and Goldberg (2019), and gives its 20 most likely fillers.
     Fillers are lowercased; dropped if not alphabetic, shorter than two letters, an English stop
     word (scikit-learn's list), or the target or its form with -s, -es, -d, -ed or -ing.
     Substitutes found in at least 3 of the word's half-A usages are nodes of a graph whose edges
     count the usages in which two substitutes appear together; Louvain communities (networkx,
     resolution 1, seed 2026) group them (Eyal et al. 2022). A usage belongs to the community
     holding most of its substitutes (ties to the earlier community); communities holding under
     5% of the word's half-A usages are dropped. Each kept community is shown by its 10 most
     frequent substitutes and the 3 half-A usages with most substitutes in it.
  2. *Neighbours.* The 15 most frequent words immediately before or after the target in half A
     (alphabetic, lowercased, stop words left out), with their counts.
  3. *Dictionaries.* Every WordNet 3.0 synset of the word's part of speech that contains the word
     and is not an instance (named people and places), and every Wiktionary sense of that part of
     speech from kaikki.org's per-word extract (downloaded once, the date recorded), each with its
     gloss and topic tags.
- **Step B, consolidation.** Claude Opus 5.5 at high effort, through the Batch API, one request per
  word, sees the word's evidence and nothing about periods, fields or DWUG's scores, and returns,
  by a JSON schema, 1 to 8 senses (key, one-line definition, invented example, and the evidence
  each rests on) and a map from every kept community to one sense or to none. Rules given in the
  prompt, committed with the code: every sense cites at least one community or neighbour (a
  dictionary entry alone is not enough, so senses the texts do not show are left out); the fewest
  senses a careful reader could tell apart in context; senses a reader could not reliably tell
  apart are merged.
- **Coverage pass.** Sonnet labels every half-A usage with the consolidated lists (as in round Y).
  For a word whose half-A usages are more than 10% *other* or *unclear*, a second Opus request gets
  its list and up to 40 of those usages with their substitutes, and may add senses (8 at most) by
  the same rules, citing those usages. One revision only; the lists are then frozen whatever their
  coverage and committed before half B is labelled.
- **Step C, labelling half B, three labellers.** Sonnet (Batch API, low effort) and JEV
  (`jev-1.14.0`) as in round Y, with 10% repeats (seed 2026). A third labeller with no generative
  model: each half-B usage's substitutes, found as in step A, place it in the kept community
  holding most of them, read through the step B map; a usage in no kept community, or in one
  mapped to none, is *other*.
- **Measures, on half B.** The divergence between the two periods' sense distributions, as round
  Y, and its Spearman with DWUG's graded change. *Change beyond variety:* the partial Spearman of
  the divergence with the graded change, controlling for the mean divergence over 20 shuffles of
  the periods (seeds 0 to 19), on ranks with linear residuals.
- **Calibrated on the gold first.** DWUG's own clusters on half B: Spearman 0.966 with the graded
  change, mean shuffled divergence alone 0.874, partial controlling for it 0.849. A measure of
  variety alone would have a partial near 0. A variant considered and dropped: the divergence minus
  its mean shuffled divergence, on which the gold itself reaches only 0.724.
- Already seen: rounds Y and Z in full; the gold calibration above; substitutes from BERT-large and
  ModernBERT-large on five invented sentences (ModernBERT gives the attention mechanism's
  substitutes, BERT does not); kaikki.org's extract for *plane*. Not computed: any list induced from
  DWUG, any label of half B, and any round Y labeller's score on half B or on the partial measure.

### Predictions (Sonnet and JEV each; the substitute labeller where stated)

**Q1 — The induced lists rank the benchmark's changes.** Spearman over the 46 words, on half B,
is at least 0.70. For all three labellers. *Falsified* below.

**Q2 — Induced lists do as well as hand-written ones.** The labeller's Spearman on half B with
induced lists is at least its round Y labels' Spearman on the same half B, minus 0.10. Sonnet and
JEV. *Falsified* otherwise.

**Q3 — The measure tracks change beyond variety.** The partial Spearman is at least 0.40, about
half the gold's 0.849. For all three labellers. *Falsified* below.

**Q4 — The lists cover the texts.** At most 10% of half B's usages are *other* or *unclear*. For
all three labellers (for the substitute labeller, *other*). *Falsified* above.

**Q5 — The labels are consistent.** At least 85% of the repeats get the same label. Sonnet and
JEV. *Falsified* below.

**Q6 — Grouping.** The mean adjusted Rand index with DWUG's clusters on half B, beside round Y's
labels on half B. Descriptive.

**Q7 — The three labellers agree.** The share of half-B usages on which each pair gives the same
label. Descriptive.

Reading rule:

- A labeller **passes rung 1 with induced lists** if Q1, Q3, Q4 and Q5 pass (for the substitute
  labeller, Q1, Q3 and Q4). It then goes on to rung 2 with the same pipeline: lists induced from
  this corpus's texts by the steps above, reviewed by the author before they are frozen.
- A labeller that fails any of them stops. If all three stop, induced lists stop.
- Q2 decides how induced lists are described: passing, as doing as well as hand-written ones;
  failing, as weaker, with the gap reported.
- Opus induces the lists and Sonnet labels with them; both are Claude models, and Sonnet's
  coverage labels on half A shape the revision pass. JEV and the substitute labeller are the
  checks on that: agreement between all three (Q7) is the measure of how much the result rests on
  one family of models.

Cost: about $3 for consolidation and revision (Opus, batch prices), $3.50 for the coverage pass
and $4 for half B on Sonnet, about $0.15 for JEV; the substitutes run locally.

### Round Q amendment: a rule-built arm (committed 2026-11-10, while step A ran and before its output was read; before step B)

Added at the author's request, to measure how much the language model's part in building the
lists matters. Nothing above changes.

- **Rule-built lists.** From step A's evidence alone, with no language model: every kept
  substitute community of a word becomes one sense, keyed by its two most frequent substitutes and
  described as "the sense in which the word could be replaced by" its ten most frequent
  substitutes, with no example. A word with no kept community gets one sense, "the word in any
  use". No coverage pass or revision: the lists are frozen as generated
  (`data/sense_lists/dwug_en_rules.toml`, with their community map) and committed before half B is
  labelled with them. They hold no DWUG text.
- **Labelling.** Half B again, by the same three labellers: Sonnet and JEV with the rule-built
  lists (10% repeats, seed 2026), and the substitute labeller reading each usage's community
  directly, which makes this arm free of generative models end to end.
- **Predictions, for each labeller.** R1: Spearman on half B at least 0.70. R2: at least the same
  labeller's Spearman with Opus's lists (round Q's Q1) minus 0.10. R3: partial Spearman at least
  0.40. R4: at most 10% *other* or *unclear*. R5 (Sonnet and JEV): repeats at least 85% the same.
  R6 (grouping) and R7 (agreement between the labellers on these lists) are descriptive.
- **Reading rule.** R2 decides how the language model's part in building the lists is described:
  passing, as changing the result little; failing, as mattering, with the gap reported. A labeller
  that passes R1, R3, R4 and R5 (the substitute labeller R1, R3 and R4) also passes rung 1 with
  rule-built lists, which then become an alternative for rung 2.
- Cost: about $3.50 for Sonnet and $0.12 for JEV on half B; the substitute labeller runs locally.

Scored by `scripts/score_labels_induced.py score_rules`, added with this amendment.

### Round Q scorer correction (2026-11-10, after the induced arm and Sonnet's rule-built results were printed, before JEV's and the substitute labeller's rule-built results were)

`score_rules` stopped with an error after Sonnet: with the rule-built lists, JEV left every half-B
usage of *afternoon* in one period *unclear*, so that word's divergence is undefined, and the
correlations cannot be computed with it. The scorer now leaves out a word whose divergence (or mean
shuffled divergence) is undefined, prints which, and computes every correlation over the rest. No
word was left out for any labeller in the induced arm or for Sonnet in the rule-built arm, so their
printed results are unchanged; for JEV on the rule-built lists the correlations are over 45 words.

### Round Q results (2026-11-10, `scripts/score_labels_induced.py score` and `score_rules`)

Scored as written, with the correction above. Every label request succeeded (4,815 for each of
Sonnet and JEV on each arm, 4,377 for the substitute labeller). Opus's first lists held 1 to 5
senses; the coverage pass found 22 words over 10% *other* or *unclear* on half A (up to 48% for
*pick*), and the revision left 14 words at the cap of 8 senses.

Lists induced by Opus (half B; round Y's hand-written lists on the same half for reference):

| | Sonnet 5.5 | JEV 1.14 | Substitute labeller |
|---|---|---|---|
| Q1 Spearman (needed 0.70) | 0.443, **falsified** | 0.502, **falsified** | 0.433, **falsified** |
| Q2 against round Y on half B (at most 0.10 below) | 0.693, **falsified** | 0.800, **falsified** | — |
| Q3 partial Spearman (needed 0.40) | 0.307, **falsified** | 0.362, **falsified** | 0.183, **falsified** |
| Q4 *other* or *unclear* (at most 10%) | 6.0%, **passed** | 6.8%, **passed** | 1.4%, **passed** |
| Q5 repeats the same (needed 85%) | 98.4%, **passed** | 98.6%, **passed** | — |
| Q6 mean ARI (round Y on half B) | 0.252 (0.330) | 0.245 (0.302) | 0.168 |

Q7: Sonnet and JEV agree on 84.9% of half-B usages, each with the substitute labeller on 66.5–66.7%.

Rule-built lists:

| | Sonnet 5.5 | JEV 1.14 | Substitute labeller |
|---|---|---|---|
| R1 Spearman (needed 0.70) | 0.385, **falsified** | 0.590 over 45 words, **falsified** | 0.420, **falsified** |
| R2 against Opus's lists (at most 0.10 below) | 0.443, **passed** | 0.502, **passed** | 0.433, **passed** |
| R3 partial Spearman (needed 0.40) | 0.272, **falsified** | 0.453, **passed** | 0.207, **falsified** |
| R4 *other* or *unclear* (at most 10%) | 11.7%, **falsified** | 16.2%, **falsified** | 1.4%, **passed** |
| R5 repeats the same (needed 85%) | 96.8%, **passed** | 95.7%, **passed** | — |
| R6 mean ARI | 0.185 | 0.227 | 0.110 |

R7: on rule-built lists Sonnet and JEV agree on 63.9%, each with the substitute labeller on
51.6–63.0%.

By the reading rules: no labeller passes rung 1 with induced lists or with rule-built lists, and
induced lists are described as weaker than hand-written ones, by 0.25 for Sonnet and 0.30 for JEV
on the same half. R2 passed for all three labellers: the language model's part in building the
lists changes the result little (JEV did better with the rule-built lists than with Opus's).

Read from the tables: lists built from the texts, by either route, cover the texts (Opus's lists
leave 6–7% unfitted) but rank the benchmark's changes far worse than the hand-written lists did on
the same usages, and no better than BERT's 0.563 on the full benchmark. Coverage is therefore not
what the induced lists lack. The substitute communities, read in step A, also split usages by
topic and grammatical number (*attack*'s military and violent-crime groups, *plane*'s singular and
plural vehicles), the context-for-meaning confusion of the earlier instruments in another form;
Opus merged such groups at first, and the revision then split 14 words up to eight senses.
Whether the hand-written lists win because their writer knew the answers or because they draw the
lines a reader draws, this round cannot say, as round Z could not; it does show that neither the
substitute evidence nor a language model consolidating it recovers those lines.

## Round E: a classifier on XL-LEXEME, trained from cue words, no generative model (committed 2026-11-10, before any classifier was trained or any usage labelled by it)

Round Q showed that lists built from the texts rank DWUG's changes far worse than hand-written
lists, whoever labels. The author chose the next design: humans draw the lines (on this corpus,
lists drafted by Claude and approved by the author), and a classic NLP model does the labelling,
with no generative model in the pipeline and no labelling by the author. Round E tests that
labeller on DWUG, with round Y's hand-written lists standing in for approved ones, on round Q's
halves, so it compares directly with Sonnet and JEV on the same usages.

### Method (fixed before any classifier is trained)

- **Lists.** Round Y's (`data/sense_lists/dwug_en.toml`), 149 senses over 46 words.
- **Cue words.** For every sense, words that signal it when they appear near the target
  (`data/sense_lists/dwug_en_cues.toml`, committed before this file), written from general
  knowledge before any DWUG usage was read for them; round Q's step A evidence for *plane* and
  *attack*, with three half-A usages per substitute group, had been seen.
- **Seeds, from half A only.** A half-A usage is a seed for a sense when a cue of that sense, and
  of no other sense of the word, appears among the five words on either side of the target
  (lowercased words, the target itself left out). Each sense's invented example from the list is
  also a seed, the target found as the first word starting with the lemma; so every sense has at
  least one.
- **Classifier.** XL-LEXEME (round U's configuration) encodes every seed and every half-B usage.
  For each word, a logistic regression (scikit-learn, C = 1, balanced class weights, 1,000
  iterations) is trained on the length-normalised vectors of its seeds and gives every half-B usage
  one listed sense; it never says *other* or *unclear*.
- **Measures, on half B.** As round Q: the divergence's Spearman with DWUG's graded change, the
  partial Spearman controlling for the mean shuffled divergence, the adjusted Rand index.
- **Reference figures on the same half B,** with round Y's lists: Sonnet 0.693, JEV 0.800 (round
  Q's Q2); DWUG's clusters 0.966 (partial 0.849).
- Already seen: rounds Y, Z and Q in full, including the reference figures above. Not computed:
  any seed count, any classifier, any label.

### Predictions

**E1 — The classifier ranks the benchmark's changes.** Spearman on half B at least 0.70.
*Falsified* below.

**E2 — It does about as well as a language model labeller.** Spearman on half B at least Sonnet's
0.693 minus 0.10, that is 0.593. *Falsified* below.

**E3 — It tracks change beyond variety.** Partial Spearman at least 0.40. *Falsified* below.

**E4 — Agreement with the language models.** The share of half-B usages on which the classifier
gives the same label as Sonnet and as JEV did with the same lists in round Y, counting their
*other* and *unclear* as disagreement. Descriptive.

**E5 — Seeds.** The number of cue seeds per sense, and the share of senses with at least three.
Descriptive.

**E6 — Examples alone.** The same classifier trained on the list's invented examples only (one per
sense): its Spearman on half B. Descriptive: how much the cue seeds add to the definitions.

Reading rule: the classifier passes rung 1 if E3 passes and E1 or E2 does (as good as the bar, or
within 0.10 of a language model labeller on the same usages). It then goes to rung 2 on this
corpus with lists and cue words drafted by Claude and approved by the author, and Sonnet and JEV
as comparison points. If it fails, a language model labeller is the fallback, and the gap is
reported as the cost of leaving generative models out.

Scored by `scripts/score_labels_classifier.py`, committed before it is run.

### Round E results (2026-11-10, `scripts/score_labels_classifier.py`)

Scored as written. XL-LEXEME encoded 8,935 usages and examples; 148 of the 149 senses' examples
contain a form of the lemma (one sense trained on cue seeds alone); 1,109 cue seeds were found in
half A.

| | Verdict | Numbers |
|---|---|---|
| E1 | **Falsified** | Spearman 0.669 on half B (needed 0.70) |
| E2 | **Passed** | 0.669, needed 0.593 (Sonnet's 0.693 minus 0.10) |
| E3 | **Passed** | partial Spearman 0.509 (needed 0.40; gold 0.849) |
| E4 | Descriptive | same label as Sonnet on 75.2% of half B (82.2% where Sonnet chose a listed sense), as JEV on 76.0% (82.9%) |
| E5 | Descriptive | 1,109 cue seeds over 149 senses; 62% of senses with three or more; median 5 |
| E6 | Descriptive | examples alone: Spearman 0.628 |
| | Descriptive | mean ARI on half B 0.266 |

By the reading rule the classifier passes rung 1 (E3, and E2): it goes to rung 2 on this corpus
with lists and cue words drafted by Claude and approved by the author, Sonnet and JEV as comparison
points.

Exploratory, after scoring, for context: with the same lists on the same half B, Sonnet's round Y
labels have Spearman 0.693, partial 0.646 and ARI 0.330; JEV's 0.800, 0.667 and 0.302. The
classifier, with no generative model and no labelling by a person, ranks the changes 0.02 below
Sonnet and 0.13 below JEV, and tracks change beyond variety less well than either (0.509 against
0.646 and 0.667). The invented example sentences alone already reach 0.628; the cue seeds add 0.04.

## Round T: does the classifier read this corpus as the author does? (committed 2026-11-10, before the corpus lists were frozen or any check 2b usage was labelled with them)

Rung 2. The classifier of round E, with sense lists and cue words for this corpus that Claude
drafted and the author refined on a review page (`data/sense_lists/corpus_draft.toml`, frozen and
committed after the author's approval and before this round is run), labels the 186 distinct
usages of check 2b, whose 100 pairs the author judged (*attention, prompt, platform, network,
component, having*).

### Method (fixed now; the lists and cues are fixed at their freezing commit)

- **Training sentences.** Every usage of the six words in the whole-corpus sample (every field and
  decade of the grid, at most 150 abstracts per cell, seed 1, round C′'s quality rules) whose cues
  name one sense, plus each sense's example sentence, with check 2b's 186 usages left out.
- **Classifier.** As round E: XL-LEXEME vectors, one logistic regression per word (C = 1, balanced
  class weights, 1,000 iterations), giving each of the 186 usages a listed sense and a probability
  for each sense.
- **Measures, over the author's judged pairs.** (a) Spearman between "the pair's two usages get the
  same sense" and the author's judgement; (b) Spearman between the overlap of the two usages' sense
  probabilities (the sum over senses of their product) and the judgement. Both with the original
  judgements and with the correction pass; per word, descriptively.
- **Comparison labellers.** Claude Sonnet 5.5 (Batch API, low effort) and JEV (`jev-1.14.0`) label
  the same 186 usages with the same frozen lists, as in round Y; measure (a) for each.
- **Reference figures on the same pairs.** XL-LEXEME's raw similarity 0.456 (0.494 with the
  correction pass); the second judge 0.616 (0.657).
- Disclosed: the labeller pilot showed Claude these 186 usages, and three choices in the drafts
  came from it (platform's delivery channel, having's obligation, component's boundary); the
  review page samples the same corpus sample check 2b drew from, so the author may have seen some of
  these sentences there as cue matches. The round is therefore partly in-sample for the lists;
  leaving the usages out of the training sentences keeps it out of sample for the classifier.

### Predictions

**T1 — Labels keep the encoder's agreement with the author.** Measure (b) for the classifier is
at least 0.456, XL-LEXEME's raw similarity on the same pairs. *Falsified* below.

**T2 — The classifier reads the corpus about as well as a language model labeller.** Measure (a)
for the classifier is at least Sonnet's measure (a) with the same lists, minus 0.10. *Falsified*
below.

**T3 — Agreement with the language models.** The share of the 186 usages on which the classifier
gives the same sense as Sonnet and as JEV. Descriptive.

Reading rule: the classifier passes rung 2 if T1 and T2 pass, and goes on to rung 3 with the same
lists and the control words' lists (frozen after their own review). If it fails, the gap is
reported and a language model labeller is the fallback for rung 3, as round E's rule says.

Scored by `scripts/score_labels_corpus.py`, committed before it is run.

### Round T results (2026-11-11, `scripts/score_labels_corpus.py`)

Scored as written, with the lists frozen at `9332d86`. XL-LEXEME encoded 16,246 training
sentences, examples and check 2b usages. Training sentences per word: *attention* 6,713, *prompt*
273, *platform* 1,110, *network* 3,385, *component* 1,282, *having* 3,268. 96 of the 100 pairs
carry a judgement.

| | Verdict | Numbers |
|---|---|---|
| T1 | **Falsified** | probability overlap 0.449 (needed 0.456, XL-LEXEME's raw similarity) |
| T2 | **Falsified** | same sense 0.397 (needed 0.414: Sonnet's 0.514 minus 0.10) |
| T3 | Descriptive | same sense as Sonnet on 76.3% of the 186 usages, as JEV on 76.3% |
| | Descriptive | with the correction pass: classifier same sense 0.441, overlap 0.522 (XL-LEXEME 0.494); Sonnet 0.563; JEV 0.504 (0.451 on the original judgements) |

By the reading rule the classifier does not pass rung 2. The gap is 0.117 below Sonnet on same
sense, 0.017 beyond the margin, and 0.007 below the encoder's own agreement. A language model
labeller is the fallback for rung 3.

Exploratory, after scoring: same-sense Spearman per word, 16 pairs each, so every figure is
noisy.

| Word | Classifier | Sonnet | JEV |
|---|---|---|---|
| *attention* | 0.87 | 0.77 | 0.77 |
| *having* | 0.54 | 0.71 | 0.59 |
| *prompt* | 0.46 | 0.75 | 0.87 |
| *platform* | 0.27 | 0.41 | 0.40 |
| *network* | 0.27 | 0.25 | -0.05 |
| *component* | 0.13 | 0.35 | 0.35 |

The classifier leads on *attention*, the word with the most training sentences, and is level on
*network*. It falls furthest behind on *prompt*, the word with the fewest (273). Round A was
written for the classifier (its adjusted shares use the classifier's out-of-fold confusion
matrix), so running it with a language model labeller needs an amendment committed before it is
run.

## Round U: the revised classifier against Sonnet, judged blind by the author (committed 2026-11-11, before any round U sentence was drawn or labelled)

Rung 2, second attempt. Round T's verdict stands: the classifier failed T1 and T2. The author
judged the classifier worth a second attempt, so the plan's rule of no repair rounds is amended for
this one case. A revised classifier gets one new round on fresh sentences, never on round T's pairs.
If it fails, the language model fallback of rounds E and T holds.

### What changed, and from what evidence

- **Lists.** `data/sense_lists/corpus_v2.toml` (`0b3bfa3`): cue words added to six senses
  (artificial neural network, part of a system, language-model input, cue to a person, quick or
  immediate, experiencing or causing). Nothing is removed or redefined, so the definitions are
  version 1's.
- **Classifier.** XL-LEXEME and one logistic regression per word with C = 1, as in round T, with two
  changes. The class weights are square-root balanced (`classifier.sqrt_balanced`) instead of
  balanced. At labelling time a usage whose cues name exactly one sense takes that sense
  (`classifier.cue_first`), and the regression labels the rest.
- **Training sentences.** Round T's whole-corpus sample, plus *prompt* without the cap in AI's and
  computer science's 2020s cells, the only cells where *prompt* reached it.
- **Evidence used.** Round T's misses, read pair by pair after scoring. A development set of 300
  random usages from the training sample (50 per word), labelled blind by Claude before any
  classifier output was seen. Some cue choices were made after seeing the development set's
  errors. On it, round T's classifier labels 234 of 300 as Claude did and the revised one 252.
  These figures are optimistic and are not the test.
- **Tried and dropped.** Lifting the cap for every word (the cells hold little more), C of 0.25 or
  4 (no difference), and one round of self-training on confident labels (no gain).

### Method

- **Test sentences.** 240 distinct usages, 40 per word (*attention, prompt, platform, network,
  component, having*), drawn uniformly (seed `round-u`) from the training sample's distinct
  sentences after removing boilerplate, check 2b's 186 usages and the development set's 300. They
  are left out of training. Claude reads neither them nor their labels before the author has
  judged.
- **Labellers.** The revised classifier, and Claude Sonnet 5.5 (Batch API, low effort) with the same
  lists.
- **Disagreements.** The usages where the two labels differ. The author sees each on a page with
  the two labels' definitions ("none of the listed meanings" for Sonnet's *other*), in an order
  drawn at random per usage (seeded), without being told which labeller gave which. The author
  chooses the first, the second, both fit, or neither fits. The key from order to labeller is kept
  off the page and read only by the scorer.

### Predictions

**U1 — Where they differ, the classifier is about as good as Sonnet.** Of the disagreements where
the author chooses exactly one label, the classifier's is chosen in at least 40%. *Falsified*
below 40%. *Untestable* if fewer than 20 disagreements are decided that way.

**U2 — Agreement.** The share of the 240 usages on which the two labellers agree, overall and per
word. Descriptive.

**U3 — Rough accuracy.** Counting agreements as right, each labeller's estimated accuracy:
agreements, plus disagreements where its label is chosen or both fit, over 240. Descriptive, and
an upper estimate.

**U4 — Per word.** U1's share for each word. Descriptive; about eight disagreements per word, so
each figure is noisy.

Reading rule: if U1 passes, the classifier with version 2 and these settings is the labeller for
rung 3, and round A gets an amendment, committed before it is run, for the lists, the weights and
cue-first labelling, including how adjusted shares treat cue-decided usages. If U1 fails, Sonnet is
the labeller for rung 3, with an amendment for that. U1 judges labels, not pairs, so it is not
comparable with round T's figures.

Disclosed: the cue additions and the weights were chosen knowing round T's misses. The review pages
and the development set sample the same training sample the test sentences come from, so a few
test sentences may have been shown as cue matches or non-seed samples before.

Scored by `scripts/score_labels_head_to_head.py`, committed before it is run.

### Round U results (2026-11-11, `scripts/score_labels_head_to_head.py`)

Scored as written. The classifier trained on 6,691 (*attention*), 356 (*prompt*), 1,102
(*platform*), 3,391 (*network*), 1,310 (*component*) and 3,519 (*having*) cue-seeded usages. The
two labellers disagreed on 48 of the 240 test usages, and the author judged all 48.

| | Verdict | Numbers |
|---|---|---|
| U1 | **Passed** | classifier chosen in 18 of 41 decided disagreements (43.9%; needed 40%); both fit 3, neither 4 |
| U2 | Descriptive | agreement on 192 of 240 (80.0%): *attention* 33/40, *prompt* 32, *platform* 34, *network* 27, *component* 33, *having* 33 |
| U3 | Descriptive | rough accuracy, agreements counted right: classifier 88.8%, Sonnet 90.8% |
| U4 | Descriptive | classifier chosen per word: *attention* 2/7, *prompt* 1/8, *platform* 2/6, *network* 6/11, *component* 3/3, *having* 4/6 |

By the reading rule the classifier, with `corpus_v2.toml` and square-root weights and cue-first
labelling, is the labeller for rung 3. Round A needs an amendment, committed before it is run, for
the lists, the weights and cue-first labelling.

Not as pre-registered, before the author judged: the page was given to the author as a local file
rather than a published page, because publishing it would have meant Claude reading the test
sentences. Its answers were kept in the browser and pasted back. The page also gained each
sentence's field and decade at the author's request, and a UTF-8 declaration after the author saw
garbled characters. The draw's first run stopped before writing anything, because its folder did not
exist (`2f1b491`). None of these touches the labels or the order-to-labeller key.

Read with care: 41 decided disagreements give a wide interval around 43.9% (roughly 30% to 60%),
so the pass is narrow. U4 shows where the classifier still loses: *prompt*, where Sonnet was chosen
in 7 of 8, which matters for round A's language-model-input prediction (A3).

## Round A: known changes over time, with a planted change and a placebo (committed 2026-11-10, before any corpus list was frozen or any usage of these cells was labelled by the classifier)

Rung 3. The classifier, trained from the frozen corpus lists and cue words, labels round X's
transitions; a cell's meaning is its sense shares, corrected by adjusted classify and count; and a
change is judged against shuffles and control words, as round X judged J. A planted change must be
found, and a placebo that changes a word's source but not its senses must not be.

### Method (fixed now; the lists and cues are fixed at their freezing commits)

- **Lists.** `data/sense_lists/corpus.toml` (the approved targets and check 2b's two controls) and
  the control words' lists from `corpus_draft_controls.toml`, each frozen after the author's review.
- **Transitions and usages.** Round X's transitions and cells: *attention* in AI, 2010s → 2020s;
  *prompt* in AI, 2000s → 2020s (word-targeted cells); *platform* in communication, 2000s →
  2010s; *network* in AI, 1980s → 1990s; *alignment* in AI, 2010s → 2020s (the weak case); and the
  quiet transitions, *attention* in AI, 1990s → 2000s, and in psychology, 2010s → 2020s. Usages
  as round X drew them (round C′'s quality rules, abstracts in both cells dropped, the word with
  its regular plural, one occurrence per abstract, at most 150 abstracts per cell), with seed 2 in
  place of round X's seed 1, so the usages check 1a and check 2b read are not reused as such.
- **Controls.** The 10 words of `data/control_words_rung3.txt` on each transition's grid cells (for
  *prompt*, AI's 2000s and 2020s grid cells), each where it has at least 100 usages in both cells.
- **Classifier.** One logistic regression per word on XL-LEXEME vectors (C = 1, balanced class
  weights), trained on every usage of the word in the whole-corpus sample (every field and decade,
  at most 150 abstracts per cell, seed 1) whose cues name one sense, each distinct sentence once,
  plus each sense's example sentence. Training sentences are not removed from the cells measured:
  their labels come from the cue rules, not from any result.
- **Shares.** Each cell's raw shares (the classifier's labels counted) and adjusted shares
  (adjusted classify and count, with the word's confusion matrix from five-fold out-of-fold
  predictions on its training sentences). Adjusted shares are primary; raw shares are reported.
- **Change, and when it is found.** The Jensen-Shannon divergence between the two cells' adjusted
  shares, in bits. A change is found when it is above (a) the 95th percentile of the same
  divergence over 200 shuffles of the word's labelled usages between the two cells, keeping each
  cell's size, and (b) the 90th percentile of the control words' divergence on that transition.
- **Planted change.** Psychology's *attention*, 2010s → 2020s (a quiet transition): a share of the
  2020s usages (10%, 20%, 30%, 40%, 60%; five draws each, seeded) is replaced by AI 2020s usages that
  carry check 1a's attention-mechanism markers, and the change is judged with the transition's own
  shuffles and controls.
- **Placebo.** Psychology's *years*, 2010s → 2020s: 40% of the 2020s usages are replaced by
  sociology's 2020s usages of *years* (five draws, seeded). *years* is a control word whose senses
  (a duration or age; a period of life or history) are used much alike across fields, so the
  replacement changes the source and leaves the sense mix close to unchanged. Judged with the
  transition's own shuffles and controls (the other nine control words).
- **Comparison labellers.** Claude Sonnet 5.5 and JEV label the target words' transitions (not the
  controls) with the same lists; the same divergence and shuffle bar, reported beside the
  classifier's.
- Already seen: round X's results on these transitions, checks 1a, 1b and 2b, rounds Y to T, and
  the review page's whole-corpus cue matches and checks. Not computed: any classifier label of these
  cells' seed-2 usages.

### Predictions

**A1 — The classifier finds the documented changes.** It finds at least three of *attention*,
*prompt*, *platform* and *network*. *Falsified* with two or fewer.

**A2 — It stays quiet where nothing documented changed.** It finds neither quiet transition.
*Falsified* if it finds either.

**A3 — In the right direction.** In each of the four documented changes, the documented sense's
adjusted share rises from the earlier cell to the later: the attention mechanism, language-model
input, the digital or social-media platform, the artificial neural network. *Falsified* if any
falls or stays within 0.05.

**A4 — It sees a planted change of a third or less.** The planted mechanism is found in at least
four of five draws at some share of 30% or less, and at every larger share. *Falsified* otherwise.

**A5 — It does not see a change of source alone.** The *years* placebo is found in at most one of
five draws. *Falsified* in two or more.

**A6 — Every word drifts a little.** The share of control-word transitions found under bar (a)
alone. Descriptive, as round X's X5.

**A7 — The weak case.** *alignment* in AI, 2010s → 2020s, with the AI-alignment share in each
cell. Descriptive.

**A8 — The comparison labellers.** Which documented changes and quiet transitions Sonnet and JEV
find under bar (a). Descriptive.

Reading rule: the instrument passes rung 3 if A1, A2, A3, A4 and A5 pass, and goes on to rung 4
(between fields, then open questions). If any fails, it stops before open questions, and the
failure is reported against round X's (J found the four documented changes but also a quiet one)
and round 5's (the grid found one of four).

Scored by `scripts/score_labels_known.py`, committed before it is run.

### Amendment to rounds T and A: text artefacts and boilerplate (committed 2026-11-10, before either round was run)

Found while the author reviewed the corpus lists. Neither changes a prediction or a bar.

- **Text artefacts.** Every corpus usage is tidied before it is encoded or labelled
  (`interpretant.drift.classifier.tidy_usage`): literal `\n` sequences, line breaks, tabs, control
  and invisible characters and the replacement character become spaces, soft hyphens rejoin the word
  they split, common mojibake is repaired, runs of spaces collapse, and the target's span is rebuilt.
  In the review sample these affected under 1% of usages. LaTeX is left as it is.
- **Boilerplate.** A sentence found in five or more usages of the same word in a sample is not an
  abstract's own text (a publisher's notice, an affiliation, a journal series line: in the review
  sample, seven sentences carried 264 usages, one alone 174). Such sentences are left out of the
  training sentences and, in round A, out of the cells measured.

### Amendment to round A: the alignment senses (committed 2026-11-10, before round A was run)

In the second review pass the author merged *alignment*'s AI-alignment sense into strategic
agreement, as `goal_or_value_alignment` ("bringing goals, strategies, behaviour or values into
agreement, of people and organisations, or of AI systems with human intentions and values"): the
classifier could not separate the two (the AI sense had 33 training sentences, 55% of them
doubtful), and merged they reach 14% doubtful. A7 therefore reports the merged sense's share in
AI's 2010s and 2020s cells, not an AI-alignment share; it remains descriptive and decides nothing.
The AI-alignment sense on its own is not measured by this instrument.

### Amendment to rounds T and A: exclusion cues and out-of-scope usages (committed 2026-11-10, before either round was run)

Found in the second review pass. Neither changes a prediction or a bar.

- **Exclusion cues.** A cue written `!word` silences its sense: when `word` appears within five
  words of the target, that sense's cues do not count, so the usage trains no sense through them
  (`interpretant.drift.classifier.cue_sense`). The lists use four: `!class` and `!tv` under
  *platform*'s `physical_structure` (a work-station class platform, a satellite TV service) and
  `!regions` and `!circuits` under *network*'s `knowledge_or_probabilistic_model` (two
  neuroscience usages where "the semantic network" is a set of brain regions). In the review sample
  they remove exactly these four training sentences and no other.
- **Out-of-scope usages.** A word's list may name `out_of_scope` words; a usage with one of them
  within five words of the target is not trained on, labelled or counted
  (`interpretant.drift.classifier.out_of_scope`). *Alignment* names the terms of grammatical
  alignment (ergative, accusative, absolutive, nominative, morphosyntactic, ergativity). In the
  review sample this is 8 of 2,197 *alignment* usages, all in linguistics (5% of that field's 164).
  Tried first as a fourth sense, it was dropped: from six training sentences the classifier learned
  "language context" and took 17 word-alignment sentences from `matching_or_correspondence`. The
  rule applies the same way in every decade, so it cannot make or hide a change over time. None of
  check 2b's 186 usages is out of scope, so round T's test set is unchanged; in round A the rule
  applies to the cells measured as boilerplate does.

### Note on a renamed sense (2026-11-11, before rounds T and A were run)

In the review of what the classifier does with usages no cue names, *network*'s
`knowledge_or_probabilistic_model` was widened to networks of concepts, beliefs, rules or signs
(an AI knowledge representation or an abstract structure of ideas) and renamed
`knowledge_or_conceptual_network`; the amendment above refers to it by its old name. No
prediction names it.

### Amendment to round A: the labeller after round U (committed 2026-11-11, before round A is run)

Round T falsified the classifier's two predictions and round U passed the revised one (U1, 43.9%
against 40%), so by round U's reading rule the revised classifier labels rung 3. This amendment
says what that changes in round A. No prediction, bar, transition, cell or seed changes.

- **Lists.** `data/sense_lists/corpus_v2.toml` for the target words and check 2b's two controls,
  in place of `corpus.toml`. The control words' lists come from `corpus_draft_controls.toml`,
  frozen after the author's review, as before.
- **Classifier.** For every word, targets and controls alike: XL-LEXEME and one logistic regression
  with C = 1 and square-root balanced class weights (`classifier.sqrt_balanced`), trained as round
  A says, plus *prompt* without the cap in AI's and computer science's 2020s cells, as in round U.
  Each usage is labelled cue first (`classifier.cue_first`): a usage whose cues name exactly one
  sense takes that sense, and the regression labels the rest.
- **Adjusted shares.** A cell's cue-decided usages are counted as labelled. Its regression-decided
  usages are corrected by adjusted classify and count, with the word's confusion matrix from
  five-fold out-of-fold predictions on its training sentences, as before. The cell's adjusted
  shares are the two parts added and divided by the cell's usages. Raw shares are reported beside.
- **Comparison labellers.** Unchanged: Sonnet and JEV label the target words' transitions with the
  same lists. Version 2 changes cue words only, so their definitions are version 1's.

Known risks, stated before the run so the results cannot be explained by them afterwards:

1. ***prompt*.** In round U the author preferred Sonnet in 7 of 8 *prompt* disagreements. Two were
   1980s interface prompts ("menus and prompts", "prompt cards") that the classifier labelled
   language-model input, three confused the noun, the verb and the adjective, and one missed a
   language-model prompt whose cue was too far away. In AI's 2000s cell, interface or dialogue
   prompts read as language-model input would raise the earlier share and shrink the rise A3
   predicts. Adjusted shares do not undo this, because the confusion matrix comes from cue-labelled
   sentences.
2. **Version 2's adjective cues for *prompt*.** `>>recognition`, `>>identification`,
   `>>detection` and `>>access` also fire on the verb with an object ("situations which prompt
   recognition of"). Cue-first labelling then labels those usages *quick or immediate*.
3. **Thin senses.** Square-root weights still favour thin senses somewhat, and the confusion matrix
   comes from cue-labelled sentences, which are easier than the usages measured. Thin senses'
   levels are therefore likely overstated. Changes are less affected where both cells have similar
   text quality.
4. ***network*.** Usages that fit none of its six senses, such as vascular, flow, gene and
   citation networks, are forced into one. About 40 of the whole-corpus sample's AI usages are of
   this kind, across all decades.

### Round A results (2026-11-11, `scripts/score_labels_known.py`)

Scored as written, with the amendment for the labeller after round U. 20,356 usages were measured
(after boilerplate and scope rules, 150 or nearly so per cell). Every one of the six transitions
had all 10 control words with at least 100 usages in both cells.

| | Verdict | Numbers |
|---|---|---|
| A1 | **Falsified** | 2 of 4 documented changes found: *attention* (JSD 0.108 bits; shuffles 0.025, controls 0.054) and *platform* (0.024; 0.024, 0.008). *prompt* (0.097; 0.036, 0.107) and *network* (0.048; 0.027, 0.059) beat their shuffles but not their controls |
| A2 | **Passed** | neither quiet transition found: *attention* in AI 1990s→2000s 0.018 bits, in psychology 2010s→2020s 0.001 |
| A3 | **Passed** | adjusted shares rise by more than 0.05 in all four: attention mechanism 0.175 → 0.518, language-model input 0.456 → 0.802, digital platform 0.673 → 0.783, artificial neural network 0.610 → 0.811 |
| A4 | **Passed** | planted mechanism found in 3/5 draws at 10% and 5/5 at 20%, 30%, 40% and 60% (291 marked usages in the pool) |
| A5 | **Passed** | the *years* placebo found in 0 of 5 draws |
| A6 | Descriptive | 12 of 60 control-word transitions (20%) beat their own shuffles |
| A7 | Descriptive | *alignment* in AI: matching 1.000 → 0.583, goal or value alignment 0.000 → 0.417; found under both bars |
| A8 | Descriptive | Sonnet and JEV, raw shares under bar (a): each finds all four documented changes and neither quiet transition |

By the reading rule the instrument does not pass rung 3, because A1 fails. By the plan a failed
rung 3 stops the instrument before open questions, with no repair round.

Exploratory, after scoring:

- **The first known risk happened.** The classifier labels 66 of AI's 150 2000s *prompt* usages
  language-model input, 63 of them by the regression. Sonnet labels 34 of those 66 a cue to a
  person and 23 other, and only 5 of all 150 language-model input. Read, they are spoken-dialogue
  system prompts ("a well-crafted prompt", "the system prompts for information") and the ontology
  tool PROMPT. The earlier share (0.456 adjusted) is mostly this error, which shrinks the measured
  change below the controls' bar.
- **Under bar (a) alone the classifier matches the language models**: all four documented changes
  and neither quiet transition, as A8 reports for Sonnet and JEV. A1 fails on bar (b). The controls'
  90th percentile is highest on AI's 2000s→2020s (0.107), the only two-decade transition. A8 does
  not apply bar (b) to the language models, because they did not label the control words.
- **The placebo and the planted change**, the tests the usage-pair measure failed, both pass: a
  change of source alone is never found, and a planted change of a fifth always is.

### Note on names and file paths (2026-11-11, after round A was scored)

Two corrections to how this file names things. Neither changes a prediction, a verdict or what was
run.

- **Two rounds U.** The heading "Round U: the revised classifier against Sonnet, judged blind by the
  author" reuses a letter: `predictions_pairs.md` already has a round U, the usage-pair
  instrument's first rung. Elsewhere this round is called the head-to-head round (H). The heading
  above is left as committed.
- **Draft names for frozen lists.** Round T names `data/sense_lists/corpus_draft.toml` and round A's
  amendment names `corpus_draft_controls.toml`. Both rounds read the frozen copies:
  `corpus.toml` (`9332d86`, then `corpus_v2.toml`, `0b3bfa3`, after round T) and
  `corpus_controls.toml` (`359c858`), as their scorers show.

### Note on the head-to-head round's name (2026-11-11)

The note above calls the head-to-head round "(H)", but H already names a cross-perplexity round
(`predictions_cross_perplexity.md`). It is called the head-to-head round, without a letter.
