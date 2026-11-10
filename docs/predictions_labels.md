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
