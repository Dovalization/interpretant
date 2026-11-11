# Two labellers compared: Claude Sonnet 5.5 and JEV

The sense-label instrument ([`plan_sense_labels.md`](plan_sense_labels.md)) gives each usage of a
word one sense from a fixed list. On the benchmark two language models did the labelling, as
separate instruments; from round E a cue-seeded classifier with no generative model labels this
corpus and the two language models run beside it as comparisons. This document gathers what the
runs showed about all three, from the records in [`plan_sense_labels.md`](plan_sense_labels.md)
(pilot, API recheck) and [`predictions_labels.md`](predictions_labels.md) (rounds Y, Z, Q, E, T, the
head-to-head round and A). Numbers marked *descriptive* were computed after scoring and decide
nothing.

## The two labellers

| | Claude Sonnet 5.5 | JEV 1.14 |
|---|---|---|
| Kind | A general language model (Anthropic) | A decision model (TypeSafe AI): returns probabilities over allowed answers, no free text |
| Access | Message Batches API, one usage per request, low effort | TypeSafe API, one usage per request, pinned at `jev-1.14.0` |
| What it is asked | The word, its sense list with definitions and examples, the usage; a JSON schema allows one label from the list, *other* or *unclear* | A Choice among the listed senses (definitions and examples as criteria), plus a companion yes-or-no question on whether any listed sense fits |
| How a label is read | The answer itself | The top sense when the companion question is above 0.7, *other* below 0.3, *unclear* between |
| Repeatability | No sampling temperature; repeats measured (98–99% the same) | Pinned version; repeats measured (97% the same) |
| Price | $1 / $5 per million input / output tokens at batch prices | $0.042 per million input tokens, output free |

The companion question follows TypeSafe's own guidance, which recommends it over a "none of
these" option. It also explains the largest difference below.

## Results, run by run

| Run | Measure | Sonnet 5.5 | JEV 1.14 |
|---|---|---|---|
| Pilot (186 usages of check 2b, author's pair judgements) | Spearman with the author | 0.501 (CLI blocks), 0.510 (API) | 0.429 (labels), 0.479 (probabilities) |
| Round Y (DWUG EN, 8,787 usages, lists written by Claude) | Spearman with DWUG's graded change | 0.767 | 0.821 |
| | Adjusted Rand index with DWUG's clusters | 0.287 | 0.267 |
| | Repeats labelled the same | 98.9% | 96.8% |
| | *other* / *unclear* | 8.4% / 0.5% | 1.5% / 7.1% |
| Round Z (same usages, lists from WordNet) | Spearman with DWUG's graded change | 0.588 | 0.635 |
| | Drop from round Y | −0.180 | −0.187 |
| | Adjusted Rand index | 0.235 | 0.220 |
| | Repeats labelled the same | 98.1% | 96.6% |

Reference figures on DWUG EN: BERT 0.563 and XL-LEXEME 0.864 (round U) for ranking the changes;
XL-LEXEME's clustering 0.273, GPT-4 0.340 and human annotators 0.633 for the Rand index (Periti
and Tahmasebi 2024).

## How far they agree

| Run | Same label | Kappa | *other* and *unclear* as one answer | Same sense when both pick a listed one |
|---|---|---|---|---|
| Pilot (186 usages) | 89% | 0.883 | | |
| API recheck (186 usages) | 90% | 0.895 | | |
| Round Y (8,787) | 84.5% | 0.843 | 88.1% (kappa 0.879) | 94.9% of 7,695 |
| Round Z (8,787) | 74.8% | 0.743 | 81.8% (kappa 0.809) | 90.6% of 6,642 |

The last two columns are descriptive. Seven in ten of the round Y disagreements (71%) involve
*other* or *unclear*. When no listed sense fits a usage, Sonnet says *other* and JEV's companion
question tends to land in its middle band, which reads as *unclear*. Counting the two as one answer,
the labellers agree on 88% of usages; when both pick a sense from the list, they pick the same one
95% of the time. What separates them is mostly whether a listed sense fits, not which one.

The words they agree on least in round Y are *heel* (54%), *twist*, *quilt* (62% each), *word*
(64%) and *tip* (69%): words with several concrete senses close to each other, or a list that missed
a common use.

## What the comparison shows so far

- **They rank the benchmark's changes about equally well.** JEV ranks a little better with the
  written lists (0.821 against 0.767), Sonnet groups usages a little better (0.287 against 0.267);
  with 46 words, neither difference is large.
- **Neither is more robust to a worse list.** With WordNet's lists both dropped by the same amount
  (0.18 and 0.19) and both found no fitting sense for about one usage in five. The dependence
  belongs to the lists, not to a labeller.
- **They differ in how they say "no fit".** Sonnet uses *other*; JEV, through its companion
  question, uses *unclear*. A measure that leaves out *unclear* and keeps *other*, as the
  divergence does, treats the two differently; worth remembering when reading coverage figures.
- **Clear errors in the pilot were more often JEV's** (a *neural network* read as an abstract
  system, a dialogue system's *prompts* read as language-model input); on the benchmark the two
  are close enough that the pilot's sample says little.
- **JEV is about 27 times cheaper and pinned.** Round Y cost $6.49 on Sonnet (5.77 million input
  and 144,000 output tokens) and $0.24 on JEV (5.73 million input tokens); round Z $6.60 and
  $0.24. For the open-ended survey the project ends with, which could label every occurrence in
  the corpus, that difference matters more than the small differences in accuracy.

Both labellers stopped at rung 1 in round Y, on a variety check that DWUG's own clusters fail
(see the correction in [`predictions_labels.md`](predictions_labels.md)). Round Q re-tested them
with sense lists induced from the texts: both fell short of its bars (Spearman 0.443 for Sonnet and
0.502 for JEV, needed 0.70), well below the same labellers with round Y's hand-written lists.

## A third labeller with no generative model (round E)

A logistic regression on XL-LEXEME's vectors, trained per word from cue-word seeds and the lists'
example sentences, labelled round Q's half B with round Y's lists: Spearman 0.669 (Sonnet 0.693,
JEV 0.800 on the same usages), partial 0.509 (Sonnet 0.646, JEV 0.667), adjusted Rand index 0.266
(Sonnet 0.330, JEV 0.302). It gives the same label as Sonnet on 75% of usages and as JEV on 76%.
It costs nothing to run, needs no labelling by a person, and every label traces back to a cue
word or an example sentence; it passed rung 1 and became the labeller for this corpus, with the
two language models as comparison points.

## On this corpus (round T, the head-to-head round and round A)

- **Round T (rung 2).** Scored against the author's 100 pair judgements, the classifier fell just
  short of both bars: 0.449 for probability overlap (needed 0.456) and 0.397 for same sense against
  Sonnet's 0.514 (needed 0.414); JEV scored 0.451. It gave the same label as Sonnet and as JEV on
  76.3% of the 186 usages. By the reading rule a language model became the fallback.
- **The head-to-head round (headed "Round U" in the predictions file).** A revised classifier
  (square-root class weights, cue-first labelling, list version 2) and Sonnet labelled 240 fresh
  usages and agreed on 192 (80%). The author judged the 48 disagreements blind and chose the
  classifier's label in 18 of the 41 decided (43.9%, needed 40%): narrowly enough to go on. It lost
  7 of 8 on *prompt*.
- **Round A (rung 3).** On round X's transitions, the classifier found 2 of 4 documented changes
  against the control-word bar. Against shuffles alone it found all four and neither quiet
  transition, as Sonnet and JEV did with raw shares (A8). The clearest gap is *prompt*: the
  classifier labels 44% of AI's 2000s usages language-model input, mostly spoken-dialogue system
  prompts, where Sonnet labels 3%.

## Practical notes

- JEV needs the senses' definitions in the Choice's criteria; option names alone are not enough
  for distinctions like *notice* against *cognitive attention*.
- Sonnet through the Claude Code CLI, in blocks of 20 to 40 usages, gave nearly the same labels
  as one usage per request through the API (95% agreement, kappa 0.942), but counts against a
  subscription's usage limits; the Batch API does not.
- Both ran without a failed or refused request in about 39,000 labels to round Q, and again in
  round T, the head-to-head round and round A (4,692 more).
