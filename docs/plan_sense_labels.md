# Plan: meaning by sense labels (ROADMAP item 8d)

Drafted 2026-11-10, after the usage-pair design stopped at rung 2 (round X) and the diagnostics
in [`diagnostics.md`](diagnostics.md). This is a plan, not a pre-registration: each round below
gets its predictions committed in `docs/predictions_labels.md` before it is run, as every earlier
round did.

**State (2026-11-11).** The instrument stopped at rung 3. Rung 1 passed (rounds Y, Z and Q with the
language models; round E with the cue-seeded classifier, which became the labeller for this
corpus). For rungs 2 and 3 Claude drafted the sense lists and cue words and the author reviewed and
approved them on review pages; they are frozen as `data/sense_lists/corpus.toml`, then
`corpus_v2.toml` (cue words added after round T), with `corpus_controls.toml` for rung 3's control
words. Rung 2 failed as round T and passed narrowly on a second attempt with fresh sentences, the
head-to-head round (headed "Round U" in `predictions_labels.md`). Rung 3, round A, found 2 of the 4
documented changes against the control-word bar (A1 falsified) and passed A2 to A5, including the
placebo and the planted change. Rung 4 has not run. Results and every amendment are in
`predictions_labels.md`; the labellers are compared in [`labellers.md`](labellers.md).

## Why this instrument

The diagnostics left one problem standing. The dataset holds the documented changes (check 1a),
the grid's embeddings work on clean data (round 6), and XL-LEXEME reads this corpus's sentences
about as well as two judges agree with each other (check 2b′). But the usage-pair measure answers
"did the contexts change?", not "did the meaning change?" (check 2b and round X's X5; A7),
and this corpus's contexts change a great deal between decades (A2). Every measure built on
context similarity inherits that.

One reading in the diagnostics did not: check 1a's blind categorical labels, one usage at a time,
"is this the attention mechanism, yes or no?", which got every documented change and every quiet
transition right. This plan turns that reading into an instrument: a language model labels each
usage with one sense from a fixed list written for that word, and a cell's meaning is its
distribution over that list. A change of context without a change of sense should then leave the
distribution where it was, which is the property the pair measure lacks (check 1b's placebo was
first read as showing this directly; see the correction in `diagnostics.md`).

What it gives up: it finds only the senses on the list. A rise in the *other* label is the only
way it can see a sense nobody wrote down, and that is weaker than discovery.

## The instrument

- **Sense lists.** One per word, 2 to 4 senses, each a one-line definition with one invented
  example, plus *other* (a sense not on the list) and *unclear* (the context cannot tell). For a
  documented change the changed sense is one entry, defined as in check 1a. The lists are written
  by Claude before any usage they will label is read. For this corpus the author reviews and
  approves them, then they are committed and frozen; for the benchmark they are not reviewed.
  Writing a fixed list answers the granularity problem the check 2b correction pass found.
- **Labelling: two labellers, each its own instrument** (decided 2026-11-10, after the pilot;
  superseded from round E, after which the cue-seeded classifier labels this corpus and the two
  language models are comparison baselines and the pre-registered fallback; see "Measuring shares
  on this corpus" below).
  Claude Sonnet 5.5 at low effort through the Message Batches API, and JEV (TypeSafe AI,
  pinned at `jev-1.14.0`) through its API. Both label one usage per request: the word, its sense
  list, and the usage's sentence (cut to 100 words, the word marked as in check 2b), with nothing
  about field, decade or cell, the answer constrained to one label from the list (for Sonnet a
  JSON schema, for JEV a Choice with a companion yes-or-no question, labelled as in the pilot).
  Every rung's predictions apply to each labeller separately: a labeller that fails a rung stops
  there, whatever the other does, and a result counts only for the labeller that produced it, so
  the second labeller can never stand in for a failed first. How far the two agree is reported at
  every rung as a finding of its own. The pilot's Sonnet labels came from the CLI in blocks; one
  usage per request removes any effect of a usage's neighbours.
- **Measures.** A cell's distribution is its share of each listed sense plus *other*; *unclear*
  is reported and left out. The change between two cells is the Jensen-Shannon divergence of
  their distributions, in bits. It is the measure DWUG's own graded change uses on its human
  sense clusters, so rung 1 compares like with like. For a documented change the changed
  sense's share is also reported.
- **Bars.** As round X: a change is found when the divergence is above the 95th percentile of
  200 shuffles of the word's labelled usages between the two cells (sampling noise) and above the
  90th percentile of the control words' divergence on the same transition (every word's drift).
- **Models.** Fixed before rung 1 and never changed, so that a failed rung cannot be repaired by
  changing model. Sonnet takes no sampling temperature, so its labels are not deterministic: 10%
  of every run is labelled twice, as separate requests, and the agreement is reported as its
  self-consistency (JEV's repeats are reported too). A refused or failed request is recorded as
  such and counted with *unclear*; it is not sent to another model, which would mix two
  instruments. Model IDs, the date and every request's token usage are recorded with every run.
- **Records.** Raw responses stay local. Labels are committed as identifiers and labels only; no
  DWUG text is committed (its licence forbids redistribution).

## Pilot: Sonnet 5.5 and JEV on check 2b's usages (before rung 1)

A small comparison of two labellers, run before any rung, on usages whose judgements already
exist and which no rung uses as evidence: the 186 distinct usages in check 2b's 100 pairs
(*attention, prompt, platform, network, component, having*), which the author judged pair by pair.

- **Labellers.** Claude Sonnet 5.5 at low effort through the CLI (the route planned then), in
  blocks of 20 with 10% repeats. JEV (TypeSafe AI's decision model, pinned at `jev-1.14.0`), one request per usage:
  the usage as state, a Choice among the listed senses with each sense's definition and example as
  its criterion, and, as TypeSafe's guidance recommends instead of a "none of these" option, a
  companion yes-or-no question asking whether the marked word has one of the listed senses. JEV's
  label is its top choice when that probability is above 0.7, *other* below 0.3, and *unclear*
  between, the guidance's starting thresholds.
- **Sense lists.** Drafts by Claude for the six words, not reviewed by the author; the pilot
  compares labellers, not lists. The lists the ladder uses are written and reviewed afterwards.
- **What is compared.** Agreement between the two labellers (Cohen's kappa, exact share). For each
  labeller, Spearman between "the pair's two usages share a listed sense" and the author's
  judgement over check 2b's 95 distinct judged pairs, with the original judgements and with the
  correction pass, beside XL-LEXEME (0.456 and 0.494) and the second judge (0.616 and 0.657); for
  JEV also the overlap of the two usages' sense probabilities. Sonnet's self-consistency on the
  repeats. Time and tokens per usage for each.
- **What it decides.** Nothing is predicted. The numbers go to the author, who chooses the ladder's
  labeller before rung 1. (Superseded the same day: both labellers run, as separate instruments.)
- **Data sent.** Short excerpts of public OpenAlex abstracts, to Anthropic through the CLI and to
  TypeSafe through its API; no personal data. TypeSafe's privacy policy says it does not train on
  user data; its retention period is not stated in its public documentation.

### Pilot results (2026-11-10, `scripts/pilot_labellers.py`)

Run as written, except that JEV first labelled 10 usages as a test of the calls (the same usages
were labelled again in the full run). The author's judgements count 96 pairs here, not 95: check
2b′ also left out a pair the second judge could not decide.

| | Sonnet 5.5 | JEV 1.14 |
|---|---|---|
| Spearman with the author, original judgements (96 pairs) | 0.501 | 0.429 (probabilities 0.479) |
| Spearman with the author, with the correction pass | 0.545 | 0.474 (probabilities 0.548) |
| *other* / *unclear* | 2% / 1% | 1% / 4% |
| Consistency on repeats | 94% (17 of 18) | not measured |
| Time and tokens | 15 s, 37,180 input and 4,698 output tokens (CLI-reported $0.196) | 36 s, 125,184 input tokens (about $0.005) |

Reference figures over the same pairs: XL-LEXEME 0.456 and 0.494, the second judge 0.616 and
0.657. The labellers agreed on 89% of the 186 usages (kappa 0.883), on every *attention* usage
and on 24 to 28 of each other word's 30 to 32.

Read from the 21 disagreements: most fall on boundaries the draft lists drew badly (*component*
as a part of a system or of an abstract whole; *platform* for media delivery channels, which no
listed sense describes; *having to*, obligation, not listed). Where one labeller was plainly
wrong, it was more often JEV (a *neural network* read as an abstract system, a 2000s dialogue
system's *prompts* read as language-model input, *prompt health tracking* and *without having
access* left unclear); Sonnet left unclear a *network* "studied by Amit et al." that JEV read
correctly as a neural network. With 96 pairs and a two-valued measure, the difference between the
labellers' correlations is within noise; both sit near XL-LEXEME and below the second judge on
this proxy, which scores whether two usages share a label, not whether a label is right. Rung 2's
gold labels test that directly.

### API recheck (2026-11-10, `scripts/pilot_labellers.py sonnet_api` and `recheck`)

The pilot's 186 usages, plus 19 repeats, labelled by Sonnet 5.5 through the Batch API, one usage
per request: all 205 requests succeeded in about 3 minutes, 141,061 input and 3,547 output tokens
(about $0.16 at batch prices, 688 input tokens a request). The labels agree with the pilot's CLI
blocks on 95% of usages (kappa 0.942) and with JEV on 90% (kappa 0.895); all 19 repeats got the
same label; *other* 4%, *unclear* 1%. Spearman with the author 0.510 (original judgements) and
0.552 (with the correction pass), against the CLI blocks' 0.501 and 0.545. One usage per request
changes the labels little. At the measured tokens the ladder costs about $28 on Sonnet.

## Measuring shares on this corpus (added 2026-11-10, with the author)

From round E on, a word's senses on this corpus are labelled by the cue-seeded classifier, and a
cell's meaning is the share of the word's occurrences in each sense, never a count of cue matches
(cue vocabulary changes over time on its own, and cells differ in size).

- **Labels.** The classifier labels every sampled occurrence, with or without a cue word; the cue
  matches are only its training material.
- **Adjusted classify and count.** Raw shares from a classifier are biased towards the senses it
  over-predicts. Each word's confusion matrix is estimated from the out-of-fold predictions on its
  cue-labelled training sentences (the review page's check), and a cell's shares are corrected by
  non-negative least squares (`interpretant.drift.classifier.adjusted_shares`). Assumptions,
  stated with every result: the classifier confuses senses the same way in every field and decade,
  and the cue labels the matrix is estimated from are close to correct once the author's review
  has cleaned them. Raw and adjusted shares are both reported; which one a prediction uses is fixed
  in its round.
- **Review loop.** Cue words are refined with the author on the review page, guided by the
  leave-one-cue-out agreement and the out-of-fold doubt of each training sentence, and only by
  those: no known answer, gold judgement or result is looked at while the cues are tuned. The lists
  and cues are frozen and committed before the classifier labels anything that is scored.
- **Rules added during the review** (amendments in `predictions_labels.md`). Usages are tidied
  before anything else (`tidy_usage`), and sentences repeated across five or more abstracts are
  dropped as boilerplate. A cue written `!word` is an exclusion that silences its sense's cues
  nearby, and a list may name out-of-scope words that remove a usage altogether (*alignment*'s
  grammatical sense).
- **The labeller after round T** (from the head-to-head round on). Class weights are square-root
  balanced (`sqrt_balanced`) instead of balanced, so thin senses pull uncertain usages to
  themselves less, and labelling is cue first (`cue_first`): a usage whose cues name exactly one
  sense takes that sense, and the regression labels the rest. In round A, adjusted shares correct
  only the regression's labels; cue-decided usages are counted as labelled.

## The method in NLP terms (added 2026-11-10)

In NLP terms, the instrument on this corpus is weakly supervised word sense disambiguation, used to
estimate sense prevalence per field and decade and to detect changes in it.

| Step | What it does here | NLP name |
|---|---|---|
| Sense lists | a fixed, coarse list of senses per word, approved by the author | sense inventory (WordNet-style, curated for this corpus) |
| Cue words | position-aware collocation rules that label some occurrences | labelling functions; weak or distant supervision (Ratner et al. 2017) |
| Example sentence per sense | one gloss example per sense, added to the training data | gloss-based supervision |
| XL-LEXEME vectors | the target word's meaning in context, from an encoder trained on same-meaning judgements | contextual embeddings from a word-in-context model (Pilehvar and Camacho-Collados 2019; Cassotti et al. 2023) |
| Classifier | a logistic regression per word on frozen embeddings | supervised WSD with a linear probe |
| Out-of-fold doubt | flags training labels that a model which never saw them disagrees with | label-noise detection, confident learning (Northcutt et al. 2021) |
| Leave-one-cue-out | scores each cue by whether a model trained without its sentences agrees | labelling-function analysis |
| Author's review | cues refined with the checks as a guide | human-in-the-loop weak supervision |
| Shares per cell | sense proportions, corrected for the classifier's confusions | quantification: adjusted classify and count, under prior shift (Forman 2008) |
| Change measure | Jensen-Shannon divergence between two cells' sense distributions, against shuffles and control words | sense-prevalence-based lexical semantic change detection; DWUG's graded change (Schlechtweg et al. 2021) |
| Placebo | same-sense occurrences from another field must not register as a change | a test of the context-for-meaning confound |
| DWUG EN | rung 1 against expert scores | benchmark validation (SemEval-2020 Task 1) |

Sonnet 5.5 and JEV, as labellers, are zero-shot WSD with generative or decision models; from round E
on they are comparison baselines, not the method.

## The ladder

Each rung has its own pre-registration and stopping rule. A failed rung 1 or 2 stops the
instrument; a failed rung 3 stops it before open questions. No repair rounds, as for items 8,
8a and 8c, with one exception the author made after round T: a revised classifier got one new round
on fresh sentences, never on round T's pairs (the head-to-head round).

1. **Benchmark (DWUG EN 2.0.1, 46 words, 9,153 usages).** Sense lists written blind to DWUG's
   clusters, change scores and usages. Primary: Spearman between the label divergence and DWUG's
   graded change, beside XL-LEXEME's APD (0.864 here, 0.886 published) and BERT's 0.563.
   Secondary: adjusted Rand index between labels and DWUG's clusters, beside the published
   GPT-4 (0.340), XL-LEXEME (0.273) and human (0.633) figures. Caveat: the SemEval-2020 answers
   are public and a model may have seen them. The labeller sees one usage at a time with no
   period, so it can recall senses but not which period a usage came from; the list writer could
   still favour senses it knows changed, and that is stated with the result.
2. **This corpus, as the author reads it.** Planned as about 130 usages labelled by the author
   and scored by Cohen's kappa; run instead without new labelling. Round T scored the classifier's
   labels of check 2b's 186 usages against the author's 100 pair judgements (falsified: 0.449
   against 0.456, and 0.397 against Sonnet's 0.514 minus 0.10). The head-to-head round then had
   the revised classifier and Sonnet label 240 fresh usages, and the author judged their 48
   disagreements blind (passed: the classifier chosen in 18 of 41 decided, 43.9% against 40%).
3. **Known answers over time, with a placebo.** Round X's transitions, cells and sampling (seed
   changed, so no usage check 1a read is reused): the five documented changes, the two quiet
   transitions, and 10 control words drawn at random (seed 2026) from round C's 30, among those
   with at least 100 abstracts in every transition's cells (as run: four documented changes, with
   *alignment* as a descriptive weak case). Plus check 1b's planting design on
   psychology's 2020s *attention*: mechanism usages (check 1a's markers) planted at the same five
   shares; and, decided with the author on 2026-11-10, a placebo that keeps the sense mix fixed:
   sociology's 2020s *years* planted into psychology's (check 1b's own placebo also changed the
   sense mix; see the correction in `diagnostics.md`). Round A pre-registered and ran it: 2 of 4
   documented changes found against the control-word bar (all 4 against shuffles alone), neither
   quiet transition, the planted change from 20%, and the placebo in none of 5 draws. The placebo
   is the test the pair measure failed, and it carries the most weight.
4. **Between fields, then open questions.** *embodiment* between HCI and philosophy, and only
   after that the questions the project started from: field-specific meanings, and one-sided
   changes such as the language-model sense of *hallucination* reaching psychology.

## Volume and cost

Historical: this estimate assumed a language model labels every rung. From round E the classifier
labels this corpus, and the language models ran as comparisons only: rung 2's 186 and 240 usages,
and rung 3's 2,040 target-word usages each for Sonnet and JEV.

Labels: rung 1 about 10,000 (with the 10% repeats); rung 3 at most about 26,500 (2,100 for the
target words, at most 21,000 for the controls, about 900 planting usages, plus repeats); rung 2
is negligible. About 36,500 in all.

Sonnet 5.5 at batch prices ($1 and $5 per million input and output tokens, half the standard
price, from Anthropic's API reference as of 3 November 2026), one usage per request at about 550
input and 30 output tokens, comes to about $0.0007 a label, so about $25 for the whole ladder,
billed against the author's API allowance ($100). JEV, at $0.042 per million input tokens with
output free and about 670 input tokens a request (measured in the pilot), comes to about $1.
Before rung 1, the pilot's 186 usages are labelled again by Sonnet through the API, about $0.15,
to check that one-usage requests agree with the pilot's blocks.

## Decisions for the author

1. ~~The labeller~~ Decided 2026-11-10: both, as separate instruments (Sonnet 5.5 over Opus 5.5
   to save usage; JEV as a cheap decision model whose comparison with Sonnet is documented).
2. **The author's time:** approving about 15 sense lists (5 target words, 10 controls) and
   labelling about 130 usages, roughly as long as check 2b's 100 pairs. As it happened: approving
   17 lists (7 words from check 2b and the targets, 10 controls) on review pages, and judging 48
   blind disagreements in the head-to-head round; no usage was labelled by the author.

## Not part of it

Clustering, the grid, the usage-pair measures (except as reference figures), and the direction
of HCI's change.
