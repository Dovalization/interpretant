# Plan: meaning by sense labels (ROADMAP item 8d)

Drafted 2026-11-10, after the usage-pair design stopped at rung 2 (round X) and the diagnostics
in [`diagnostics.md`](diagnostics.md). This is a plan, not a pre-registration: each round below
gets its predictions committed in `docs/predictions_labels.md` before it is run, as every earlier
round did.

## Why this instrument

The diagnostics left one problem standing. The dataset holds the documented changes (check 1a),
the grid's embeddings work on clean data (round 6), and XL-LEXEME reads this corpus's sentences
about as well as two judges agree with each other (check 2b′). But the usage-pair measure answers
"did the contexts change?", not "did the meaning change?" (check 1b's placebo, assumption A7),
and this corpus's contexts change a great deal between decades (A2). Every measure built on
context similarity inherits that.

One reading in the diagnostics did not: check 1a's blind categorical labels, one usage at a time,
"is this the attention mechanism, yes or no?", which got every documented change and every quiet
transition right. This plan turns that reading into an instrument: a language model labels each
usage with one sense from a fixed list written for that word, and a cell's meaning is its
distribution over that list. A change of context without a change of sense should then leave the
distribution where it was, which is the property check 1b's placebo showed the pair measure lacks.

What it gives up: it finds only the senses on the list. A rise in the *other* label is the only
way it can see a sense nobody wrote down, and that is weaker than discovery.

## The instrument

- **Sense lists.** One per word, 2 to 4 senses, each a one-line definition with one invented
  example, plus *other* (a sense not on the list) and *unclear* (the context cannot tell). For a
  documented change the changed sense is one entry, defined as in check 1a. The lists are written
  by Claude before any usage they will label is read. For this corpus the author reviews and
  approves them, then they are committed and frozen; for the benchmark they are not reviewed.
  Writing a fixed list answers the granularity problem the check 2b correction pass found.
- **Labelling.** One request per usage. The request shows the word, its sense list and the
  usage's sentence (cut to 100 words, the word marked as in check 2b), and nothing about field,
  decade or cell. The answer is constrained by structured outputs to one label from the list.
  The same instructions and list for every usage of a word, so the shared prefix can be cached.
- **Measures.** A cell's distribution is its share of each listed sense plus *other*; *unclear*
  is reported and left out. The change between two cells is the Jensen-Shannon divergence of
  their distributions, in bits. It is the measure DWUG's own graded change uses on its human
  sense clusters, so rung 1 compares like with like. For a documented change the changed
  sense's share is also reported.
- **Bars.** As round X: a change is found when the divergence is above the 95th percentile of
  200 shuffles of the word's labelled usages between the two cells (sampling noise) and above the
  90th percentile of the control words' divergence on the same transition (every word's drift).
- **Model.** One model for every rung, chosen before rung 1 and never changed after it, so that a
  failed rung cannot be repaired by changing model. Current models take no sampling temperature,
  so labels are not deterministic: 10% of every run is labelled twice and the agreement is
  reported as the instrument's self-consistency. A refused request is recorded as *refused* and
  counted with *unclear*; it is not sent to a fallback model, which would mix two instruments.
  The model ID and the date are recorded with every run.
- **Records.** Raw responses stay local. Labels are committed as identifiers and labels only; no
  DWUG text is committed (its licence forbids redistribution).

## The ladder

Each rung has its own pre-registration and stopping rule. A failed rung 1 or 2 stops the
instrument; a failed rung 3 stops it before open questions. No repair rounds, as for items 8,
8a and 8c.

1. **Benchmark (DWUG EN 2.0.1, 46 words, 9,153 usages).** Sense lists written blind to DWUG's
   clusters, change scores and usages. Primary: Spearman between the label divergence and DWUG's
   graded change, beside XL-LEXEME's APD (0.864 here, 0.886 published) and BERT's 0.563.
   Secondary: adjusted Rand index between labels and DWUG's clusters, beside the published
   GPT-4 (0.340), XL-LEXEME (0.273) and human (0.633) figures. Caveat: the SemEval-2020 answers
   are public and a model may have seen them. The labeller sees one usage at a time with no
   period, so it can recall senses but not which period a usage came from; the list writer could
   still favour senses it knows changed, and that is stated with the result.
2. **Gold on this corpus.** The author labels about 130 usages with the frozen lists through the
   same interactive flow as check 2b: 20 for each of *attention, prompt, platform, network* and
   *alignment*, 10 for each of 2 of rung 3's control words, and 10 repeats. Agreement is Cohen's kappa
   between the model and the author, beside the author's own consistency on the repeats.
3. **Known answers over time, with a placebo.** Round X's transitions, cells and sampling (seed
   changed, so no usage check 1a read is reused): the five documented changes, the two quiet
   transitions, and 10 control words drawn at random (seed 2026) from round C's 30, among those
   with at least 100 abstracts in every transition's cells. Plus check 1b's planting design on
   psychology's 2020s *attention*: mechanism usages (check 1a's markers) planted at the same five
   shares, and a placebo of 2020s *attention* usages from a field other than AI. Check 1b's
   unmarked AI pool cannot serve as the placebo for labels: check 1a read 60% of AI's 2020s
   *attention* as the mechanism against 31% marked, so many unmarked usages are mechanism usages.
   The placebo is the test the pair measure failed, and it carries the most weight.
4. **Between fields, then open questions.** *embodiment* between HCI and philosophy, and only
   after that the questions the project started from: field-specific meanings, and one-sided
   changes such as the language-model sense of *hallucination* reaching psychology.

## Volume and cost

Labels: rung 1 about 10,000 (with the 10% repeats); rung 3 at most about 26,500 (2,100 for the
target words, at most 21,000 for the controls, about 900 planting usages, plus repeats); rung 2
is negligible. About 36,500 in all.

Upper estimate per label, Message Batches API (50% of the standard price), about 900 input
tokens and 200 output tokens (the label plus thinking at low effort), no caching assumed:

| Model | Batch price per million tokens, input / output | Per label | Rung 1 | Rung 3 | All |
|---|---|---|---|---|---|
| Claude Opus 5.5 | $2 / $10 | $0.0038 | about $38 | about $100 | about $140 |
| Claude Sonnet 5.5 | $1 / $5 | $0.0019 | about $19 | about $50 | about $70 |
| Claude Haiku 5.5 | $0.05 / $0.25 | $0.0001 | about $1 | about $3 | about $4 |

Prices from Anthropic's API reference as of 3 November 2026; checked again before any batch is
sent. Thinking tokens are the uncertain part, so a 200-label pilot on rung 1's benchmark is run
first and the estimate is redone from its measured usage before the full batch. Caching the
shared prefix would lower the input share, but caching in batches is best-effort and short
prefixes are not cached, so it is left out of the estimate.

## Decisions for the author

1. **An API key.** None is set in the environment or in `.env`. Calls go through the official
   `anthropic` Python SDK, a new dependency.
2. **The model**, fixed before rung 1. The default is Claude Opus 5.5, the most capable model at
   these prices; Sonnet 5.5 halves the cost and Haiku 5.5 makes it negligible, at a risk to
   label quality that rung 1 would measure but not repair.
3. **The author's time:** approving about 15 sense lists (5 target words, 10 controls) and
   labelling about 130 usages, roughly as long as check 2b's 100 pairs.

## Not part of it

Clustering, the grid, the usage-pair measures (except as reference figures), and the direction
of HCI's change.
