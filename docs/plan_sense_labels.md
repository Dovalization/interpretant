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
- **Labelling.** Through the Claude Code CLI in headless mode (`claude -p` with a fixed system
  prompt, no tools, no settings or MCP servers, no session persistence, a pinned model and
  effort, and a JSON schema for the answer), called by a committed script. Each call labels a
  block of about 40 usages of one word: the word, its sense list, and the numbered usages'
  sentences (cut to 100 words, the word marked as in check 2b), with nothing about field, decade
  or cell. A block mixes the word's usages from every cell of the run, planted items included, in
  a seeded random order, so its position in a block tells the model nothing about a usage's cell;
  check 1a was labelled the same way. The schema allows one label from the list per item, and a
  block whose answer does not cover every item exactly once is sent again.
- **Measures.** A cell's distribution is its share of each listed sense plus *other*; *unclear*
  is reported and left out. The change between two cells is the Jensen-Shannon divergence of
  their distributions, in bits. It is the measure DWUG's own graded change uses on its human
  sense clusters, so rung 1 compares like with like. For a documented change the changed
  sense's share is also reported.
- **Bars.** As round X: a change is found when the divergence is above the 95th percentile of
  200 shuffles of the word's labelled usages between the two cells (sampling noise) and above the
  90th percentile of the control words' divergence on the same transition (every word's drift).
- **Model.** Claude Sonnet 5.5 at low effort, unless the pilot below leads the author to choose
  JEV. One model for every rung, chosen before rung 1 and never changed after it, so that a
  failed rung cannot be repaired by changing model. Current models take no sampling temperature,
  so labels are not deterministic: 10% of every run is labelled twice, in a different block and
  position, and the agreement is reported as the instrument's self-consistency, which also
  measures any effect of a usage's neighbours in its block. A refused block is recorded as
  *refused* and its items counted with *unclear*; it is not sent to another model, which would
  mix two instruments. The model ID, the CLI version, the date and each call's reported token
  usage are recorded with every run.
- **Records.** Raw responses stay local. Labels are committed as identifiers and labels only; no
  DWUG text is committed (its licence forbids redistribution).

## Pilot: Sonnet 5.5 and JEV on check 2b's usages (before rung 1)

A small comparison of two labellers, run before any rung, on usages whose judgements already
exist and which no rung uses as evidence: the 186 distinct usages in check 2b's 100 pairs
(*attention, prompt, platform, network, component, having*), which the author judged pair by pair.

- **Labellers.** Claude Sonnet 5.5 at low effort through the CLI, as above, in blocks of 20 with
  10% repeats. JEV (TypeSafe AI's decision model, pinned at `jev-1.14.0`), one request per usage:
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
  labeller before rung 1. If Sonnet is kept, JEV also labels rung 2's gold usages, reported beside
  Sonnet and deciding nothing.
- **Data sent.** Short excerpts of public OpenAlex abstracts, to Anthropic through the CLI and to
  TypeSafe through its API; no personal data. TypeSafe's privacy policy says it does not train on
  user data; its retention period is not stated in its public documentation.

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

In blocks of 40 that is about 900 CLI calls. A smoke test on 2026-11-10 (three invented
sentences, Claude Opus 5.5 at low effort) answered in about 6 seconds with a valid labelled
object, using about 1,300 input tokens of fixed overhead and 377 output tokens, 122 of them
thinking. A block of 40 should take well under a minute, so the whole ladder is a few hours with
a handful of calls in parallel. On a subscription login the calls count against the plan's usage
limits, not a bill; the CLI's reported cost for the smoke test was $0.018 on Opus 5.5 and
$0.006 on Sonnet 5.5 (2 seconds, no thinking tokens), which as API figures put the whole ladder at
roughly $7–12 for Sonnet 5.5. The pilot's measured time and tokens re-estimate this before rung 1.
JEV is priced at $0.042 per million input tokens with output free, so the whole ladder on JEV
would cost well under a dollar.

The Message Batches API remains the alternative if usage limits get in the way: the same
prompts at half the standard price, about $140 for Opus 5.5 one usage per request (prices from
Anthropic's API reference as of 3 November 2026). It needs an API key, which is not set.

## Decisions for the author

1. **The labeller**, fixed before rung 1, after the pilot: Claude Sonnet 5.5 at low effort (chosen
   2026-11-10 over Opus 5.5 to save usage) or JEV. JEV needs a TypeSafe API key.
2. **The author's time:** approving about 15 sense lists (5 target words, 10 controls) and
   labelling about 130 usages, roughly as long as check 2b's 100 pairs.

## Not part of it

Clustering, the grid, the usage-pair measures (except as reference figures), and the direction
of HCI's change.
