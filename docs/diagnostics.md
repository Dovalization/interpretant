# Diagnostics: where the problem lies

Checks from the diagnostic plan in `docs/assumptions.md`. Like predictions, each check's rules
are committed before anything it counts is looked at; results are appended below and not edited.

## Check 1a: are the known changes present in the cells? (committed 2026-11-10, before any marker was counted or any sampled usage was read)

Rounds 5 and X judge an instrument by whether it finds changes documented in a field's history.
That assumes the changes are present in the cells the instrument reads (assumption A5). This check
measures their presence directly, so a missed change can be put down to the instrument or to the
data.

### Method

- **Usages.** Exactly the usages round X's scorer reads: for each case, the word with its regular
  plural, one occurrence per abstract, at most 150 abstracts per cell, seed 1, round C′'s quality
  rules, abstracts in both cells of a transition removed (`scripts/score_pairs_known.py`).
- **Cases and the sense whose presence is measured.**

  | Case | Cells | Sense |
  |---|---|---|
  | *attention*, AI | 2010s, 2020s | the attention mechanism of neural networks |
  | *prompt*, AI (word-targeted cells) | 2000s, 2020s | an instruction or input given to a language model |
  | *platform*, communication | 2000s, 2010s | a digital or social-media platform |
  | *network*, AI | 1980s, 1990s | an artificial neural network |
  | *alignment*, AI | 2010s, 2020s | aligning an AI system with human intentions, preferences or values |
  | *attention*, AI (quiet) | 1990s, 2000s | the attention mechanism |
  | *attention*, psychology (quiet) | 2010s, 2020s | the attention mechanism |

- **Markers.** A usage's context is marked for the sense when it matches, as whole words and in
  any case, one of:
  - attention mechanism: *self-attention, self attention, cross-attention, transformer(s),
    multi-head, multihead, attention mechanism(s), attention layer(s), attention weight(s),
    attention head(s), attention module(s)* (round C's list);
  - prompt: *language model(s), large language, LLM(s), ChatGPT, GPT, generative AI, chatbot(s),
    few-shot, zero-shot, in-context, prompt engineering, prompt tuning, prompt learning* (round
    C″'s list);
  - platform: *social media, social network(s), social networking, Facebook, Twitter, YouTube,
    Instagram, TikTok, WeChat, Weibo, Reddit, digital platform(s), online platform(s),
    platform economy, platform companies, platformisation, platformization*;
  - neural network: *neural network(s), neural net(s), connectionist, connectionism,
    backpropagation, back-propagation, perceptron(s), hidden unit(s), hidden layer(s), Hopfield,
    Boltzmann*;
  - alignment: *language model(s), LLM(s), RLHF, human feedback, human preference(s), human
    values, value alignment, AI alignment, AI safety*.
- **Blind reading.** Twenty usages per cell, drawn at random (seed 2026) from those above, are
  pooled across all fourteen cells, shuffled, and labelled one by one without the cell, the
  marker match or the case's other usages in view: *yes* if the context uses the word in the
  case's sense, *no* otherwise, *unclear* if it cannot be told. The labels are made by Claude,
  recorded in `data/diagnostics/check_1a_labels.tsv`, and open to the author's correction; a
  correction is recorded as such.
- **Present.** A documented change is *present* in its cells when the marked share and the read
  share (yes over yes plus no) each rise by at least 0.10 from the earlier cell to the later. A
  quiet transition is *quiet* when the marked share changes by less than 0.05 and the read share
  by less than 0.10.

### What each outcome crosses out

- A documented change **present** but missed by an instrument: the instrument (or its bars),
  not the data, failed on that case.
- A documented change **absent**: the case is not a known answer for these cells; any verdict
  that used it (round 5, round X) says nothing about the instrument for that case, and the next
  step is the cells (A1, A4), not the instrument.
- The two shares **disagree** (one rises, one does not): the markers are unreliable for that
  sense, and the reading decides, reported as such.
- A quiet transition **not quiet**: something changed where nothing was documented; reported,
  and the transition is not used as a quiet case again.

Scored by `scripts/check_known_presence.py`, committed before it is run.
