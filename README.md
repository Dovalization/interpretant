# interpretant

[![CI](https://github.com/Dovalization/interpretant/actions/workflows/ci.yml/badge.svg)](https://github.com/Dovalization/interpretant/actions/workflows/ci.yml)

`interpretant` measures how academic communities change what they do with a word. It trains one word-embedding model per field and decade on a grid of research abstracts (eleven fields, 1950s–2020s) and two centuries of canonical books, puts every model in one coordinate system, and judges each word's movement against the movement of the whole vocabulary. Every claim is tested against predictions committed to git before the evidence existed.

One finding has survived a second training seed, a second method, a lemmatised rebuild of the corpus and pre-registered tests: **_embodiment_ means two different things**, virtual reality's sense of body ownership in HCI and phenomenological corporeality in philosophy. A second was stated too strongly here and is withdrawn in that form. HCI's abstracts did change between the 1990s and the 2020s, inside the field's own venues too: the GUI-engineering vocabulary of the 1990s receded, and the vocabularies of empirical studies and of machine learning came in. But the earlier headline, that HCI's language moved _towards the behavioural and social sciences'_, held under one instrument and frame of reference and failed under another, so this README claims the change and no direction.

![Nearest neighbours of embodiment in philosophy and HCI, by decade](figures/embodiment.png)

---

## Findings

Results are tested against predictions committed before the evidence: four rounds on the embedding models (the trained models; a second seed with unseen decades and field pairs; a second, count-based method; the corpus rebuilt with plurals and British spellings collapsed), seven rounds on cross-perplexity, three with a field classifier, one with both instruments, a held-out check on every cell's model, three calibration rounds for a sense-clustering instrument, which it did not pass, one for the embedding grid on documented changes and one on the SemEval-2020 English benchmark, two for an instrument that compares usages pair by pair, which stopped at its second rung, a set of diagnostics, and seven for a sense-label instrument, which passed its benchmark rung, passed a second attempt at its corpus rung and stopped at its third, on known changes. Every prediction and every miss is in [`docs/predictions.md`](docs/predictions.md), [`docs/predictions_cross_perplexity.md`](docs/predictions_cross_perplexity.md), [`docs/predictions_senses.md`](docs/predictions_senses.md), [`docs/predictions_pairs.md`](docs/predictions_pairs.md), [`docs/diagnostics.md`](docs/diagnostics.md) and [`docs/predictions_labels.md`](docs/predictions_labels.md); the story is in [devlog 04](devlogs/04.md) (the grid), [devlog 05](devlogs/05.md) (cross-perplexity and the lemmatised corpus), [devlog 06](devlogs/06.md) (senses, and when to stop) [devlog 07](devlogs/07.md) (calibrating the grid) and [devlog 08](devlogs/08.md) (sense labels on this corpus). What each instrument did when checked against changes whose history is known, and what that taught, is in [_Calibrate before you measure_](docs/known_answers.md).

**Robust across seeds and methods**

- **_Embodiment_ means two different things.** In HCI it is virtual reality's sense of body ownership and presence (neighbours: _ownership, presence, illusion, immersion, self-location_); in philosophy it is phenomenological corporeality (_corporeality, intersubjectivity, temporality, spatiality, affectivity_). It passed every test it faced: unseen decades, a second seed, a second method, and the lemmatised corpus in both seeds. The word is now about three times as frequent in HCI as in philosophy.
- **A subfield's model predicts its parent field better than the reverse** (AI → computer science, cognitive psychology → psychology): first seen as a reversed prediction, then predicted and confirmed on the lemmatised corpus.

**Depends on the instrument**

- **HCI's abstracts changed between the 1990s and the 2020s; no direction is claimed.** Both instruments see the change. HCI's 1990s GUI-engineering vocabulary (_menu, display, graphical, user_interface, software_) receded, and a classifier trained on the 1990s recognises 0.87 of HCI's 1990s abstracts as HCI and 0.75 of its 2020s ones (most fields drift from their 1990s selves by this measure; computer science, AI and neuroscience drift further than HCI). The change is there inside the venues HCI had in both decades (CHI, UIST, IJHCI, HFES and others), though the label widened too (915 sources in the 1990s cell, 3,601 in the 2020s). Two vocabularies came in: that of empirical studies (_research, study, participant, experience_) and that of machine learning (_dataset, accuracy, deep, network_). With technology terms left in, the 1990s classifier's largest gain for HCI is towards AI; with 29 of them removed, it is towards psychology. This two-part description comes from exploratory readings (rounds G and V), not from scored predictions. On the direction, the rounds part:
  - **Language models, each decade's own (rounds J and K′).** Psychology's, sociology's, neuroscience's, cognitive psychology's, communication's and linguistics' 2020s models predict HCI's 2020s abstracts better than their 1990s models predicted its 1990s ones, in every seed and by more than they gained on other texts; AI's and computer science's do not. The gain is a change in HCI's word mix, and it holds inside the stable venues. Read alone, this says HCI's language moved out of computing's register and towards the behavioural and social sciences'.
  - **A classifier held at the 1990s (rounds S, V and N).** A classifier trained on the 1990s sends about twice as many of HCI's 2020s abstracts to the behavioural and social sciences as of its 1990s ones, in every seed, with or without the technology terms. But every field's 2020s abstracts drift that way. Net of the drift all fields share (round N), HCI has moved towards the behavioural and social sciences no more than other fields since the 1990s, and towards computing more than any other field. Psychology did not move towards HCI.

  - **Both instruments in one frame (round W).** Trained on the same 1990s abstracts, scored on the same held-out ones and netted against the same fields, both instruments put HCI's move towards the behavioural and social sciences over computing, on the whole cell and inside the stable venues, in every seed. This is exploratory: the round's predictions are untestable as scored, because the check guarding them (finding a move planted in 10% of HCI's abstracts) was set against the wrong baseline and failed for both instruments.

  The two sets of rounds differ in three things at once: the instrument, the frame of reference (each decade's own models against a fixed 1990s one) and what a gain is netted against. Round W, which held the frame and the baseline fixed, points away from the instrument and towards round N's measure (counting abstracts assigned to each field), which a synthetic test shows can report a move for a field that only changes more than the rest. That reading is exploratory too. Earlier versions of this README said that HCI's language "moved out of computing's register" and that HCI and psychology "met, with HCI travelling further"; neither statement survives round N. Three explanations of the first observation, philosophy's model predicting HCI's abstracts better, were tested and failed on the way (HCI borrowing philosophy's words; philosophy turning to technology; philosophy learning HCI's objects).
- **Fields converged between the 1980s and the 2010s under the language models, and not under the classifier.** The convergence is one of word mixes: across all field pairs the mix effect is +0.09 bits and the rate effect −0.03, so fields came to share more words while no field's model got better at another's language. A classifier finds the fields as easy to tell apart as before (round S), which fits a convergence of shared words in which every field keeps enough of its own to be recognised. The language models' result was first seen as a reversed prediction, then predicted and confirmed on the lemmatised corpus.

![Fall in excess cross-entropy on HCI's abstracts for every field's model, 1990s to 2020s](figures/hci_gains.png)

![Excess cross-entropy between philosophy and HCI, 1990s to 2020s](figures/philosophy_hci.png)

**Robust in part**

- **No change detected in _intentionality_** in philosophy from the 1980s to the 2020s, under both methods and in both seeds of the lemmatised corpus. This was listed as stability; a calibration on documented changes (round 5) showed the grid misses most of them, so the absence of a change is no longer read as evidence that nothing changed.
- _Representation_ in AI moves towards learned representations (_embedding, latent, encoder, learned_) in every run including the lemmatised one; its symbolic 1980s (_knowledge, reasoning, declarative_) holds in TWEC only.
- HCI's _affordance_ is about experiential qualities (_physicality, experiential, interactivity, materiality_) in the 2010s in every run; the 2020s depend on the method.
- Philosophy's 2020s _consciousness_ moves (a candidate under TWEC in both corpora); the lemmatised neighbours carry both earlier readings at once, computational vocabulary (_computational, substrate, information_processing_) and theories of consciousness (_qualia, sentience_).

![Nearest neighbours of representation in AI, 1980s, 2000s and 2020s](figures/representation_ai.png)

**Open**

- _Information_ converging between psychology and computer science: seen in both unlemmatised TWEC seeds, absent in PPMI-SVD, and under the committed threshold on the lemmatised corpus. Not robust.
- The share of HCI abstracts using philosophy's theme words (_experience, value, meaning, agency, embodied_) doubled from the 1990s to the 2020s; observed in round G, not predicted.
- The direction of HCI's change. In round W's frame both instruments point towards the behavioural and social sciences, but the round's predictions are untestable as scored, and by the rule committed before it no further round on the direction is run.

**Calibration of the sense-label instrument**

- **It does not mistake a change of source for a change of meaning, and it sees a planted change of a fifth.** Planting sociology's _years_ into psychology's 2020s cell (same senses, different source) registered in none of five draws, and a planted attention-mechanism share was found in every draw from 20%. The usage-pair instrument needed a planted share of 30%, and its own placebo could not separate a change of source from one of sense (see the correction in `docs/diagnostics.md`). Against the control words' drift it found two of four documented changes (_attention_ in AI, _platform_ in communication), so it stops before open questions; against random shuffles alone it found all four and neither quiet transition, as two language models labelling the same usages did.

My guesses before each round were worse than the method. Round 1 passed 3 of 9 predictions; the misses, stated as predictions, passed on unseen evidence in round 2 (7 of 9); round 4, on the lemmatised corpus, passed 9 of 11, and one of its misses was a wrong guess about the instrument itself (neighbour pools shrank when frequent plurals merged, instead of growing). The predictions files record every miss and every correction of an interpretation.

---

## How it is tested

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '14px', 'lineColor': '#6b7280'}}}%%
flowchart TD
    classDef stage fill:#f1f5f9,stroke:#94a3b8,color:#1e293b,rx:8

    C["<b>Corpus</b><br/><i>field&nbsp;×&nbsp;decade&nbsp;cells,&nbsp;cleaned&nbsp;and&nbsp;phrase-merged</i><br/>OpenAlex&nbsp;grid&nbsp;·&nbsp;full&nbsp;cells&nbsp;·&nbsp;Books"]:::stage
    E["<b>Embeddings</b><br/><i>one&nbsp;model&nbsp;per&nbsp;cell&nbsp;on&nbsp;a&nbsp;shared&nbsp;compass</i><br/>TWEC&nbsp;(Word2Vec)"]:::stage
    D["<b>Drift</b><br/><i>within&nbsp;fields&nbsp;and&nbsp;across&nbsp;fields</i><br/>cosine&nbsp;drift&nbsp;·&nbsp;neighbourhood&nbsp;shift&nbsp;·&nbsp;corpus-wide&nbsp;noise&nbsp;floor"]:::stage
    L["<b>Language models</b><br/><i>who&nbsp;predicts&nbsp;whom</i><br/>Kneser-Ney&nbsp;bigrams&nbsp;·&nbsp;cross-perplexity"]:::stage
    R["<b>Robustness</b><br/><i>does&nbsp;a&nbsp;finding&nbsp;survive?</i><br/>second&nbsp;seed&nbsp;·&nbsp;PPMI-SVD&nbsp;·&nbsp;resampled&nbsp;runs&nbsp;·&nbsp;pre-registered&nbsp;predictions"]:::stage

    C --> E --> D --> R
    C --> L --> R
```

**One model per cell, one space.** Models trained separately share no coordinate system. TWEC (_Training Temporal Word Embeddings with a Compass_, Di Carlo et al. 2019) trains a compass first, here on the same token budget from every cell so that large recent cells do not define the space for the 1850s, and then trains each cell's model with the compass's context vectors frozen. Every cell lands in one space, so a word can be compared across decades within a field and across fields within a decade.

**Drift is judged against the corpus itself.** Retraining a cell moves an ordinary word's neighbours almost as much as a decade does, so a raw distance means little. Within each pair of cells, every word above 150 occurrences in both is ranked within its frequency bin on cosine drift and on the change of its 25 nearest neighbours; only words in the top 5% on both count as candidates. Everyday control words check the floor; they are flagged in under 1% of measurements.

**Whose language predicts whose.** A Kneser-Ney bigram model per cell scores every other cell's held-out abstracts. The excess cross-entropy of field B's text under field A's model, over B's own model, measures how predictable B's language is from A's (Vilhena et al. 2014). Parent and subfield cells share no abstracts, every model trains on the same number of tokens, and each run is repeated with five seeds. Predictable is not borrowed: every fall is split into a word-mix effect (the text's frequencies moved) and a rate effect (the model learned), checked against unigram models, and, where a subfield label spans decades, against venue-stable cells.

**Senses by labelling.** For meaning rather than context, each occurrence of a word gets one sense from a fixed list the author approved. Cue words (position-aware collocation rules, also approved) label some occurrences; a logistic regression per word on XL-LEXEME's contextual vectors, trained on those and on each sense's example sentence, labels the rest. A cell's meaning is its share of each sense, corrected for the classifier's confusions (adjusted classify and count), and a change is the Jensen-Shannon divergence between two cells' shares, judged against random shuffles, the drift of ten control words, a planted change and a placebo. Claude Sonnet 5.5 and JEV label the same usages as comparisons.

**Nothing counts once.** A candidate becomes a finding only if it survives a second training seed, a count-based PPMI-SVD model with the settings of Levy, Goldberg and Dagan (2015), and a prediction written and committed before the evidence was looked at.

---

## The corpus

Every question here compares something specific, one field across decades or several fields in the same decade, so the corpus is a grid of cells, one per field and decade, each built the same way. The design and its rationale are in [`docs/corpus.md`](docs/corpus.md).

![Research abstracts per field and decade cell](figures/corpus_grid.png)

Regular plurals and British spellings are collapsed before training (`corpus lemmas`); verbs, participles and _-ing_ nominalisations are left alone, because a general lemmatiser turns _learning_ into _learn_. The unlemmatised grid remains on disk as the frozen evidence for rounds 1–3.

**The OpenAlex grid** (about 677,000 abstracts): a seeded random sample of up to 10,000 English research abstracts per field and decade, 1950s–2020s, for eleven fields sharing OpenAlex's single topic taxonomy. Only research prose is kept (articles, reviews, book chapters, books, dissertations, conference papers); archive catalogue records, scraped web pages, duplicates and non-English text are removed, and no author may exceed 1% of a cell.

**Full cells**: where a word is too rare for a 10,000-abstract sample, its home field is taken in full, philosophy for the 1980s–2020s and HCI for the 2000s–2020s, for within-field trajectories.

**Books** (358 titles, 1820s–2020s, about 42 million tokens): canonical works in philosophy of mind, AI, design and cultural theory, each dated by the English edition it contains (a translation is the translator's English) and capped at 250,000 tokens per author per decade. Every edition decision is recorded in `scripts/redate_books.py`.

---

## Why "interpretant"

In Peirce's theory of signs, the _interpretant_ is neither the word nor the thing it points to; it is the meaning the sign produces in a community of minds, and it is not fixed. The word "intelligence" in a 1965 philosophy paper and a 2005 machine learning paper share a signifier and possibly an object, but they carry different interpretants: different networks of association, implication and use that have accumulated around the same sign in different communities.

Word vectors are a natural instrument for this. The distributional hypothesis behind Word2Vec, "you shall know a word by the company it keeps" (Firth, 1957), converges with Wittgenstein's observation that, for a large class of cases, meaning is use: a vector encodes not what a word _is_ but how a community used it. So the pipeline does not measure a word drifting. It measures a community changing what it does with a sign, which is why philosophy's _intelligence_ and computer science's _intelligence_ are treated as two things, and why the corpus is split by field. The longer account, with Peirce's three registers of the interpretant and infinite semiosis, is in [`docs/methods.md`](docs/methods.md).

---

## Explorer

`web/` is a static Next.js explorer of the same data: a word's neighbour columns per field and
decade, its drift ranks and frequency, the cross-field divergence matrix, the who-predicts-whom
matrix with each pair over time, and the full prediction ledger. It reads JSON exported from the
models and tables (`uv run python -m interpretant.export web/data`), so it needs no Python to
build:

```bash
cd web && pnpm install && pnpm dev
```

## Getting started

```bash
# Install dependencies
uv sync

# Books: re-date the manifest and ingest (author cap of 250,000 tokens per decade)
uv run python scripts/redate_books.py
uv run interpretant corpus books ingest

# OpenAlex grid: download (needs OPENALEX_API_KEY in .env) and clean
uv run interpretant corpus openalex download
uv run interpretant corpus openalex ingest

# Collapse plurals and British spellings, merge multi-word terms, then audit coverage
uv run interpretant corpus lemmas learn      # writes data/processed/lemmas.tsv for review
uv run interpretant corpus lemmas apply
uv run interpretant corpus phrases learn
uv run interpretant corpus phrases apply
uv run interpretant corpus audit --processed-dir data/phrased --output data/phrased/audit.json

# Train one TWEC model per field x decade cell, then measure drift
uv run interpretant twec grid
uv run interpretant drift grid          # data/drift/within_field.parquet, cross_field.parquet

# Robustness: a second seed and a count-based method
uv run interpretant twec grid --seed 43 --output-dir models/twec_grid_seed43
uv run interpretant ppmi grid

# Cross-perplexity between fields (corrected design, five seeds)
uv run interpretant drift perplexity --drop-shared --fixed-budget 200000 \
  --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 \
  --output data/drift/cross_perplexity_corrected.parquet

# Redraw the figures above
uv run interpretant figures

# Score the predictions against the evidence
uv run python scripts/score_predictions.py
```

---

## Stack

| Tool                   | Role                                                  |
| ---------------------- | ----------------------------------------------------- |
| `uv`                   | Package management and environments                   |
| `Click`                | CLI                                                   |
| `gensim`               | Word2Vec training (TWEC compass and cell models), phrase detection |
| `scipy`                | PPMI-SVD (sparse SVD)                                 |
| OpenAlex API           | Field × decade abstract samples                        |
| `pandas` / `pyarrow`   | Drift and cross-perplexity tables                     |
| `matplotlib`           | Figures                                               |
| `Ruff`, `mypy` (strict), `pytest` | Linting, type checking, tests (synthetic fixtures, no real NLP in the suite) |

---

## Limitations

**Embeddings capture the interpretant, not the full Peircean triad.** There is no object in the model, no connection between the word vector and whatever the word refers to in the world. The project measures drift of the collective interpretant, not drift of reference or truth. Two words can have converging vectors because a community started using them interchangeably, or because they were genuinely tracking the same phenomenon from different angles, and the embeddings cannot distinguish between these.

**Frequency and training noise.** Rare words look volatile even when the change is noise, and retraining a cell moves an ordinary word's neighbours almost as much as a decade does (a median Jaccard distance of about 0.4 against 0.6). Drift is therefore only read through the corpus-wide noise floor, and words below 150 occurrences in a cell are not measured there.

**The corpus is not the discourse.** The grid captures research abstracts as OpenAlex classifies them; the interpretants measured here are those of academic communities, as their abstracts present them. OpenAlex's topic labels are noisy in small early cells, and a label's scope can widen over time: the HCI cell spans 915 sources in the 1990s and 3,601 in the 2020s, which inflated one whole-cell measurement until a venue-stable check separated the field's move from the label's. Abstracts before the 1970s are often fragments, most fields are thin before the 1980s, and the 1930s–1950s rest mostly on the books.

**Movement between fields needs a reference.** Whether a field moved towards another depends on whether the other is taken as it was then or as it is now, on what the movement is netted against, and on whether the instrument weighs every word or the words that tell fields apart. The HCI rounds differ on all three, which is why the findings above report HCI's change and no direction.

**Senses are not yet read on open questions.** A word's vector mixes its senses. An instrument that clusters contextual embeddings (SciBERT) of each occurrence was calibrated on three changes with a known history, _attention_'s mechanism sense in AI, the language-model sense of _hallucination_ and of _prompt_, and did not pass: the encoder placed each change where its history puts it, but no rule for turning clusters into one meaning's share held on a case it was not chosen on. By a rule committed before the last round, sense-level drift was stopped. Its readings, such as AI's sense of _intelligence_ spreading into other fields in the 2020s, are exploratory only. A second design compared usages pair by pair with XL-LEXEME, an encoder trained on same-meaning judgements, and stopped at its second rung; measures of that kind respond to context as well as sense: on the DWUG EN benchmark the average pairwise distance keeps 0.84 of its agreement with human scores when the periods are shuffled (it mostly measures a word's variety), and on this corpus XL-LEXEME's similarities follow a word's context within one sense (check 2b), while a third of ordinary words' contexts drift between decades (round X). A third design labels each occurrence with one sense from an author-approved list (a classifier on XL-LEXEME vectors, weakly supervised from cue words, with no generative model). On DWUG EN it ranked the changes at 0.67, against 0.69 for Claude Sonnet 5.5 labelling the same usages. On this corpus it passed a placebo and a planted change and found two of four documented changes against the control words' drift; one miss is a labelling error on _prompt_ (dialogue-system prompts of the 2000s read as language-model prompts), the other a change inside the 1980s that decade cells blur. It has no reading on open questions yet.

**The grid is cautious.** Asked to find documented sense changes (round 5), it flagged one of four, _alignment_ in AI's 2020s, and missed _attention_'s mechanism sense in AI (a near miss), _platform_ in communication and _network_ at AI's connectionist turn; it flagged none of the transitions where nothing documented happened, and control words stay quiet throughout. Its candidates are evidence of change; its silences are not evidence of stability.

**Decade granularity is coarse.** Training on ten-year windows smooths over within-decade shifts and may misattribute the timing of changes that happened rapidly at decade boundaries.

**Equal cells trade volume for comparability.** A 10,000-abstract cell cannot measure late-arriving words such as _affordance_ or _intentionality_; full cells can, but only within their field. Cross-field comparisons also rest on few control words, and one of them (_food_) genuinely differs across fields.

---

## Roadmap

The plan is in [`ROADMAP.md`](ROADMAP.md). Next: the sense-label classifier improved on the spent rounds, then one new pre-registered set of known changes at half-decade windows; and the web explorer online. The instruments' record against known answers is written up in [`docs/known_answers.md`](docs/known_answers.md).

---

## License

MIT
