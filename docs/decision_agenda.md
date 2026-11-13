# Decision agenda before the store rebuild

*2026-11-12. The decisions that shape the rebuild, step by step: what each step does, why (with
sources and measurements), and what it replaces. Recommendations were Claude's; the decisions are
the author's. Revised after two reviews on 2026-11-12; the amendments as first recorded are in
commits `3ca8d4e` and `949d3c2`. The scripts behind the measured numbers are in
`scripts/evidence/` (commit `dd4a5aa`). Where `docs/pipeline_map.md` or the four design drafts
(`design_store_rebuild.md`, `design_data_acquisition.md`, `design_detection_experiments.md`,
`design_sense_counting.md`) disagree with this document, this document holds, until the docs pass
retires or merges them.*

Terms:

- **Static** pipeline: one vector per word per cell (TWEC, PPMI). **Contextual** pipeline: one
  vector or sense label per occurrence in its sentence (XL-LEXEME). They replace "type track" and
  "usage track".
- **Cell**: one field in one decade, the unit the corpus is sampled by (an OpenAlex field label,
  English abstracts, a seeded random sample).
- **Usage**: one occurrence of a target word in its sentence.
- **Comparison**: one target word in one field between two periods (4.7); **control**: a word
  matched to a comparison's target and scored the same way, the bar the target must clear (step 5,
  6.3).
- **J**: the change measure, the across-period distance minus the within-period distances (6.3).
- **Usage id** and **content key**: the two identifiers of a usage (4.5).
- **Counter**: a method that assigns usages to senses (step 7).
- **Stage**: one step of the pipeline as DVC runs it (step 9).

Rule for every step (the author): use spaCy's features, or another maintained library, wherever
one fits, and replace hand-rolled code with them, existing code included. Own code stays only
where no maintained option exists or the library is measured to be wrong on this corpus, and then
it is kept small and the reason recorded. Unmaintained libraries do not count. A new dependency
must be reasonable: when a library would serve one small job that a few lines of the project's
own (or the standard library) handle as well, the project's own code wins. When scipy and
scikit-learn both offer a job, the choice is by kind of job and then the same everywhere: scipy
for maths and statistics (tests and intervals, rank correlation, linear algebra, optimisation,
the distance between two vectors); scikit-learn for machine-learning building blocks and batch
work on matrices (classifiers, cross-validation, preprocessing such as `normalize`, nearest
neighbours).

The steps run from what the project keeps doing (which decides what code survives), through the
foundations every later step depends on (cleaning, words, usages), to the experiments, with the
code layout and the reproducible runs last, since they mostly follow.

## Step 0: delete what is already dead

Done in `ed94882` (1,036 lines out; 143 unused `noqa`): dead functions in `drift/direction.py`,
`visualization/figures.py`, `drift/metrics.py`, `predictions.py`, `drift/senses.py` and
`export.py`, an unused re-export, stale `noqa` comments; the unlemmatised layer's phrase model set
aside in `data/retired/phrases_unlemmatised/`.

Principle (the author): code that will be rebuilt or is no longer used goes; git keeps it. No
frozen-but-runnable code.

## Step 1: what stays in the project

**1a. The static pipeline.** Whole-field comparison (which field's language reads like which,
convergence) leaves the project; comparing fields word by word stays (methodology job 1, roadmap
rung 3). Code is not kept only so old results can be rerun (the author): old findings are
recorded as past experiments, with their method and downsides, and the code goes. The direction of
HCI's change was closed by round W on 2026-11-10.

- Delete: `drift perplexity` and `decompose`, `drift heldout`, `drift separability`,
  `drift direction` (with `perplexity.py`, `heldout.py`, `separability.py`, `direction.py`, their
  tests and CLI commands); nothing in the contextual pipeline imports them.
- Keep: TWEC, PPMI and `drift grid`, to rerun on the new cleaned text (detection experiment 4, and
  the only instrument that scans every word).
- Delete for now: the web explorer's export and the figures code; rebuilt when there are new
  results.
- Set aside: the old results (`data/drift` tables, `models/`) in `data/retired/static/`.
- Library verdicts for the code that stays: the TWEC trainer is built on gensim's internals, and
  `cade` (the packaged method; last release November 2020, pre-alpha, a patched gensim that
  clashes with the installed one) does not count, so the alternatives are decided in step 6b;
  neighbour search by scikit-learn's `NearestNeighbors` (cosine); frequency bands by Polars'
  `qcut` and `rank`; PPMI construction stays the project's own (no maintained PPMI library), its
  SVD already scipy's.

**1b. Past rounds' scripts.** Precedent: on 2026-11-11 the 19 old scorers were deleted, since the
fresh start set their rules aside; results stay in `docs/` and `data/`.

| Fate | Scripts |
|---|---|
| Live: replaced during the rebuild, the old script deleted in the same commit | `score_pairs_known` (round X; the store's re-baseline), `score_pairs_benchmark` (round U on DWUG; calibration), `check_planted_pairs` (planted changes), `score_labels_benchmark` (DWUG labelling; the human benchmark), `build_vector_store`, `fetch_sense_words`, `pilot_labellers` (its `usage_id`, `usages` and DWUG helpers feed the store's dev set and two scripts) |
| Keep the data, delete the script | `gold_pairs` (check 2b; its Markdown judgements converted to a TSV in `curated/` first), `check_known_presence` (check 1a; its marker words move first, since `check_planted_pairs` imports them): the author's judgements are what matters |
| Delete: field-level (1a) | `score_venue` (K), `score_decomposition` (G), `measure_label_noise` (S) |
| Delete: closed line | `score_labels_wordnet` (round Z; cites a missing script) |
| Superseded by step 7a | `score_labels_induced` and `drift/induction.py` (round Q's substitutes), the clustering half of `drift senses` (SciBERT, KMeans, silhouette) |
| Superseded by step 7 | `score_labels_classifier`, `score_labels_corpus`, `score_labels_head_to_head`, `score_labels_known`, `finetune_classifier`, `dev_harness` (its last experiment committed first as a record, `fdf83cc`), `scripts/review/` (once the keystroke judging page of 4b exists) |
| Superseded by step 4.7 | `scripts/discovery/` (`data/discovery/longlist.json` kept as data) |
| Superseded by step 5 | `select_control_words`, `select_cross_field_controls`, `select_sense_controls` (the four control files kept as data) |

Groups "delete" and "keep the data" go once their results are recorded and check 1a's marker
words have moved; live scripts go with their replacements; superseded ones go once their results
are in the record below.

**The record of past experiments** is a separate document, `docs/experiments.md`, not a README
section: one entry per experiment with what it found, the method and settings, its downsides
(including the older cleaning), its status (holds, withdrawn, untestable) and the commit that
produced it. The predictions files stay as the detailed ledger, and the README points to the
record. It is written before any step 1 deletion.

## Step 2: cleaning and duplicates

Inputs: `pipeline_map.md` stages 2–3; store draft decisions 3, 6, 7; spaCy's documentation
(Matcher, SpanRuler, custom components and extensions, `url_match`, `nlp.pipe` with `as_tuples`);
counts of each problem in the 1,363,172 unique raw abstracts (`raw_problems.py`): curly quotes
29%, URLs 1.4%, LaTeX commands 0.9% against dollar maths 0.15%, copyright 0.45%, soft hyphens
0.28%, DOIs without a scheme 0.19%, control characters 0.17%, ligatures 0.17%, mojibake 0.03%
(461), HTML almost none.

**2.1 One cleaning path for both pipelines**, inside the single spaCy pass, replacing four rule
sets that applied on different paths.

**2.2 Stand-off cleaning: the text is never edited.** The stored text is the repaired text, never
changed afterwards; every removal is a labelled character range in a spaCy span group, copied into
the abstracts table as a Polars list column of (start, end, reason). Cutting removed text would
shift every stored position; keeping it would leave it in the sentences. This is the established
form: brat never modifies the annotated text, the ISO 24612 Linguistic Annotation Framework
addresses annotations by character offsets, STAM likewise, and spaCy's document model is one
(non-destructive tokens keep `token.idx`; the `SpanRuler` writes matches to `doc.spans`; `DocBin`
saves span groups). The two views are derived after the corpus-level rules, never stored as
edited text: the **contextual view** takes the sentences not marked removed (small ranges inside
kept sentences stay, so target offsets do not move); the **static view** (lowercase, no
punctuation) skips every marked range. Stage order: spaCy pass → corpus-level rules → derived
views (the full order is in step 9).

**2.3 The rules.**

- *Before spaCy, repair only:*
  - NFKC normalisation. It expands ligatures, and also rewrites the ellipsis, full-width
    punctuation, no-break spaces, superscript and subscript digits, the micro sign and ™ (UAX #15
    warns that it erases formatting distinctions); measured on 322,329 raw abstracts it changes
    2.0% of them and joins a word to a following digit in 24 (0.01%; *log₂* → *log2*), so it
    stays.
  - Control and format characters removed by Unicode category; soft hyphens removed.
  - Abstracts with mojibake are dropped (461, 0.03%), detected by a short list of the usual
    mis-decoding signatures (such as `Ã©` and `â€™`) in the cleaning patterns file, so the
    detector is data and is checked in the sample audit. `ftfy` was declined: it would repair
    those 0.03% instead of dropping them; the hand-written repair table goes.
- *In the spaCy pass:* one `SpanRuler` patterns file (data, reviewable) labels every text rule, so
  every removal carries its reason: catalogue and citation records, web-page scrapes, exam-dump
  spam, JSTOR and copyright footers, the leading "Abstract" label, LaTeX (only spans with LaTeX
  markers `\`, `^`, `_`, since `$…$` also catches prose between two dollar amounts; `pylatexenc`
  declined for now, revisited if the sample audit shows harm), URLs and e-mails (spaCy's
  `like_url`, and DOIs kept as single tokens through the tokenizer's `url_match`). Keeping rules:
  fewer than 5 words; malformed abstracts with no sentence punctuation (a "sentence" over about
  150 words) flagged, their fate left to the sample audit. LaTeX is left out of the static view
  only.
- *After the pass, corpus-level rules:* template sentences (shared by 20 or more abstracts,
  spaCy's sentences with digits masked, counted with Polars over every source after footers are
  marked; an abstract made mostly of them is dropped); a sentence repeated within one abstract is
  kept once, later copies marked "repeated sentence" (353 of 322,329 raw abstracts, 0.11%, repeat
  a sentence of 40 or more characters, often the abstract pasted twice; without this rule two
  mentions would share a usage id, 4.5); near-duplicates (2.5); the author cap, 1% of a cell with
  a floor of at least 5 works, so small cells are not gutted.
- *Normalisation of the static view:* accents folded (café → cafe); non-Latin text goes by the
  language check instead of being deleted; digit-only tokens dropped from the static view and
  kept in contextual sentences (*1990s* is not a number to spaCy and stays); hyphenated words kept
  whole on purpose (*self-organising*, *meta-analysis*), so compounds are handled in 4.2.
- *Sample audit first:* the old regexes and the new patterns run side by side and must catch the
  same abstracts before the regexes are deleted; count what the template rule alone catches and
  delete patterns it makes redundant; check how the mojibake abstracts spread over cells before
  dropping them.

**2.4 Language.** lingua alone (`lingua-language-detector`, version 2.1.1 installed; version 2.2
needs Python
3.12), restricted with `from_languages` to English and the languages found in the corpus. Per
abstract first; per sentence only for abstracts that come out mixed or uncertain, with
`detect_multiple_languages_of` (documented as experimental, so validated on the sample), since
short sentences are where detectors err (lingua's documentation: probabilities from few n-grams
"are not reliable"). Sentences without letters are not judged. Non-English sentences are marked
removed; an abstract is dropped when most of it is not English (the share fixed after reading a
sample). Each sentence's language and confidence are kept (4.6).

**2.5 Duplicates.** Near-duplicates (a preprint and its published version; OpenAlex keeps a
conference paper and its journal version as two works on purpose) by MinHash with `datasketch`,
keeping the earliest and recording each removal; exact duplicates are a special case. The
threshold comes from the sample study; published practice ranges from 0.5 (Broder 1997; The Pile)
to 0.8 (Lee et al. 2022, with edit similarity). The same work in two cells is kept in every cell
it belongs to (some overlap is legitimate: AI sits inside computer science, and a full cell holds
its sample cell; but in a comparison between two such fields, shared works make them look alike),
its cells recorded in the abstracts table; leaving shared works out of a
comparison is one library helper every measure uses, not a rule each measure writes.

Replaces: four cleaning rule sets, three language checks, two sentence splitters, two URL
regexes, the mojibake table, six duplicate and shared-text rules.

## Step 3: words (tokens, lemmas, phrases)

Inputs: `docs/spacy.md`; `pipeline_map.md` stage 4; the lemma audit (`lemma_audit*.py`,
`lemma_gaps.py`).

**3.1 Tokens.** spaCy tokenises everywhere; the static view is written from its tokens.
`preprocess_text` retires (it deleted curly apostrophes instead of spacing them, fusing 542,162
tokens such as *one's* → *ones*), and the regex tokenisers retire as their users are rebuilt.

**3.2 Lemmas.** The static view takes **spaCy's in-context lemma** for plural nouns and
third-person verbs, found by morphology (`Number=Plur`; `Person=3`, `VerbForm=Fin`), so the reading
decides (*leaves* → *leaf* as a noun, *leave* as a verb). Auxiliaries stay as they are (*is*,
*has*, *does*); ordinary verbs merge (*goes* → *go*). Target matching (4.1) uses spaCy's full
lemma after the exceptions, so *prompted* is found. Exceptions go through spaCy's
`AttributeRuler` from one file, `curated/lemma_exceptions.tsv` (today `data/lemma_overrides.tsv`):
the lexicalised plurals the project keeps (*media*) and the reviewed corrections of spaCy's errors
(3.3); spaCy receives them as a list (`attribute_ruler.add`). The voted lemma map stops running on
the text; its rules survive only as the tool that finds spaCy's errors. `TARGET_VARIANTS` (26
hand-written plurals and spellings of target words) and the protected target list go, since
beside the lemmas they would be a second list that can disagree; a check confirms that no target
word is mapped away.

*Why:* over 22.6 million plural-noun and third-person-verb tokens, where both spaCy and the voted
map merge, spaCy differs on 1,136 types (0.46% of tokens): about half spelling only (spaCy keeps
British forms), the rest systematic errors (*senses* → *sens*, *uses* → *us*, *analyses* →
*analyse*, *motives* → *motif*, *doses* → *dos*, unknown words cut to *identifie*, *smartwatche*).
spaCy publishes no lemma accuracy for its English pipelines, so this measurement is the evidence.

**3.3 The lemma review.** The list is the types where both spaCy and the voted map merge and
disagree, spelling-only differences left out (577 types, 0.024% of tokens, measured 2026-11-12;
very skewed: the 50 most frequent cover 89% of their tokens, 100 cover 93%, 400 cover 99%,
`review_coverage.py`), plus the voted map's remaining wrong merges (*cosmos*, *thanks*, *mises*,
*forceps*; plurals whose singular means something else, such as *novels*, *currents*, *logos*;
names such as *hans*, *rogers*). It is rebuilt from the new static view, where the apostrophe
artefacts are gone. The author reviews the 100 most frequent types and every case touching a word
in the word tables (about two sittings); for the rest, 30 drawn at random are checked, and if the
voted map is right on at least 27 (set before reading), its lemma is accepted for the whole tail;
otherwise more are reviewed. Approved corrections go into the exceptions file.

**3.4 Spellings.** spaCy's `norm_` table, applied to the lemma and filtered to spelling variants
(it also holds informal norms such as *cos* → *because*); undocumented, so it is read once,
filtered and saved as a versioned data file, and a spaCy update cannot change it silently. The
project's rules are the backstop for derived words the table lacks (*personalisation*,
*operationalise*, *judgement*).

**3.5 Phrases** (static pipeline). Term candidates from spaCy's `Matcher` on tags and morphology
(adjectives and nouns ending in a noun, up to four words), scored by **C-value** with a frequency
floor in Polars (Frantzi and Ananiadou 1999; Frantzi, Ananiadou and Mima 2000), which discounts
candidates nested in longer terms; not gensim's `Phrases`, whose 3,502 current phrases include
boilerplate and citations (*your_browser*, *york_routledge*), non-terms (*lend_itself*,
*1980s_and_1990s*) and terms cut to pairs (*systolic_blood*). No maintained library scores terms
(PyATE archived June 2023; textacy's last release April 2023). Broken-word repair stays. The
static view never joins a phrase containing a single-word target, so a target keeps all its uses
(in AI, *neural network* is 30% of *network*'s uses in the 1980s and 50% in the 1990s, and the old
rules removed them; *artificial intelligence* is 77–86% of *intelligence*'s). spaCy's noun chunks
were considered and set aside: they return whole noun phrases (*the new transformer model*), not
terms. Phrase targets (*neural network*, *artificial intelligence*) are studied in
the contextual pipeline through the `PhraseMatcher` (4.2). gensim stays only for skip-gram
training, if step 6b keeps it.

**3.6 No second full pass.** The exceptions are applied to the cached documents, not by parsing
again: spaCy pass → corpus-level rules → the lemma review → exceptions applied to the cache by an
`AttributeRuler` → matching (measured on 2,000 abstracts: parsing 25.6 s, loading the cache and
applying exceptions 0.2 s, about two minutes for the corpus on one process).

Replaces: two lemma maps (one map plus the `AttributeRuler`), about six regex tokenisers,
`TARGET_VARIANTS`.

## Step 4: usages and the store

Inputs: store draft decisions 1, 2, 4, 5; the data-acquisition draft; `pipeline_map.md` stages
7–8; steps 2 and 3.

**4.1 Matching.** spaCy's `Matcher` on the full lemma after exceptions (so irregular plurals and
British spellings match through step 3), or on the form for words the word table marks so
(*having*); the `PhraseMatcher` for phrase targets. Every usage records its part of speech,
morphology, any named entity it sits in, and the static view's token beside it, so the two
pipelines' counts reconcile. A usage is **in scope** when it has the part of speech its word
table gives and is not inside a name (*medium* is a noun, not the adjective of *medium effect
size*, 22% of its uses; *learning* has 36% verb readings); the usages table keeps every match, and
measures use in-scope usages by default and may include the rest on purpose.

**4.2 Compounds.** Hyphenated words stay whole, so a lemma search alone would miss every compound
(*self-attention* has the lemma *self-attention*): 2–8% of a target's uses (`compounds.py`), among
them the senses studied (*self-attention*, *cross-attention*) and others (*attention-deficit*,
*actor-network*, *e-learning*). A second `Matcher` rule finds the target as any part of a
hyphenated word; each usage records the compound and the target's position (head or modifier);
the encoder marks only the target part (*self-⟨t⟩attention⟨/t⟩*); compounds are in scope by
default and a measure may leave them out on purpose. Multi-word compounds (*neural network*) are
marked the same way by the `PhraseMatcher`. Using compounds as evidence of a sense is step 7's.

**4.3 One mention per abstract.** The usages table lists every mention; for encoding and
measures, one per abstract is drawn **among its in-scope mentions only** (the sampling frame is
the eligible units: Kish 1965), by the smallest cryptographic hash (sha256) of (seed, work id,
content key), so abstracts stay independent observations, so one abstract's draw never depends on
other abstracts or on row order
(hash-based assignment: Kohavi et al. 2009; Kohavi, Tang and Xu 2020; their experiments found
only cryptographic hashes free of correlations).

*Why one, and not the first:* in abstracts with several mentions (15–44% of them), the first lands
in the opening sentence about twice as often as a random one (*network* 49% against 23%,
`occurrences.py`). All mentions with each abstract weighted equally stays possible without
re-reading, and was not chosen because within one text a word usually keeps its sense ("one sense
per discourse", Gale, Church and Yarowsky 1992: 51 of 54 pairs with coarse senses; with fine
senses about a third of ambiguous words vary within a text, Krovetz 1998), while every measure
would have to handle same-abstract pairs, split the classifier by abstract and resample abstracts
in bootstraps (if it is switched on, that grouped handling is written once, in the library). This
is the project's own reasoning, not a convention: DWUG and SemEval-2020 sample sentences. Abstracts
with several mentions carry a changing sense more often (57% against 26% in check 1a), so reports
state that shares are shares of abstracts.

**4.4 Context.** The store records the usage and each encoder decides how to read it, so
XL-LEXEME can be replaced later (the author's requirement). Stored, encoder-independent: the
sentence text, the target's character offsets, the sentence's place in its abstract (the
abstract is in the abstracts table), the identifiers (4.5). One **encoder table** declares each
encoder's model and pinned revision, its context policy (the sentence by default, as XL-LEXEME
was trained on Word-in-Context sentence pairs; sentence plus neighbours or the whole abstract
possible), its window and centring (XL-LEXEME: 128 subword tokens including the markers, centred
on the target) and how it marks or pools the target; swapping an encoder is one table entry and an
encoding run.
The old 100-word pre-cut is dropped.
Sentence length is recorded (2,462 target sentences: median 25 words, 1.1% under 8 words, 2.1%
beyond XL-LEXEME's window, `sentence_lengths.py`).

**4.5 Identifiers**, each defined once in the library (replacing five copies of `key_of` and two
other schemes):

- the **content key**: sha256 of exactly the text an encoder reads plus the target's position in
  it, 20 hex characters; for sentence encoders such as XL-LEXEME it is the hash of the sentence
  and the target's offsets, so identical text is encoded once and shared, and an encoder reading
  neighbours or the whole abstract gets a different key, so identical sentences in different
  abstracts never share its vector wrongly;
- the **usage id**: (work id, word, content key), the usages table's row identifier. It changes
  only if that mention's sentence changes (a mention number would shift whenever a rule found an
  earlier mention), and it is unique once repeated sentences are kept once (2.3).

pandera checks enforce both: usage ids unique, content keys equal to the hash recomputed from the
text and offsets, offsets pointing at the target.

**4.6 Store format.** Parquet throughout, written and read with Polars.

- An **abstracts** table is the durable record of the cleaned corpus: work id, cells, source,
  year, repaired text, sentence boundaries, each sentence's language and confidence, flags, and
  the removal ranges with their reasons (2.2).
- A **usages** table points into it instead of copying text: usage id, content key, word,
  compound and position, part of speech, morphology, lemma, the static view's token, in scope,
  sentence and target offsets, sentence length, sentence language and confidence,
  near-duplicate group.
- **Vectors** per encoder name and version and per word
  (`vectors/encoder=<name>-<version>/word=<word>/`, read by Polars' hive partitioning), one row
  per content key with the vector as 16-bit floats (1,884 bytes per vector; a filtered read of 100
  keys in 0.07 s; storing in float16 changes J by about 1e-6, cast to 32-bit before computing, as
  Polars advises). A new encoder or new weights write a new folder and never read the old one,
  since the content key covers the text, not the model. Encoding adds only missing keys (an
  anti-join of usages against existing keys); no numpy shards, sidecar keys or index file.
- spaCy's **`DocBin`** is a rebuildable cache of the annotations, tied to spaCy's version (target
  words occur in nearly all abstracts: 57% hold one of the 39 targets alone; about 2–2.5 GB for
  the corpus, `docbin_size.py`). It keeps spaCy's default attributes: in 3.8.18 they include
  `NORM`, `SENT_START` and `ENT_ID` (the documentation lists older defaults), and a round trip
  kept sentences, norms and span groups, while an explicit list without `SENT_START` lost the
  sentences. Span groups are always saved; only extension attributes (lingua's sentence
  languages) need `store_user_data=True`. A round-trip test guards this.
- The static pipeline's normalised text stays as plain text files.

Common practice: DWUG keeps a usages table per word with the context, target offsets and sentence
offsets; Parquet partitioned by column is the standard analytical layout; no vector database is
needed, since measures compare usages within one word.

**4.7 Words and comparisons.** Three tables in `curated/`, one per kind of unit (Wickham, "Tidy
Data", 2014: "each type of observational unit forms a table"):

- `words.tsv`: lemma, part of speech, match by `lemma` or `form`, `held_out` (yes or no). A word is
  a (lemma, part of speech) pair, as SemEval-2020 joined English targets with their tag "because
  English frequently combines various POS in one lemma and many of our target words underwent
  POS-specific semantic changes" (Schlechtweg et al. 2020): usages and frequency counts cover that
  part of speech only.
- `comparisons.tsv`: comparison id, target, field, the two periods, purpose (documented change,
  stable check, held-out change or held-out stable), n per period (6.3).
- `comparison_controls.tsv`: comparison id, control, status and reason (step 5).

Roles follow from the data: a target heads a comparison, a control is listed under one; *attention*
has three comparisons (AI 2010s→2020s, AI 1990s→2000s, psychology 2010s→2020s), each with its own
controls; a check that no word is both target and control is a join. Held-out words are filtered
from every development output (6.2). *component* and *having* are stable checks (check 2b's
author-judged words, 7.1); the six-word shortlist gets a role after checking its history.
Candidates come from the long list (`data/discovery/longlist.json`, kept as data;
`scripts/discovery/` deleted; the list regenerated from the rebuilt static grid later). Caps by
role: targets and their controls share one, so they stay comparable; candidates keep a lower one
(300 now, since one cap of 1,000 would mean up to 12 million long-list usages against 1.9 million
in the whole store today, where the control word *paper* alone has 125,784). Caps and the floor
start as provisional values in the parameter files (step 9) and may be revised by the experiments.
Controls are encoded only in the cells of their comparisons. The static pipeline reads its own
list; the static outputs record the word tables' version, since the phrase rule of 3.5 makes the
normalised text depend on them. Same-abstract controls are drawn from the abstracts table when
needed, not listed. These tables replace `words_of_interest.txt`, the control-word files as
sources, the list typed into `build_vector_store.py` and the scope part of `longlist.json` (the
static and contextual target lists barely overlap today: 39 against 11).

**4.8 Downloads** are a curated decision, not a stage. The pipeline reports gaps (word, field,
period, usable usages, OpenAlex's total of abstracts containing the word, estimated cost); the
author approves a committed **download plan** (word, cells, reason); the download script carries
out only the plan, after the author's go, records what it fetched (counts, totals, seed), and its
output is registered with `dvc add` (step 9). Order: the plan is drafted from existing counts and
provisional values; downloads run before the full spaCy pass, so one pass cleans the regular
samples and the downloads together; after the pass, the gap report counts usable usages in final
terms (4.1, 4.2) and any cell still under the floor gets a small curated top-up, after which the
corpus-level rules (templates, near-duplicates) rerun in full, not as a patch. The thin words are
sized per year, not per decade, for the yearly series (experiment 3); first candidates:
*attention* (AI), *platform* (communication), *alignment* (AI). OpenAlex's total of matching
abstracts is recorded per cell (the share sampled); when it is small, the whole set is
downloaded; fetching continues until the floor of usable usages is met, since OpenAlex's stemmed
search returns related forms. Kept rules: one source per word (a word's cells all regular or all
downloaded); downloaded cells never feed the static pipeline (the abstracts table marks each
abstract's source). The floor (500) is provisional. Cost: OpenAlex gives each account $1 of usage
a day and a search request costs 10 credits, so the earlier estimate (under 200 of 10,000 credits a
day) is re-estimated with the plan. A seeded OpenAlex sample is repeatable only while records are
not added, withdrawn or reclassified (4b), which is why downloads are registered, not regenerated.

**4.9 Models and running time.** The small model (`en_core_web_sm`) for the whole corpus:
cleaning, sentences, the static view (the last full pass, lemma votes with tagger and lemmatizer,
read 1,274,029 abstracts, 206 million tokens, in 30 minutes on 10 CPU processes; `senter`, lingua
and the `SpanRuler` keep it under an hour). `senter` ships disabled in the small model and is
enabled for the pass, with abbreviation special cases (it matched the parser on the sample). The
transformer model (`en_core_web_trf`) is a candidate for target sentences only, re-annotating
tags, morphology and entities (and dependencies if step 7 needs them) within the small model's
sentences, so boundaries stay consistent. spaCy's own scores (general English): tags 97.3%
against 98.1%, entities 0.843 against 0.899, labelled dependencies 0.899 against 0.938, sentence
boundaries 0.907 against 0.886 (the meta does not say which component produced the sentence
score, so sentence quality rests on the comparison below). On lemmas the two models differed on
63 of 10,395 types (`compare_taggers.py`). Decided by a one-field comparison, read on the words
where the small tagger errs
(*medium*, *learning*, 4.1): if more than a couple of percent of usages change their in-scope
flag or lemma between the models, target
sentences use the transformer, on bazzite's GPU after its repository is synced.

## Step 4b: corpus quality

How we know the corpus is good: four layers, the last two curated. The written record follows
"Datasheets for Datasets" (Gebru et al. 2018), with a Data Statement (Bender and Friedman 2018)
beside it.

| Layer | What it does | How |
|---|---|---|
| 1. Definition | What a cell is (OpenAlex field label, language, document types, decade, the random sample and its seed) and what the source changed: download dates; OpenAlex replaced its topic classifier in October 2026 (assignments changed; the old ones, frozen on 5 October 2026, remain available; the raw cells are dated 9–11 November 2026, so they appear to carry the new ones); closed-access abstracts of two large publishers withdrawn in November 2022 and November 2024 (Kramer 2024); OpenAlex's coverage (an abstract for over 60% of 2022 works, about 45% before 2000). Known weaknesses: field labels come from a classifier, nested fields overlap, HCI's label widened from 915 venues to 3,601 | Written once, reviewed |
| 2. Automatic checks | pandera on every rebuild (4.6); counts per removal reason, and the share of short abstracts, per cell and per decade | Automatic |
| 3. Human review of samples | Below | Curated |
| 4. Sign-off | No rebuilt corpus is used for experiments until the datasheet and the review are read and approved, with the decision recorded | Curated |

**Layer 3, the review**, sized to about an hour (the author's budget) and as automatic as possible
without losing quality, on established audit
methods: sampled audits with a fixed set of categories (Kreutzer et al. 2022), auditing what
filters removed as well as what they kept (Dodge et al. 2021, on C4), stratified random samples,
a codebook written before reading, blind repeats for a single reviewer, and lot quality assurance
sampling, which screens groups with a fixed sample and decision rule chosen so both kinds of
misclassification stay tolerable (Hund, Bedrick and Pagano 2015).

- **Automatic first:** language (lingua confidence), duplicates (MinHash), boilerplate (the
  `SpanRuler` and template counts), malformed or cut-off abstracts (length, missing sentence
  punctuation), and likely wrong-field abstracts (a scikit-learn field classifier trained on data
  separate from what it flags, treating nested fields such as AI inside computer science as
  compatible; OpenAlex's calibrated topic score fetched for those if useful).
- **Codebook, six categories:** valid abstract; wrong field (clearly another discipline, not
  merely interdisciplinary, since the fields overlap by design); not English or mixed;
  boilerplate or junk; duplicate; truncated or malformed.
- **Sample, first rebuild:** 15 random kept abstracts per field (165), split between two eras
  (before 2000, from 2000), so about 80 per era pooled over fields; about 10 dropped items per
  removal rule (about 120); 80 automatically flagged abstracts (finding problems fast and testing
  whether the flags can be trusted); 20 blind repeats. About 385 judgements, about 1 hour 5
  minutes in sittings of about 15 minutes; later rebuilds about 250 (no flagged items, no
  repeats), fewer if the flags proved reliable.
- **Rules set before reading** (screening rules, with Wilson intervals reported):
  - wrong field: 2 or more of a field's 15 gives it a second sample of 15, and 3 or more of the 30
    gives it the closer look (at a true rate of 5% a field gets a second sample 17% of the time and
    the closer look 11%; at 20%, 83% and 82%; with 11 fields, a second sample somewhere happens
    in 87% of reviews; the stricter 4 or more would flag 4.5% and 78%);
  - junk or non-abstracts under 2%, judged per era, not overall, since quality differs by era
    (zero junk in an era's 80 puts its rate likely under 3.7%);
  - a removal rule with 2 or more good texts among its 10 dropped items gets a closer look.
- Results are reported automatically into the datasheet; the author reads one summary.
- **Judging page:** field, decade, each abstract's beginning and end with removed spans
  highlighted (so truncation and over-cleaning show), the six categories on keys 1–6, progress
  saved; adapted from `scripts/review/judge_template.html`.
- **When:** the first rebuild, and again only when cleaning rules or sources change; otherwise
  layers 1 and 2.

*Why these numbers:* about 12 abstracts per field could not show "wrong field under 5%" (zero
problems in 12 still allows about 22%); quality differs strongly by era (in the raw cells 34% of
1950s abstracts are under 250 characters against 3.6% of the 2020s', median length 503 against
1,204 characters; Kim, Holst and Ginis 2026, a preprint, report integrity problems in about 21% of
pre-2000 OpenAlex abstracts against about 6% for 2020–2025: too little content, metadata in place
of the abstract, truncation). The cut at exactly 200 characters they report does not show in these
cells (0.3–1.0% of abstracts per decade fall within 197–203 characters).

## Step 5: control words

Inputs: the detection draft's experiments 2 and 3; `pipeline_map.md` stage 11; round X's stored
vectors and the existing cells' counts (`second_review/control_matching.py`,
`bar_stability.py`, `spread_check.py`). How a target is judged against its controls is in 6.3.

- **The four control-word files are retired as sources** and kept as data (past rounds used
  them). They contradict each other (*control* is a target and a rung-3 control) and do not match
  the targets; round A found 4 of 4 changes against shuffles, 2 of 4 against them.
- **Shuffles on for every measure**, the established control condition (Dubossarsky et al. 2017,
  "Outta Control"; Dubossarsky et al. 2019, "Time-Out", which shuffle corpora for static vectors;
  Liu, Medlar and Glowacka 2021 permute period labels for usage vectors, as here). They are the
  floor, not the bar: a word whose contexts moved only with its field's topics passes them, as
  round X's quiet transition did, and across domains a shuffle control left most of a measure's
  agreement in place (Schlechtweg et al. 2019, SURel).
- **Matched controls, generated per comparison** and stored in
  `curated/comparison_controls.tsv` after the author approves the proposal (step 8):
  - same part of speech as the target (the word table's);
  - frequency (the share of abstracts containing the word in that part of speech) between half
    and twice the target's in each period (Card 2023 draws background terms within a factor of
    two; SemEval-2020 matched "the same POS and comparable frequency development", taking the
    smallest tolerance that found a control); the realised tolerance is recorded per comparison;
  - within-period spread (the mean distance between a word's usages within one period) within
    ±50% of the target's in the **early** period, the state before the change: a change raises
    the later period's spread, so matching on it would pick naturally varied controls and absorb
    part of the change (the rule against adjusting for a variable the treatment has affected:
    Rosenbaum 1984);
  - at least n usages in each period, n being the comparison's (6.3);
  - not a target, a sense-list word or a candidate, and never a held-out word: the rank test's
    level is chosen during development (6.3), so a held-out word among the controls would shape
    the method before its own confirmation ("If the training dataset is not separated from the
    test dataset during all pre-processing, modeling and evaluation steps, the model has access
    to information in the test set before its performance is evaluated": Kapoor and Narayanan
    2023).

  Every eligible candidate is a control, not thinned to the closest 30: thinning was one more
  choice to make, and the rank test of 6.3 needs the numbers. The pool starts at about 60
  frequency-matched candidates per comparison (encoding resumes by missing keys, so encoding them
  is cheap); if fewer than about 40 are eligible after the spread band, more candidates are drawn
  and encoded. Experiment 2 still checks whether the controls' J relates to frequency or spread
  within the bands, and a relation there tightens the rule before any target is scored.

  Every comparison J scores has at least n usages in both periods (6.3), so the target's
  frequency and early spread are always measurable. A word under the floor in the early period
  is not a detection comparison, as the benchmarks also require usages in both periods
  (SemEval-2020 checked 50 per period, DWUG sampled 100): its arrival is tested by sense
  counting's anachronism test (7.3) and by planted new senses on the synthetic set (0b), the
  way new senses are usually evaluated (Cook and Hirst 2011).

  **Targets whose cells are word downloads** (4.8) are not matched on frequency: their downloaded
  cells hold the word by construction, and candidates as rare as the target in the regular cells
  cannot supply n usages (*prompt* is in 15 of about 10,000 regular AI abstracts of the 2010s,
  against n of 500 or more). Their controls come from the regular cells, matched on part of
  speech and spread; each control's two periods come from one source, as 4.8 requires for every
  word. Equal n is how the field removes frequency's effect on sampling noise (Liu, Medlar and
  Glowacka 2021 subsample; SemEval-2020 and DWUG draw a fixed number of usages per word), and the
  spread band covers how varied a word's uses are. Rounds X and A judged *prompt*'s downloaded
  cells against regular-cell controls the same way. These controls are far more frequent than
  the target, beyond the range where J was found independent of frequency (an open question).
- **Controls are never replaced on their change score.** Replacing the controls that changed most
  lowers the bar, an outcome-driven exclusion of the kind that inflates false positives (Simmons,
  Nelson and Simonsohn 2011); controls that change a lot measure the method's background change
  and stay. Controls are replaced only for reasons that do not use their score, written into the
  code and the confirmation plan before any target is scored: wrong part of speech in practice,
  mostly a name or fixed phrase, frequency mismatch after cleaning, a documented change in a
  dictionary, or the author's blind reading of a few usages per period (as SemEval-2020 verified
  its stable words), done before scoring. The set is then frozen for that comparison; every
  replacement is recorded with its reason. Controls are compared with each other, never with
  their shuffles (topic drift moves every word, so that test would reject them all).
- Same-abstract controls are drawn at experiment time from the abstracts table; the static grid
  uses its frequency bands.

*Why within a factor of two:* on the existing cells (nouns, at least 100 abstracts per period),
30 candidates matched in both periods need ±34% (*attention*, AI 2010s→2020s), ±39%
(*alignment*), ±57% (*platform*, communication 2000s→2010s) and ±61% (*network*, AI
1980s→1990s), and 60 candidates ±44% to ±70%; within SemEval's ±15% there are 10 candidates for
*attention*, 2 for *alignment* and none for *platform* or *network*, because a documented change
usually came with a rise in frequency; *prompt* and *hallucination* are nearly absent from the
regular sample's early periods, which is why their cells are word downloads (298 and 323 early
usages in the existing store). Frequency matters less here than for static vectors: with n
equalised, round X's 157 control-and-transition pairs show no relation between J and frequency
(Spearman −0.04) and a weak one with the size of the frequency change (+0.14, p = 0.07).

*Why match on spread, and at ±50%:* the distance between two samples of a word "gets larger as w
acquires more meanings" (Dubossarsky et al. 2017), and SemEval-2020, which did not control its
test sets for polysemy, found "strong correlations … between graded change and polysemy"
(Schlechtweg et al. 2020). J subtracts the within-period distances, but only partly cancels
spread: on round X's 175 control-and-transition pairs at equal n, APD follows spread completely
(Spearman +1.00) and J still rises with it (+0.21, p = 0.004), so controls less varied than their
target would set the bar too low. Around round X's documented changes, ±50% of the early
period's spread keeps 12–20 of about 30 unmatched controls and ±25% only 5–10. The later
period's spread rises with the change (*alignment* 0.157 → 0.219, *attention* in AI 0.192 →
0.279).

Replaces: four control-word files from four scripts, parsed three ways.

## Step 6: detection experiments

Inputs: `docs/design_detection_experiments.md`, steps 2–5, and established practice. Detection
asks whether a word's use changed between periods, and how much. Established methods first,
exploratory ones after (the author):

| Order | Experiment | Standing | Needs |
|---|---|---|---|
| 0a | Calibration on the datasets the LSCD Benchmark covers (Schlechtweg, Virk and Arefyev 2024; DWUG in five languages, DiscoWUG, RefWUG, NorDiaChange, ChiWUG, DURel, SURel, RuSemShift, RuShiftEval; only DWUG EN is English): encoder and measures against human graded change, comparable with published numbers, J's being those for the distance between mean vectors (6.3). The benchmark's code is not a dependency (Hydra scripts pinned to Python 3.10, no licence); its datasets and standard metrics are used with scipy and scikit-learn | Standard practice | Encoders, the published datasets |
| 0b | A systematic synthetic set as the development set: planted scenarios (a sense gained gradually, a sense gained as the old one is lost) and distractors that are not changes of meaning (a frequency change, a short-lived spike) (Shoemark et al. 2019) | Established | The rebuilt store |
| 1 | J on every usage, judged against matched controls and shuffles (6.3) | Established | The rebuilt store; the matched controls (step 5) |
| 2 | Matched controls (step 5) as the bar, with the check that their J does not follow frequency or spread | Established (SemEval-2020 style) | The word tables |
| 3 | Year-by-year series of a word's change, replacing half-decade windows (Kulkarni et al. 2015, with a permutation null; Shoemark et al. 2019: the full series beats two points). Each year, or a window of two or three years where a year holds fewer than about 50 usages (fixed per word before measuring), is compared with one reference period of several hundred usages; the series is scored by change points and by its linear trend, and the last period is among the references tried on the synthetic set (Shoemark et al. found it the better reference, and linear trends to match or exceed change points) | Established | Years per usage (4.6); downloads sized per year (4.8); `ruptures` (1.1.10, September 2025, maintained), added when the experiment starts |
| 4 | One measure for the static grid instead of the two-threshold rule, and the choice among the static models (6b) | The project's own diagnosis | The rebuilt grid |
| 5 | Same-abstract controls, aimed at topic change | Exploratory: no method separates topic shift from semantic change (Kutuzov et al. 2022 call the two inseparable in the data; Hengchen et al. 2021 note that methods conflate meaning with topical information; partial precedents: Perrone et al. 2019 for genre, Card 2023 for frequency-matched background terms) | The abstracts table |
| 6 | Removing the era signal from the vectors (concept erasure, LEACE), the eraser always fitted on words other than the one it is applied to | Exploratory | Last, since it alters every vector; `concept-erasure` (EleutherAI, last release January 2024) or LEACE's closed form written with numpy and scipy, decided then |

*Why experiment 3's design:* usages per year in the existing store, before 2020, are about 20 for
*attention* in AI, 4–16 for *alignment*, 2–57 for *platform* in communication and 150–320 for
*network*; with no change at all, J's 95th percentile is 0.016 for 20 usages against 20 and
0.0085 for 20 against 500 (simulated), beside decade-level changes of 0.010 (*platform*), 0.025
(*alignment*) and 0.036 (*attention*).

*Why experiment 6 fits the eraser on other words:* J is a function of the two periods' mean
vectors (6.3), and LEACE makes those means equal for the data it is fitted on, so fitting it on
the target's own usages removes J by construction (*attention*, AI 2010s→2020s: 0.0355 before,
−0.0007 after). Fitted on the 30 other words of that transition, it leaves the documented changes
(*attention* 0.0376; *alignment* 0.0225 from 0.0261) and halves a stable word's (*component*
0.0012 from 0.0023).

Checks every experiment reads: the documented changes (*alignment* and *attention* in AI,
2010s→2020s; *network* in AI, 1980s→1990s; *platform* in communication, 2000s→2010s), the quiet
transitions, planted changes, the placebo, shuffles and controls. Exploratory experiments are
judged hard against planted changes and the placebo, since no published result supports them.

**6.1 How results are judged.** Explore freely on the synthetic set (0b) and the benchmark (0a);
confirm once on held-out cases under a confirmation plan committed to git before the confirming
run (the frozen method, the cases, the pass rule; the commit's timestamp shows it came first):
the standard split of exploratory and confirmatory analysis, against the "garden of forking
paths" (Gelman and Loken 2013). Development material is not the project's domain (synthetic
changes; historical general English), so the held-out set from the project's fields is the test
that counts.

**6.2 The held-out set.** All known cases were used in round X, and four are too few to split, so
a fresh set is built and held back:

- **About 20 documented changes and 20 stable words** (chosen like matched controls), drawn from
  words used nowhere in development (*prompt* and *hallucination* are development words, 7.1).
  Claude drafts them with evidence, never change scores (sense, field, dating source, corpus
  counts per decade; sources: Wiktionary through the kaikki.org extracts, well-documented recent
  cases, the corpus); the author reviews them (reading a few usages per decade is annotation, not
  measuring) and judges which count as documented. The set is committed, marked not for use until
  confirmation, its hash recorded. If 20 documented changes cannot be found in the project's
  fields, the plan states the number reached and what it can show.
- **Nothing leaks into development.** Held-out words carry `held_out` in the word table and are
  left out of every development output (screening, reports, gap lists, the static grid's ranked
  lists) until the confirmation plan is committed. Labels for held-out words are made only after
  the plan is frozen, and the labelling procedure is written into it. The plan carries a leakage
  checklist (what development used, the held-out list's hash, that held-out words were filtered
  from every development output, that their labels came after the plan), after Kapoor and
  Narayanan's model info sheets (2023: leakage found in 294 papers across 17 fields, often not
  apparent from the papers themselves); SemEval-2020 likewise kept its gold labels hidden.
- **The pass rule** is set before any measuring (for example, at least 16 of 20 changes found and
  at most 2 of 20 stable words flagged), and the confirmation reports both rates with Wilson
  intervals beside pass or fail. *Why 20:* with 10 of each, a method that finds 70% of changes
  passes "8 of 10" 38% of the time and one with 20% false alarms passes "at most 1 of 10" 38% of
  the time; with 20, those fall to 24% and 21%, while a method at 90% and 5% passes 96% and 92%;
  8 of 10 has a 95% interval of 49–94%, 16 of 20 of 58–92%.

**6.3 The statistics.** All the rules for measuring and judging change, in one place.

- **The measure, J.** The mean distance between usages of different periods minus the mean of the
  two within-period mean distances (cosine distance, a usage never paired with itself). It is a
  known statistic: half the energy distance with squared distances (Rizzo and Székely 2010),
  which is the maximum mean discrepancy with a linear kernel (Gretton et al. 2012; Sejdinovic et
  al. 2013), so it equals half the squared distance between the two periods' mean unit vectors,
  estimated without a usage's pairing with itself. It therefore sees a shift of the mean and
  nothing else, and is the noise-corrected form of the distance between mean vectors (PRT; rank
  correlation
  0.98 on DWUG).
  *Why J:* on DWUG EN (round U, `second_review/j_identity.py`, `measure_candidates.py`), APD
  agrees with human graded change at 0.864 but keeps 0.842 with the periods shuffled, so it
  mostly ranks words by their variety of uses (Kutuzov, Velldal and Øvrelid 2022 proposed
  comparing within-period subsets as future work); PRT keeps 0.60 when shuffled; J agrees at 0.54
  and falls to −0.12. Tried and not better: the energy distance with plain distances (0.46), a
  Gaussian-kernel discrepancy (0.33), J divided by the within-period distance (0.29), a period
  classifier's AUC (0.40). J grows with the square of a shift in sense shares, so sizes are
  compared on its square root where a scale matters.
- **Computation.** From sums of unit vectors, with no usage-by-usage similarity matrix (the
  pairwise and mean-vector forms agree to 1e-7 on DWUG; 100,000 shuffles of 1,000 usages take
  about a second); vectors cast from 16-bit to 32-bit floats first. APD, when reported beside J,
  by scikit-learn's `cosine_similarity`.
- **Equal n.** Each comparison draws the same n usages per period for the target and every
  control (n = the cap, or the target's count if lower, never below the floor; seed recorded);
  candidates with fewer than n are not eligible (step 5); n is a parameter, so a change reruns
  the target and its controls together. *Why:* J's chance level falls with the number of usages
  (with no change, its 95th percentile is 0.0030 at 100 usages per period, 0.0017 at 200, 0.0006
  at 500), so unequal counts would let the controls' noise set the bar; Liu et al. take it as
  given that high-frequency words come out highly significant, and subsample their corpora.
- **The bar: a rank test against the controls.** The target's J is compared with its controls' J
  at the same n: p = (1 + controls at or above the target) / (controls + 1) (Phipson and Smyth
  2010; North, Curtis and Sham 2002), and the target counts as changed at p at or under the level
  frozen in the confirmation plan (0.05 unless development argues otherwise). With 59 controls at
  most two may be at or above the target, under 39 none (with 30, the smallest p is 1/31 =
  0.032), and under 19 no result is possible.
  *Why not the interval against the 95th percentile:* a percentile of 30 values is not a 5% test
  (simulated with a stable target, a J above the 95th percentile of 30 controls was flagged 7.7%
  of the time, 5.9% with 59), and requiring the interval's lower end to clear it counts sampling
  noise twice, since the controls' J carry the same noise at the same n: the interval rule flagged a
  stable target 0.1% of the time but found a moderate change only 2% of the time (a large one 68%),
  against 35–44% (93–97%) for the rank rule at
  3–4% false alarms (`decision_rule.py`). On round X's vectors, *network* in AI, 1980s→1990s (J
  0.0045, interval 0.0010 to 0.0119, 29 controls) is above the 95th percentile (0.0024), fails the
  interval rule and has rank p = 0.067; leaving one control out moves that percentile between
  0.0014 and 0.0025. Precedent for a quantile bar of frequency-matched background terms: Card
  2023.
- **The size: a BCa bootstrap interval**, reported beside the test, resampling abstracts and never
  pairing a usage with its own copy (which would bias J upwards); scipy's default BCa (second-order
  accurate where the percentile interval is first-order: Efron 1987; DiCiccio and Efron 1996; its
  jackknife costs about as much as the bootstrap with a statistic computed from vector sums);
  a `DegenerateDataWarning` stops the run instead of switching method. Near zero, J's
  large-sample theory changes form (Gretton et al. 2012) and a naive bootstrap is not guaranteed;
  tested, the interval contained the true J for 96–100% of simulated words and zero for 99% of
  250 real same-cell splits, so it errs wide there (`bootstrap_null_real.py`).
- **The floor: shuffles.** scipy's `permutation_test` relabels periods; its p-value is
  (exceedances + 1) / (shuffles + 1). Every word gets 100,000 shuffles (computed from sums of
  vectors, timed on one word before relying on it), with the rule that 1/(shuffles + 1) is at
  least ten times below 0.05 / m for m screened words. *Why:* with 1,000 shuffles the smallest
  p-value is about 0.001, while Benjamini–Hochberg needs a word standing alone at 0.05 / m: above
  50 screened words a single finding cannot pass, and several words tied at the smallest p-value
  pass only together (at least 2 of 100, 6 of 300) (simulated: a very large change got p = 0.001,
  `first_review/finding10_demo.py`). Liu et al. escalate step by step (1,000, then 10,000 and
  100,000 shuffles for small p-values). Benjamini–Yekutieli's threshold for a single word is lower
  still (0.05 / m divided by 1 + 1/2 + … + 1/m: about 2.7e-5 at 300 words, under 1e-5 from about
  700 words), so beyond that the shuffles rise with it.
- **Screening many words.** The false discovery rate correction applies to the shuffle p-values:
  Benjamini–Hochberg (`scipy.stats.false_discovery_control`), with Benjamini–Yekutieli
  (`method="by"`) reported beside it, since words in one field share its topic drift (BH holds
  under independence or positive dependence, BY under any: Benjamini and Yekutieli 2001; Liu et
  al. note the dependence and the cost). The rank p-value cannot be corrected (its smallest value
  is 1 / (controls + 1)), so screened words above their controls are candidates for reading,
  about 5% of stable words by construction.
- The simulations above (the permutation floor, J's chance level by n, the rank rule against the
  interval rule) become tests of the statistics module.

## Step 6b: the static models (after detection experiment 4)

Which static model the pipeline keeps, decided by experiment 4 on the synthetic set and the
benchmark, which also tests whether the old grid's misses came from its rule, its windows or the
embeddings; nothing is presumed. The options:

1. TWEC (the project's trainer on gensim's internals; `cade` does not count, step 1a);
2. skip-gram cells trained apart with gensim and aligned with scipy's `orthogonal_procrustes`
   (Hamilton, Leskovec and Jurafsky 2016);
3. temporal referencing: one skip-gram model on all the text, the target words tagged by period;
   plain gensim, no alignment (Dubossarsky et al. 2019, where it outperformed alignment on a
   synthetic task and a manual test set);
4. PPMI-SVD alone (the project's own code, deterministic, no gensim).

*The evidence is mixed:* Hamilton et al. 2016 found PPMI clearly worse and SVD against skip-gram
"somewhat equivocal", with skip-gram finding 72% of known shifts on the smaller corpus against
SVD's 96%; Schlechtweg et al. 2019, across domains (SURel), have skip-gram best at its best setting
(0.85 against PPMI's 0.79 and SVD's 0.64) but close to PPMI on average (0.52 and 0.50); in
DIACR-Ita, Procrustes-aligned skip-gram scored 0.94 against 0.83 for two TWEC systems, and no
independent evaluation found has TWEC clearly ahead; in SemEval-2020 Task 1, 7 of the 10 best
systems in subtask 1 (8 of 10 in subtask 2) used static (type) embeddings, before XL-LEXEME
existed (Schlechtweg et al. 2020). Field-decade cells are small corpora. The
README's best finding rests on skip-gram and PPMI agreeing. If the grid does not earn its place in
experiment 4, the question lapses and gensim goes with it.

## Step 7a: finding senses

Inputs: rounds Y, Q, Z and C–C″; `docs/related_work.md`; the literature below. Finding senses is
not settled (the author), and different approaches are to be tried. Hand-written lists are the
best so far on one benchmark; each automatic method failed in one configuration, not as a
method:

| Method | What was tried | Result |
|---|---|---|
| Hand-written lists | Rounds Y, Q (DWUG), corpus lists | 0.69–0.80 (rank correlation with human graded change) |
| Substitutes and neighbours | Round Q: ModernBERT "or even [MASK]", Louvain communities | 0.39–0.59 |
| Dictionaries | Round Z (WordNet); Wiktionary as a gap check | about 0.18 below hand-written; two gaps in four words |
| Clustering usage vectors | Rounds C, C′, C″: SciBERT, KMeans, k by silhouette | failed calibration |

What failed in rounds C–C″ was the rule turning clusters into a sense's share on unseen data; the
encoder did place every known sense. Clustering may therefore serve to find senses even if
counting is done another way.

*Literature:* under equal conditions APD predicts graded change best, and sense-based approaches
correlate less while in principle explaining more; XL-LEXEME is the best embedding model for
sense induction (Periti and Tahmasebi 2024: adjusted Rand index against DWUG EN's human clusters
0.273, BERT 0.136, both by correlation clustering of pair similarities; GPT-4 reached 0.340; the
0.633 they quote for annotators is agreement on pair judgements, Krippendorff's alpha, not a
ceiling on that scale). DWUG's human clusters come from correlation clustering of a usage-pair
graph (Schlechtweg et al. 2021); predicting the missing edges before clustering improves
agreement with human senses on graphs that already hold human-judged pairs (Noble, Periti and
Tahmasebi 2024), which the corpus lacks. K-Means and affinity propagation are the most used
algorithms, agglomerative clustering and HDBSCAN also in use (Montanelli and Periti's survey);
sense induction is still open. Substitute-based induction at scale (Eyal et al. 2022, Louvain on
substitute graphs) was applied to 12 million scholarly abstracts by Lucy et al. 2023, who induce
senses once on a pooled sample and score each sense per subfield, validated against Wiktionary's
field labels. The hand-written lists' 0.69–0.80 and clustering's 0.27 are different measures and
are not compared directly.

- **7a.1** Automatic methods serve first as an aid to the hand-written lists (proposing senses,
  finding usages no listed sense fits, for the author's review), not as a replacement.
- **7a.2** Established first: (1) XL-LEXEME pair scores with correlation clustering, as DWUG built
  its clusters; (2) XL-LEXEME vectors with affinity propagation, agglomerative clustering
  (distance threshold) or HDBSCAN (which leaves outliers unassigned); (3) substitutes grouped by
  community detection at scale (Eyal et al. 2022; Lucy et al. 2023) with **Leiden** (Traag,
  Waltman and van Eck 2019: Louvain's successor, which guarantees connected communities),
  label propagation as a quick check. Then exploratory: (4) hand-written senses as seeds,
  clustering only the usages no listed sense fits; (5) clustering generated definitions
  (Giulianelli et al. 2023), a comparison arm only, since it needs a generative model.
- **7a.3** Scored by agreement with DWUG's human clusters (adjusted Rand index) on the openly
  published DWUG datasets (English, German, Swedish and others), the hand-written lists by the
  same measure, and on the corpus by the author's labels (step 7). Round Y's bar is 0.273.
- **7a.4 The lists are frozen before labelling.** The aids run on the rebuilt store, the author
  reviews the proposed senses, the lists are frozen as a numbered version, and only then does the
  author label. A gap found after the freeze (for example among usages a counter puts under "no
  listed sense fits") makes a new list version: the author re-labels only the items marked "other
  or unclear", the frozen version's counts stay the confirmatory result, and the new version's
  are reported beside them as exploratory. *Why:* a sense added after the labels and shares exist
  would leave the labels against the old list and is a forking path (Gelman and Loken 2013);
  annotation practice revises the scheme in a pilot, freezes it, then annotates, and re-annotates
  what a later change touches (OntoNotes revised sense groupings until agreement between
  annotators reached 90%, Hovy et al. 2006; content analysis pretests and freezes its codebook;
  Pustejovsky and Stubbs's MATTER cycle versions its guidelines).
- **Tools:** scikit-learn 1.8 (`AffinityPropagation`, `AgglomerativeClustering`, `HDBSCAN`);
  networkx (`label_propagation_communities`; `leiden_communities` needs networkx 3.7 or later
  (September 2026), since in the installed 3.6.1 it is a stub for other backends and raises
  `NotImplementedError`;
  its default quality function is the Constant Potts Model, not modularity, so the choice is set
  explicitly and stated when results are compared with Louvain's); `correlation-clustering` 2.0.0
  (April 2026, by DWUG's first author, depending only on networkx, numpy and scipy; BSD-3-Clause
  in PyPI's `license_expression` and the repository's licence file), the one new dependency.
  DWUG's `WUGs` scripts (no package, conflicting licences) and `chinese-whispers` (last release
  July 2024) are references only.

## Step 7: counting senses

Inputs: the sense-counting draft; check 1a; rounds E, Q and Y; the literature below.

**7.1 Words.** The first set is eight words with reviewed lists: documented changes *attention*,
*network*, *platform*, *alignment* (`corpus_v2.toml` and its working draft); arriving senses
*prompt* (the anachronism test) and *hallucination* (`round_a_prime.toml`); stable checks
*component* and *having*, the check 2b words whose pairs the author judged, with no documented
change. Second wave: *transformer*, *embedding*, *cloud*, and the held-out set of 6.2, where
counting is confirmed (the documented changes are development material here too). Fixes from
checking the lists against steps 2–4: cues written for split hyphens (*attention*'s `<<self`,
`<<head`) stop working when hyphenated words stay whole, so compounds recorded in 4.2 become cues
when cues move to spaCy's `Matcher`; counting uses every reading a list covers (*prompt*'s
adjective and verb senses), while the in-scope flag of 4.1 serves detection; *having* is matched
by form (its reading: verb or auxiliary); *network* gains its own actor-network sense (Latour,
Callon, Law; 214 uses), identified by the compound and kept separable, added to
`corpus_v4_draft.toml` with the social sense excluding its cues (the frozen version 2 is
unchanged).

**7.2 The author's labels.** They train counter 1b, build counter 2's sense vectors, set the "no
sense fits" cut-off, measure each counter's error rates for the correction, choose the counting
method and test it all, so they must be human, blind, random and spread across periods (the
earlier correction failed on error rates measured from easy cue-labelled usages), and they are
split and sampled so they can carry those jobs:

- *Building needs few labels:* averaging contextual vectors per sense holds up with about three
  examples per sense (Loureiro et al., *Computational Linguistics* 2021, shown for coarse noun
  senses with BERT), drawn from the labels and each list's example.
- *Error rates are the binding constraint:* labels are drawn by stratified sampling, by period
  and by the sense a baseline counter predicts, with inclusion probabilities recorded and
  inverse-probability weights in every estimate, so rare senses are covered without bias
  (stratified evaluation: Kumar and Raj; importance-weighted evaluation for rare categories under
  300 labels: Poms et al. 2021). At least about ten labels per sense in each period where it
  exists; below that, pooled error rates with honest intervals. Period balance applies only where
  a sense exists (an arriving sense has no earlier examples).
- *A written nested split per word:* an outer split only for testing, building and calibration
  inside, so no label builds and grades the same thing (Varma and Simon 2006).
- *Check 1a's labels are mapped, not assumed comparable.* Its 280 blind yes/no judgements on
  *attention*, *network*, *platform*, *alignment* and *prompt* stored no work id; rerunning its
  seeded sampling recovered 279 (`first_review/first_mention_1a.py`). The author refines them by
  answering "which sense?" for the 183 "no" and "unclear" items (the 97 "yes" items are the
  changing sense already; about 45 minutes). 71% come from abstracts with one mention, where the
  first mention is the random mention; "unclear" was 2–5% whatever the position (7 items, weak
  evidence). The mapped labels serve building and error rates, each recording its mention
  position and whether it was the only mention; once the new random-mention labels exist, one
  comparison on multi-mention abstracts checks first against random mentions, and only a gap
  there removes the 1a labels from the error-rate set (sample selection bias: Zadrozny 2004).
  Items whose abstract the new cleaning drops are excluded; items whose sentence reads
  differently are listed for the author.
- *Volume and timing:* about 200 new labels across the eight words of 7.1 (about 1–1.5 hours, in
  sittings of about 15 minutes; an earlier plan's 15 labels per stable word were too few for error
  rates) beside check 1a's, more only for words whose intervals come out too wide; each list's
  senses
  plus "other or unclear", on the keystroke judging page of 4b. After the store rebuild and the
  freezing of the lists (7a.4), before the counters are evaluated; labels for held-out words only
  after the confirmation plan is frozen (6.2). Claude's 300 development labels are not ground
  truth; verifying them (about 30 minutes, with about 30 items labelled blind first to measure
  anchoring) is held in reserve for words whose results come out uncertain, and doubles as a
  measure of Claude's labelling.

**7.3 The counters**, in order: (1) the cue classifier, the current baseline; (1b) a supervised
classifier (scikit-learn logistic regression on XL-LEXEME vectors, strongly regularised) trained
on the author's random cross-period labels, the standard baseline, which removes the cue
classifier's era bias at its source; (2) nearest sense by examples, in the manner of LMMS
(Loureiro and Jorge 2019; LMMS Reloaded 2022): sense vectors averaged from XL-LEXEME vectors of
labelled usages and each list's example, which suits XL-LEXEME's training on usage pairs; (3)
nearest definition, zero-shot, expected weaker (with XL-LEXEME, example-based senses beat
gloss-based ones for English in Lautenschlager et al. 2024, and a gloss carries no target word to
mark); (4) cues as spaCy `Matcher` patterns with recorded compounds as cues, alone or cue-first
before (2); (5) a bi-encoder trained on definitions (Blevins and Zettlemoyer 2020) only if the
others fall short, trained by the project since the original code is archived (January 2024,
PyTorch 1.2, non-commercial licence); Sonnet and JEV as the comparison arm. Every counter outputs
probabilities and can answer "no listed sense fits" (a cut-off set on the author's "other or
unclear" items); is evaluated out of fold; is scored per period, with sense vectors built from
examples balanced across periods; and must pass two hard tests, anachronism (no language-model
*prompt* before about 2020, no AI *hallucination* before the late 2010s) and stability (no
invented change in *component* and *having*).

**7.4 Proportions.** Each cell's sense shares, estimated two ways:

- **Labelled cells: the stratified estimate is the reference.** Because the labels are a random
  sample of the cells themselves, within each predicted-sense stratum the labelled share of each
  true sense, weighted by the stratum's size, estimates the cell's shares with no assumption
  about the counter (the stratified estimator of survey sampling; prediction-powered inference,
  Angelopoulos et al. 2023, and design-based supervised learning, Egami et al. 2023, are its
  recent forms). Written in the project's own code; `ppi-python` (0.2.3, December 2024, MIT,
  takes weights) is the library route if its intervals are wanted.
- **Every cell: adjusted counting.** ACC and PACC are computed in the project's own code with a
  confusion matrix weighted by the inverse inclusion probabilities and one constrained
  least-squares solve (`scipy.optimize.nnls`), since QuaPy 0.2.3 fits its adjusted counts from
  predictions and labels only, with no sample weights. EM only with bias-corrected temperature
  scaling (BCTS), through **QuaPy** (0.2.3, October 2026, BSD; it brings pandas back as an
  installed dependency, which the author accepted; the project's code stays on Polars), fed a
  sample redrawn in proportion to the weights and the counter's probabilities directly (its
  `aggregation_fit` and `aggregate` take precomputed predictions; tested in a throwaway environment
  with the project's versions): uncalibrated EM collapsed in tests and is reliable
  only with few classes and a calibrated classifier (Esuli, Molinari and Sebastiani 2020), while
  EM with BCTS is hard to beat (Alexandari, Kundaje and Shrikumar 2020). The method is chosen
  once, across all words, by QuaPy's artificial-prevalence protocol on the author's labels, since
  no method wins everywhere and per-word labels are too thin to choose by word (Schumacher et al.,
  JMLR 2025; ACC, PACC and EM stand out in Moreo and Sebastiani's re-evaluation; with small
  multiclass training sets Schumacher et al. find EM among the most robust). Plain
  classify-and-count is always reported as the baseline (with its adjusted variants it is near
  state of the art once tuned: Moreo and Sebastiani 2021; alone it is weaker on average:
  Schumacher et al. 2025).
- **Pooled or per-period error rates.** The methods assume a sense looks alike in every period;
  in labelled cells the gap between the stratified estimate and the pooled adjusted estimate
  tests that, and where it fails (computing *attention*, whose senses changed how they are
  expressed), error rates are estimated per period, given about ten labels per sense there.
- **Intervals** by bootstrap, resampling both the cell's usages and the author's labels, since
  error rates from few labels are noisy and the corrections amplify that (a synthetic test showed
  adjusted counts overshooting; with 100 training labels, analytic intervals covered only 64–75%,
  and bootstrapping training and test samples together restored 91–95%: Tasche 2019). Vaz,
  Izbicki and Stern (*JMLR* 2019) give intervals for two classes and large samples only, and show
  that the precision cannot beat what the labels alone allow. Where a key word's interval is too
  wide, more labels for that word only. Reference: *Learning to Quantify* (Esuli, Fabris, Moreo
  and Sebastiani 2023, open access).

*Why weights and the stratified reference* (simulated: three senses, 1,000 usages per cell, 30
labels per cell stratified by predicted sense, `second_review/share_estimators.py`): with the
weights ignored, adjusted counting was as biased as classify-and-count (−0.040 on a rise of
0.25); with them, unbiased. With a counter equally good in both periods, pooled adjusted counting
is the most precise (root-mean-square error 0.033 against 0.053 for the stratified estimate);
with a counter worse at the new sense in the early period, pooled adjusted counting is off by
+0.05 and the stratified estimate is unbiased; per-period adjusted counting and the stratified
estimate coincide. This simulation becomes a test of the quantification module.

**7.5 One of each.** One Claude labelling path (the Message Batches API `BatchLabeller`; the
command-line `Labeller` is deleted). One divergence function: scipy's `jensenshannon`, which
returns the Jensen-Shannon distance, the square root of the divergence (squared where the
divergence is meant, and every report names which). One sense-list format (the corpus format:
keys, definitions, examples, cues, exclusions, and a version number), into which the lists are
written before they are frozen (7a.4). One supervised classifier (counter 1b's), written once. Cue
words move to spaCy's `Matcher`.

**7.6 Fine-tuning.** The counters are used as they are first; fine-tuning (XL-LEXEME's top layers,
or a trained bi-encoder) only if they fall short on the author's labels, since labels are few and
fine-tuning risks learning the era again.

Replaces: two Claude labellers, two sense-list schemas, divergence written three times, logistic
regression written four times.

## Step 8: code layout

Inputs: `docs/review_codebase_2026-11-12.md` (its critical items: a `usages/` package, which
became `corpus/target_matching.py`, `corpus/usage_selection.py` and the catalog's usages dataset;
`corpus/cleaning.py` and an annotation module holding the spaCy pass; `lemmas.py` split; a `cli/`
package; the repeated helpers extracted: unit rows, divergence, torch device, cell reading, label
tables); steps 1–7.

**8.1 One job per package**, so code is easy to find and to change; pipelines compose the jobs;
only the catalog reads and writes data. Layers (one import-linter contract): `catalog/` at the
bottom; the job packages side by side, none importing another (`corpus/`, `embeddings/`,
`detection/`, `senses/`, `evaluation/`, `review/`); `pipelines/` composing them; `cli/` on top,
thin. `experiments/` sits outside the package; `common/` at the bottom only for helpers needed in
three or more places.

```
src/interpretant/
├── catalog/      abstracts.py, usages.py, vectors.py, words.py, schemas.py, provenance.py
│                 (one typed load and save per dataset; paths from the catalog file)
├── corpus/       openalex.py (fetching through pyalex, carrying out download plans),
│                 cleaning.py, deduplication.py, language.py, annotation.py,
│                 target_matching.py, usage_selection.py, lemma_exceptions.py, spelling.py,
│                 term_extraction.py
├── embeddings/   twec.py, ppmi.py, encoders.py
├── detection/    static.py, contextual.py, controls.py, significance.py, change_points.py
├── senses/       inventory.py, cues.py, induction.py, disambiguation.py, quantification.py,
│                 llm_labellers.py
├── evaluation/   benchmarks.py, synthetic_changes.py, known_answers.py, metrics.py
├── review/       judging pages, review samples, reading judgements back
├── pipelines/    download_abstracts.py, build_corpus.py, prepare_review.py,
│                 train_static_embeddings.py, encode_usages.py, detect_change.py,
│                 count_senses.py (one module per stage, and their settings)
├── cli/          one module per pipeline, same names
└── common/       only for helpers used in three or more places
experiments/      one runner per experiment, named by its question
tests/            mirrors src/
```

**8.2 Jobs receive data, never paths.** Jobs are plain functions of data; the pipelines and
experiment runners build objects and pass them in (dependency injection: Fowler 2004; "functional
core, imperative shell": Bernhardt, "Boundaries", 2012; Percival and Gregory, *Architecture
Patterns with Python*, 2020). This keeps the job packages independent where they meet:

- the spaCy pipeline is built in `corpus/annotation.py`, and `pipelines/count_senses.py` passes it
  (or its vocabulary) to `senses/` for cues and the `DocBin` cache;
- the encoding stage encodes the usages and the sense lists' examples and definitions into the
  same per-encoder store, same content key, and `senses/` gets those vectors from the pipeline;
- `evaluation/` supplies benchmark data and scoring; the experiment runner composes load →
  measure → score, so `evaluation/` imports neither `detection/` nor `senses/`;
- spaCy receives its patterns and exceptions as lists (`ruler.add_patterns`,
  `attribute_ruler.add`).

**8.3 The catalog** (`catalog/`, the data catalog: Kedro's `DataCatalog`; Fowler's Repository
pattern; Cockburn's ports and adapters, 2005) is responsible for the project's data at rest. It
knows every dataset by name in one catalog file (path, format, encoder and version), which
`dvc.yaml` reads through its templating (`vars:`), so each path is written once (values read
through `vars` are not tracked as parameters; the parameters are in their own files, step 9). It
loads and saves each dataset with the format's own function (Polars for Parquet, `DocBin`,
gensim's save and load, TOML and TSV for `curated/`), models the project trains included; it runs
the pandera checks on read and write; it hands out large datasets as streams (iterators), so
memory stays bounded. One typed `load` per dataset (a grouped load would lose the types under
strict mypy; bundle only if three stages load the same group); only pipelines and experiment
runners call it. It is not external services (pyalex in `corpus/`, the Batch API in `senses/`,
the Hugging Face Hub in `embeddings/`, whose results are saved through the catalog; model weights
are pinned by name and revision in the encoder table and kept in Hugging Face's cache), not
research decisions (usage selection is in `corpus/`), and not versioning (DVC's lock;
`provenance.py` shrinks to reading it for reports, or goes).

**8.4 Placements.** The OpenAlex client is `corpus/openalex.py` (corpus compilation includes
collecting the texts; a second source would become a module beside it), using `pyalex` (0.21,
February 2026, MIT, depending only on requests and urllib3) in place of the hand-written HTTP
client. `evaluation/` holds everything DWUG-specific, planted changes, the held-out known answers
and the metrics. The shifted-shares protocol stays in `senses/quantification.py`; `review/`
replaces `scripts/review/`. Model-calling code goes by job (substitutes and generated definitions
in `senses/induction.py`, the Claude and JEV labellers in `senses/llm_labellers.py`, encoders in
`embeddings/`), shared GPU and batching helpers in `common/`. Settings objects belong to
`pipelines/`; job functions take plain arguments. "Static" and "contextual" are flows across
packages and CLI command groups, not
packages. Interfaces only at the swap points: the encoder table, the counters, the cleaning
patterns file, the word tables and the settings files. Each pipeline reads as a table of contents
(one call per stage), with a one-page `docs/architecture.md`.

**8.5 Names.** Job modules are nouns, pipelines verb-plus-object phrases (`encode_usages`,
`detect_change`); the field's standard terms (`significance`, after the "statistically
significant detection" papers; `inventory`; `disambiguation`; `term_extraction`; `provenance`);
British spelling, no abbreviations; experiments named by their question, not by round letters.
The composing package is `pipelines/` (the author), the most widely recognised name (Kedro,
ZenML, scikit-learn), chosen over `workflows/` (collides with `.github/workflows/`) and `stages/`
(DVC's term); each module is one stage, and "the spaCy pipeline" is always named so inside
`corpus/`.

**8.6 Folders.**

- Generated data follows the Cookiecutter Data Science convention: `data/external/` for DWUG and
  other datasets, `data/raw/` for downloads, `data/interim/` for the annotation cache,
  `data/processed/` for tables and vectors, `models/` for trained static models.
- Hand-curated inputs live in a committed `curated/` folder, never overwritten by a run: the word
  tables (4.7), sense inventories, cleaning patterns, lemma exceptions, the spelling table,
  download plans, the held-out known answers. `curated/` changes only through review (the author):
  runs write proposals into `data/` (generated controls, lemma exceptions from spaCy's errors),
  and a file enters `curated/` only after the author approves it; the spelling table is committed
  once after checking and then left alone.
- Everything a person judged lives in `curated/` and is committed (check 1a's answers, the gold
  pairs, the head-to-head choices, the coming labels and reviews): human judgements are the one
  thing that cannot be regenerated. Raw model outputs stay out.
- Reports: each experiment runner writes `reports/<experiment>/report.md` with its key tables
  (from Polars, as Markdown), matplotlib figures (`reports/<experiment>/figures/`) and the data
  and settings that produced it (step 9). The `web/` explorer (a Next.js app, 118 tracked files)
  is deleted with the step 1 deletions, after `docs/experiments.md` records what it showed; a
  presentation is designed again once there are new results.
- Notebooks: plain Jupyter notebooks in a committed `notebooks/` folder, named by order and topic
  in the Cookiecutter Data Science manner (`1.0-gd-attention-usages.ipynb`), with `nbstripout` as a
  pre-commit hook so no outputs or data reach git; `ipykernel` and `nbstripout` as
  development-only dependencies (chosen over marimo and plain-Python cell files as the most common
  setup). Notebooks are scratch: code needed twice moves into the package, anything that matters
  moves into an experiment runner, and no result comes from a notebook.
- `scripts/` disappears once its contents are rebuilt (the evidence scripts with it; git keeps
  them). The layout is not a separate refactor but the shape the rebuild builds into, phase by
  phase, a commit per phase.
- `docs/` is reorganised into `docs/reference/` (living documentation such as `spacy.md`,
  `methods.md`, `related_work.md`, `architecture.md`, `datasheet.md`), `docs/records/` (the
  predictions files, diagnostics, `experiments.md`) and `docs/design/` (drafts and this agenda),
  after a pass over the 23 documents (housekeeping; it touches every link, so it pairs with step
  10).

**8.7 Replacement audit** (the rule for every step, applied to existing code; extended as code is
touched):

| Hand-rolled today | Library replacement |
|---|---|
| Unit-normalised rows, written seven times (`grid._unit_rows` and others) | `sklearn.preprocessing.normalize` (handles zero rows) |
| `metrics.cosine_distance` | `scipy.spatial.distance.cosine` |
| Sentence regexes (`quality._SENTENCE`, `senses._SENTENCE_END`) | spaCy `senter` with special cases (4.9) |
| Six regex tokenisers | spaCy tokens (3.1) |
| Mojibake table (`classifier.MOJIBAKE`) | affected abstracts dropped, found by a signature list in the patterns file (2.3) |
| URL and DOI regexes | spaCy `like_url` and `url_match` (2.3) |
| Language checks (`looks_english`, `english_share`, `foreign_share`) | lingua (2.4) |
| Duplicate rules | `datasketch` MinHash (2.5) |
| Judgements parsed from Markdown with a regex (`gold_pairs.py`) | a TSV file read with Polars |
| Bootstrap intervals and shuffle tests | `scipy.stats.bootstrap`, `scipy.stats.permutation_test` (6.3) |
| Already library-backed, kept | Jensen-Shannon distance (`scipy.spatial.distance.jensenshannon`), NNLS (`scipy.optimize.nnls`), community detection (`networkx`; Leiden replaces Louvain, 7a.2), KMeans and silhouette (scikit-learn) |

## Step 9: reproducible runs

**DVC** for pipelines and provenance, after comparing it with Snakemake (runs steps, no data
versioning or lock file), Kedro (a full framework to restructure around), Hydra (configuration
only; pydantic settings already exist), MLflow (experiment tracking at a scale the Markdown
reports cover) and a manifest of the project's own (a re-implementation).

- `dvc.yaml` lists the stages, one per module in `pipelines/`: each stage's command is the CLI
  (`interpretant build-corpus`, …), its dependencies the data, the `curated/` files and the exact
  modules it uses (the spaCy pass: the annotation, cleaning and language modules, the patterns
  file and its settings; not `senses/`), so a change reruns only what it can affect; the list
  sits beside the stage definitions and is checked when stages change. Its outputs are what it
  writes.
- Settings live in parameter files DVC tracks (`params.yaml` or TOML), loaded by the pipelines'
  pydantic settings objects, so a changed seed, cap or floor reruns exactly the stages that use
  it. Caps, the floor and the download plan's sizes start as provisional values there and may be
  revised by the experiments.
- `dvc repro` replaces every "skip if the file exists" check (the compass, cells, shards, caches).
- **What is costly or cannot be remade is kept.** DVC deletes a stage's outputs before running it
  "unless `persist: true` is used in `dvc.yaml`"; unchanged, that would wipe every stored vector
  when any upstream input changed and rerun everything when a stage depended on all of `src/`. So
  the vector store and the annotation cache (`DocBin`) are `persist: true` outputs: encoding adds
  only missing keys, and vectors for usages that no longer exist stay unread until a deliberate
  clean-up command removes them. These outputs are their inputs plus earlier runs, which is safe
  because a vector is found by the hash of the exact text the encoder read, in a folder named by
  the encoder and its version (4.6).
- Downloads are not a stage: an approved plan runs by hand and its output is registered with
  `dvc add data/raw/...`, versioned by hash and never regenerated by DVC (OpenAlex's seeded
  samples can change as records are added, abstracts withdrawn or works reclassified, and they cost
  credits).
- `dvc.lock`, committed, is the provenance record: hashes of each output's inputs, settings and
  code; reports cite the commit and the lock entry they came from; it records the word tables'
  version, since they are dependencies of the stages that read them.
- Large data: the cache links outputs with reflinks (copy-on-write; APFS on the Mac and btrfs on
  bazzite), so it does not double disk use; huge regenerable outputs may be `cache: false`, which
  also turns off DVC's run cache for that stage.
- No remote at first; an SSH remote (`dvc-ssh`) to move data between the Mac and bazzite is added
  when the rebuild needs the GPU.
- Caveats: stewardship moved from Iterative to lakeFS (November 2025), and the last release was
  March 2026 while commits continue; if DVC were abandoned, the pipelines still run as CLI
  commands and only the rerun logic and lock file would need replacing.

**9.1 Stage order.** The rebuild runs in this order; each stage starts from provisional values
and nothing waits on work that comes later:

1. Downloads from the approved plan (4.8), registered with `dvc add`.
2. The spaCy pass: repair, annotation, the `SpanRuler`, lingua; the `DocBin` cache (step 2, 4.9).
3. Corpus-level rules: template sentences, repeated sentences, near-duplicates, the author cap;
   the abstracts table (2.3, 2.5, 4.6).
4. The corpus review and sign-off (4b), before any experiment uses the corpus.
5. The lemma review (3.3); exceptions applied to the cached documents (3.6).
6. The derived views: the static text and the contextual sentences (2.2).
7. Matching and the usages table; one mention drawn per abstract (4.1–4.5).
8. The gap report; a top-up download if a cell is under the floor, after which stages 2–7 run for
   the new abstracts and the corpus-level rules rerun in full (4.8).
9. Control candidates generated and approved (step 5); encoding of targets, controls, sense-list
   examples and definitions (4.6, 8.2).
10. The static models (step 6b), beside stage 9.
11. Development experiments (step 6) on the synthetic set, the benchmark and the development
    words.
12. Sense lists frozen (7a.4); the author's labels (7.2); counting (7.4).
13. The confirmation plan committed (6.2); held-out labels; the confirming run.

## Step 10: terminology pass (later)

Names should follow common NLP and data-project usage, as *static* and *contextual* replaced
"type track" and "usage track" (the author). One pass over code, docs and data paths, after the
rebuild's names settle, listing each project-made term, the standard one, and where it is used.
Candidates so far (to be checked against the literature, not assumed):

| Project term | Possible standard term |
|---|---|
| cell (a field in a decade) | corpus slice, time bin, subcorpus |
| grid (both the corpus of cells and the TWEC models) | two names, one per meaning |
| layer (`openalex`, `openalex_full`) | source, subcorpus |
| known answers; quiet transitions | gold-standard changes; stable (negative) cases |
| planted changes | simulated or synthetic semantic change |
| sense shares | sense distribution |
| J (across minus within) | half the energy distance with squared distances, equal to the maximum mean discrepancy with a linear kernel (6.3); a short name is chosen in the pass |
| word stream (the static pipeline's text) | normalised token stream, normalised text |
| rounds, rungs | experiments; calibration stages |
| usages-table columns | DWUG's names where they mean the same (*context*, target offsets, *grouping* for periods) |

## The author's tasks

Everything the author does by hand, in stage order (9.1). Sittings are about 15 minutes.

| Task | Step | When | Items | Time |
|---|---|---|---|---|
| Approve the download plan | 4.8 | Before stage 1 | Three thin words, per year | One sitting |
| Corpus review | 4b | Stage 4 | About 385 judgements, plus second samples for fields that trigger | About 1 hour 5 minutes |
| Sign off the datasheet and the review | 4b | Stage 4 | One summary | One sitting |
| Lemma review | 3.3 | Stage 5 | The 100 most frequent types, every case touching a word in the word tables, 30 from the tail | About two sittings |
| Approve the generated controls; blind reading where a control is in doubt | 5 | Stage 9 | About 60 proposed per comparison | One sitting per few comparisons |
| Review the proposed senses and freeze the lists | 7a.4 | Stage 12 | Eight words | Not yet sized |
| Refine check 1a ("which sense?") | 7.2 | Stage 12 | 183 items | About 45 minutes |
| New labels | 7.2 | Stage 12 | About 200 | About 1–1.5 hours |
| Review the held-out set | 6.2 | Before stage 13 | About 40 words, a few usages per decade each | Not yet sized |
| Held-out labels | 6.2, 7.2 | Stage 13 | Not yet sized | Not yet sized |
| In reserve: verify Claude's development labels | 7.2 | Only for uncertain words | About 30 blind, then the rest | About 30 minutes |

The sized tasks come to about 4 hours; three are not yet sized.

## Tests the rebuild must include

- **Data checks** (pandera, on every read and write): usage ids unique; content keys equal to the
  hash recomputed from the text and offsets; offsets pointing at the target (4.5, 4.6).
- **Invariants:**
  - no target word is mapped away by the lemma exceptions (3.2);
  - no word is both target and control in a comparison (4.7, step 5);
  - held-out words appear in no development output (6.2);
  - an abstract's drawn mention does not change when other abstracts are added or removed (4.3);
  - the era eraser is never fitted on the word it is applied to (step 6, experiment 6).
- **Round trips:** a `DocBin` round trip keeps sentences, norms, span groups and sentence
  languages (4.6).
- **Statistics module** (6.3): J's pairwise and mean-vector forms agree; the permutation floor
  against Benjamini–Hochberg; J's chance level by n; the rank rule against the interval rule on
  simulated words.
- **Quantification module** (7.4): weighted and unweighted adjusted counting, and the stratified
  estimate under a counter that is worse in one period, on simulated cells.
- **Counter checks** on every evaluation (7.3): anachronism and stability.
- **One-off audits**, not kept as tests: the old regexes against the new patterns (2.3); the
  small against the transformer model (4.9).

## Open questions

- The static model (6b), after experiment 4.
- Caps and the floor (500), provisional; the download plan's credit cost, to be re-estimated
  (4.8).
- The small or transformer model for target sentences, by the one-field comparison (4.9).
- Thresholds from the sample study and the sample audit: MinHash similarity (2.5), the share of
  non-English that drops an abstract (2.4), which patterns the template rule makes redundant
  (2.3).
- Whether the current cells carry OpenAlex's new field assignments, from the download records
  (4b).
- The rank test's level, frozen in the confirmation plan (6.3); whether 20 documented changes can
  be found for the held-out set (6.2).
- The six-word shortlist's role, after checking its history (4.7).
- Whether J stays independent of frequency for rare words: a downloaded target's controls are far
  more frequent than it (5% or more of abstracts against about 0.15% for *prompt*), beyond the
  range checked (words in at least 100 abstracts per cell); experiment 2 tests it on the words
  already downloaded (step 5).
- `concept-erasure` or LEACE's closed form, at experiment 6.
- Dependencies agreed but not installed, each installed when its step starts and after the
  author's yes on the stack: DVC, QuaPy, pyalex, `correlation-clustering`, networkx 3.7 or later,
  `ruptures`, `ipykernel`, `nbstripout`.
- Sources read only as abstracts or records (their full texts were not opened): Efron 1987,
  DiCiccio and Efron 1996, Simmons, Nelson and Simonsohn 2011, Kish 1965, Kohavi, Tang and Xu
  2020, Frantzi, Ananiadou and Mima 2000, *Learning to Quantify*, Pustejovsky and Stubbs.
- Step 10 (terminology, including J's short name) and the docs pass, after the rebuild.

## Housekeeping (outside the steps)

- [x] pyarrow: nothing imported it and nothing else depended on it; removed (2026-11-12).
- [x] Commit this agenda, `pipeline_map.md`, the four design drafts and the code review
  (2026-11-12).
- [x] The scripts behind the agenda's numbers committed as a record (`dd4a5aa`).
- [ ] Uncommitted loose files: `data/sense_lists/round_a_prime_v2_draft.toml`, the untracked
  round E and round Q label files (to be sorted into `curated/`).
- [ ] `related_work.md`: add the sources cited in the steps (the second review's verified links
  are kept with its sources, outside the repository).
- [ ] `docs/experiments.md`: written before any step 1 deletion.
- [ ] Push: the local commits wait for the author's go.
- [ ] Devlog 07: waiting for the author's review.
- [ ] bazzite's copy of the repository: behind the Mac; sync before any GPU run.
- [ ] Docs pass: go through the 23 documents in `docs/`, merge what overlaps, retire what is no
  longer relevant, sort the rest into `reference/`, `records/` and `design/` (8.6), with step 10.

## Reference: what moves to spaCy

| Moves to spaCy | Stays outside |
|---|---|
| Tokens and sentences (`senter` with abbreviation special cases), URL and e-mail detection, possessives and contractions, spellings (`norm_`), target matching (`Matcher`, `PhraseMatcher`), cue words (`Matcher`, `DependencyMatcher`), lemma exceptions (`AttributeRuler`), the name filter (entities), removal spans (`SpanRuler` span groups, 2.2), the annotation cache (`DocBin`), language (lingua as a component) | OpenAlex downloads, template sentences (using spaCy's sentences), near-duplicates (datasketch), mojibake (affected abstracts dropped), static models (gensim), encoders (transformers), the classifiers (scikit-learn) |

## Reference: merges

| Today | After |
|---|---|
| 4 cleaning rule sets, 3 language checks, 2 sentence splitters, about 6 regex tokenisers, 2 URL regexes | 1 cleaning step in the spaCy pass, with lingua |
| 6 duplicate and shared-text rules | work id plus MinHash, template sentences and repeated sentences |
| 2 lemma maps | spaCy's lemmas plus one exceptions file in the `AttributeRuler` |
| 5 copies of the usage key, plus 2 schemes | the usage id and the content key, each defined once (4.5) |
| Divergence 3 times, unit rows 7, logistic regression 4, torch device 4, cell reading 6 | one library helper each |
| 2 Claude labellers | 1 |
| 2 neighbour implementations | 1 |
| 4 control-word files | one generated set per comparison (step 5) |
| 2 sense-list schemas | 1, versioned |
