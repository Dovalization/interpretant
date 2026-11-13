# Decision agenda before the store rebuild

*Draft, 2026-11-12. The order in which to decide strategy before more code is written. It draws
on `docs/pipeline_map.md` (what every step does today), the four design drafts
(`design_store_rebuild.md`, `design_data_acquisition.md`, `design_detection_experiments.md`,
`design_sense_counting.md`) and `docs/review_codebase_2026-11-12.md`. Each step lists its
inputs, the decisions, and what they let us delete, merge or move to spaCy. Recommendations are
Claude's; the decisions are the author's. Record each decision under its step as it is made.*

Terms (2026-11-12): the **static** pipeline gives one vector per word per cell (TWEC, PPMI); the
**contextual** pipeline gives one vector or sense label per occurrence in its sentence
(XL-LEXEME). They replace "type track" and "usage track".

Rule for every step (the author, 2026-11-12): use spaCy's features, or another maintained
library, wherever one fits, and replace hand-rolled code with them, existing code included. Own
code stays only where no maintained option exists or the library is measured to be wrong on this
corpus, and then it is kept small and the reason recorded. Unmaintained libraries do not count.
A new dependency must be reasonable: when a library would serve one small job that a few lines
of the project's own (or the standard library) handle as well, the project's own code wins.
When scipy and scikit-learn both offer a job (scikit-learn is built on scipy), the choice is by
kind of job and then the same everywhere: scipy for maths and statistics (tests and intervals,
rank correlation, linear algebra, optimisation, the distance between two vectors); scikit-learn
for machine-learning building blocks and batch work on matrices (classifiers, cross-validation,
preprocessing such as `normalize`, pairwise similarities over many vectors, nearest neighbours).

Review of steps 0–2 under this rule (2026-11-12):

| Step | Hand-rolled part | Verdict |
|---|---|---|
| 1a | TWEC trainer on gensim's internals | `cade` (the packaged method) is inactive since 2020, pre-alpha, needs a patched gensim: does not count. The library route (cells trained apart with gensim, aligned with scipy's `orthogonal_procrustes`) is a method change, decided in step 6b |
| 1a | Neighbour search for neighbourhood shift | scikit-learn's `NearestNeighbors` (cosine) |
| 1a | Frequency bands imitating pandas' `qcut` | Polars' `qcut` and `rank`; the old tables are retired |
| 1a | PPMI construction | kept: no maintained PPMI library; the SVD is already scipy's |
| 1b | APD and J | scikit-learn's `cosine_similarity` (one matrix product, much faster than scipy's `cdist` on large sets), when rebuilt |
| 1b | Shuffle tests | scipy's `permutation_test`, when rebuilt |
| 2.2 | Mojibake, control characters, ligatures | `ftfy` considered and declined: it would only repair 461 abstracts (0.03%) instead of dropping them; NFKC and `unicodedata` categories cover the rest |
| 2.2 | LaTeX | `pylatexenc` considered and declined for now: 0.9% of abstracts, and a risk of reading dollar amounts as maths; the token flag stays, revisited if the sample audit shows harm |
| 2.3 | Per-sentence logic for mixed abstracts | lingua's own confidence values and `detect_multiple_languages_of` (experimental, so validated on the sample), restricted with `from_languages` |

Principle: decide what the project keeps doing first, because that decides what code survives;
then the foundations every remaining step depends on (cleaning, words, usages); then the
experiments; the code layout last, since it mostly follows.

## Step 0: delete what is already dead (no decision needed)

Nothing on a live path reaches these; each goes with the tests that only exercise it.

- [x] `drift/direction.py` 94–315 (about 220 lines, test-only)
- [x] `visualization/figures.py`: `known_answer_ranks`, `readings`
- [x] `drift/metrics.py`: `_top_k_neighbors`, `build_decade_index`
- [x] `predictions.py`: `load_vectors`, `word_count`, `overlap` (test-only, checked 2026-11-12)
- [x] `drift/senses.py`: `sense_shares`, `jensen_shannon`, `context_words`, `examples`
- [x] `export.py`: `METHODS` and the `__main__` block
- [x] `corpus/__init__.py`: the unused re-export
- [x] the 95 stale `noqa` comments
- [x] data: the old `phrases.*` files in `data/processed` (the unlemmatised layer's phrase
  model, set aside in `data/retired/phrases_unlemmatised/`, not deleted)

Decision (2026-11-12): delete. Done in `ed94882` (1,036 lines out; 143 unused `noqa`;
`drift/direction.py` keeps `assignment_runs` and `present`).

Principle stated by the author: code that will be rebuilt or is no longer used goes; git keeps
it. No frozen-but-runnable code.

## Step 1: what stays in the project

**1a. The static pipeline** (TWEC and PPMI grids, `drift grid`, perplexity and decompose, held-out,
separability, direction, the web explorer, the figures). Options: a frozen record (models,
tables, explorer and figures kept, no new work, not moved to the new cleaning); partly retired
(grid, explorer and perplexity kept, which the README figures use; held-out, separability and
direction deleted); rebuilt on the new cleaned text.

*Recommendation (first version, withdrawn):* a frozen record. Held-out, separability and
direction turned out to support README findings (rounds S, V, N and the per-cell check), and the
author ruled out keeping code only so old results can be rerun.

Decision (2026-11-12): whole-field comparison (which field's language reads like which,
convergence) leaves the project; comparing fields word by word stays (methodology job 1, roadmap
rung 3). The direction of HCI's change was closed by round W on 2026-11-10; the rest of the
field-level work was never stopped, only left out of the fresh start's jobs.

- Delete: `drift perplexity` and `decompose`, `drift heldout`, `drift separability`,
  `drift direction` (with `perplexity.py`, `heldout.py`, `separability.py`, `direction.py`, their
  tests and CLI commands); nothing in the contextual pipeline imports them.
- Keep: TWEC, PPMI and `drift grid`, to rerun on the new cleaned text (detection experiment 4,
  and the only instrument that scans every word).
- Delete for now: the web explorer's export and the figures code; rebuilt when there are new
  results.
- Set aside: the old results (`data/drift` tables, `models/`) in `data/retired/static/`.
- Record first: every finding as a past experiment (what, method and settings, downsides
  including the older cleaning, the commit that produced it) before any of this is deleted.

**1b. Past rounds' scripts.** Precedent: on 2026-11-11 the 19 old scorers were deleted, since
the fresh start set their rules aside; results stay in `docs/` and `data/`.

| Proposed fate | Scripts |
|---|---|
| Live: replaced during the rebuild, the old script deleted in the same commit | `score_pairs_known` (round X; the store's re-baseline), `score_pairs_benchmark` (round U on DWUG; calibration), `check_planted_pairs` (planted changes), `score_labels_benchmark` (DWUG labelling; the human benchmark), `build_vector_store`, `fetch_sense_words`, `pilot_labellers` (its `usage_id`, `usages` and DWUG helpers feed the store's dev set and two scripts) |
| Keep the data, delete the script | `gold_pairs` (check 2b; its Markdown judgements converted to a TSV in `curated/` first, step 8), `check_known_presence` (check 1a; its marker words move first, since `check_planted_pairs` imports them): the author's judgements are what matters |
| Delete: field-level (step 1a) | `score_venue` (K), `score_decomposition` (G), `measure_label_noise` (S) |
| Delete: closed line | `score_labels_wordnet` (round Z; cites a missing script) |
| Decide with step 7a (finding senses is open) | `score_labels_induced` and `drift/induction.py` (round Q's substitutes), the clustering half of `drift senses` (SciBERT, KMeans, silhouette; failed in rounds C, C′, C″ in that one design) |
| Decide with step 7 | `score_labels_classifier`, `score_labels_corpus`, `score_labels_head_to_head`, `score_labels_known`, `finetune_classifier`, `dev_harness`, `scripts/review/` |
| Decide with step 4 | `scripts/discovery/` (made the long list that sets the store's scope; single-letter names; one input has no producer) |
| Decide with step 5 | `select_control_words`, `select_cross_field_controls`, `select_sense_controls` (three of the four control sets; the fourth has no script) |

Decision (2026-11-12): the table as it stands. Groups "delete" and "keep the data" go once their
results are recorded and check 1a's marker words have moved; the live scripts go with their
replacements; the rest wait for steps 4, 5, 7a and 7.

**The record of past experiments** is a separate document, `docs/experiments.md`, not a README
section: one entry per experiment with what it found, the method and settings, its downsides
(including the older cleaning), its status (holds, withdrawn, untestable) and the commit that
produced it; the predictions files stay as the detailed ledger, and the README points to the
record. Written before any step 1a or 1b deletion.

## Step 2: cleaning and duplicates

Inputs: `pipeline_map.md` stages 2–3; store draft decisions 3, 6, 7.

The single spaCy pass gives two views of each abstract: the cased sentence, with punctuation and
numbers, for the contextual pipeline (the encoders read real sentences), and a normalised word
stream (lowercase, no punctuation) for the static one. Keeping rules decide whether an abstract
or a sentence is kept at all, for both views; normalisation applies to the word stream only.

**2.1 One cleaning path for both pipelines**, inside the spaCy pass, replacing four rule sets
that apply on different paths today. *Recommendation:* yes.

**2.2 Which rules survive.**

| Rule | Today | Recommendation |
|---|---|---|
| Catalogue and citation records, web-page scrapes | Rule set 1 (regex) | Keep |
| Exam-dump spam | Quality v2 | Keep |
| Fewer than 5 words | Rule set 1 | Keep |
| Template sentences (shared by 20+ abstracts); an abstract mostly made of them dropped | Quality v2, store only | Keep, counted over every source and after footers are stripped (today: counted before, matched after) |
| JSTOR and copyright footers; the leading "Abstract" label | Rule set 1; the contextual path | Keep |
| Author cap, 1% of a cell | Rule set 1; falls to 1 work in cells under 200 | Keep, with a floor (for example at least 5 works) so small cells are not gutted |
| LaTeX math | `$…$` regex, which also deletes prose between two dollar amounts | Keep, stripping only spans with LaTeX markers (`\`, `^`, `_`) |
| URLs and DOIs | Two different regexes | spaCy's `like_url` plus one DOI pattern |
| Mojibake (broken encodings, "Ã©") | A hand-written table | `ftfy`, an established library for this (a new dependency, so it needs the author's yes on the stack); or keep the table (decided below: neither; affected abstracts dropped) |
| Soft hyphens, control characters | The contextual path only | Keep, for both pipelines |
| Folding to ASCII | Deletes non-Latin characters outright | Fold accents (café → cafe) in the word stream only; non-Latin sentences go by the language check instead |

*2.2 decided (2026-11-12),* after reading spaCy's documentation (Matcher, SpanRuler, custom
components and extensions, `url_match`, `nlp.pipe` with `as_tuples`) and counting each problem
in the 1,363,172 unique raw abstracts (curly quotes 29%, URLs 1.4%, LaTeX commands 0.9% against
dollar math 0.15%, copyright 0.45%, soft hyphens 0.28%, DOIs without a scheme 0.19%, control
characters 0.17%, ligatures 0.17%, mojibake 0.03% (461), HTML almost none):

- *Before spaCy, repair only:* NFKC (which also expands ligatures), control and format
  characters removed by Unicode category, soft hyphens removed; abstracts with mojibake dropped
  (no `ftfy`; the hand-written repair table goes), detected by a short list of the usual
  mis-decoding signatures (such as `Ã©` and `â€™`) kept in the cleaning patterns file, so the
  detector is data and is checked in the sample audit.
- *In the spaCy pass:* DOIs as single tokens through `url_match`; lingua (per abstract, per
  sentence only where an abstract comes out mixed: 2.3); one
  `SpanRuler` patterns file (data, reviewable) labelling every text rule (catalogue and citation
  records, web-page scrapes, spam, footers, copyright, the leading label, LaTeX tokens, URLs,
  e-mails), so every removal carries its reason. LaTeX is dropped from the word stream only;
  contextual sentences keep it, so target offsets do not move.
- *After the pass, corpus-level rules:* template sentences (spaCy's sentences, digits masked,
  counted with Polars over every source); near-duplicates by MinHash; the author cap with a floor.
- *Sample audit first:* run the old regexes and the new patterns side by side and confirm they
  catch the same abstracts before the regexes are deleted; count what the template rule alone
  catches and delete patterns it makes redundant; check how the mojibake abstracts spread over
  cells before dropping them.

**2.3 Language.** Three checks today. *Recommendation:* lingua alone, per sentence; drop
non-English sentences, and an abstract when most of its sentences are not English (the share
fixed after reading a sample). (Decided below: per abstract first, per sentence only for mixed
or uncertain abstracts.)

**2.4 Duplicates.** Six mechanisms today. *Recommendation:*

- Near-duplicates (a preprint and its published version) by MinHash, keeping the earliest and
  recording each removal; the threshold from the sample study (store decision 3). Exact
  duplicates become a special case.
- The same work in two cells: some overlap is legitimate (AI inside computer science; a full
  cell holds its sample cell), but in a comparison between those two fields it makes them look
  alike. Keep the work in every cell it belongs to, record its cells in the abstracts table, and let
  each measure decide whether to leave shared works out. Today the contextual rounds drop them
  from both cells and the store keeps them; this makes the choice explicit.

**2.5 Hyphenated words** (store decision 6). *Recommendation:* kept whole, as today, on purpose
(*self-organising*, *meta-analysis*).

**2.6 Numbers** (store decision 7). *Recommendation:* digit-only tokens dropped from the word
stream, kept in the contextual sentences; *1990s* is not a number to spaCy and stays.

Decision (2026-11-12): as recommended, with these refinements.

- 2.1: one cleaning path for both pipelines, inside the spaCy pass.
- 2.2: as recorded above.
- 2.3: lingua per abstract first; per sentence only for abstracts that come out mixed or
  uncertain (short sentences are where detectors err); sentences without letters are not judged;
  lingua limited to English and the languages found in the corpus.
- 2.4: near-duplicates by MinHash, threshold from the sample study; a work in several cells kept
  in each, its cells recorded in the abstracts table; leaving shared works out is one library helper
  that every measure uses, not a rule each measure writes.
- 2.5: hyphenated words kept whole. Consequence for step 4: a lemma matcher no longer finds
  *attention* inside *self-attention*, as `word_pattern` does today; whether compounds count as
  usages of their head is decided there.
- 2.6: digit-only tokens dropped from the static word stream, kept in contextual sentences.

## Step 3: words (tokens, lemmas, phrases)

Inputs: `docs/spacy.md`; `pipeline_map.md` stage 4; the lemma audit (2026-11-12).

**3.1 Tokens.** spaCy tokenises everywhere; the static word stream is written from its tokens.
`preprocess_text` retires, and the regex tokenisers retire as their users are rebuilt (cue
words in step 7, control selection in step 5). *Recommendation:* yes.

**3.2 One lemma map, one exception list.** `TARGET_VARIANTS` (26 hand-written plurals and
spellings of target words, applied before counting) and the map's protected target list exist
because of each other; the map now handles plurals, irregulars and spellings itself, and
keeping `TARGET_VARIANTS` would leave two maps that can disagree (an earlier claim here, that it
maps *mediums* wrongly, was a mistake: *mediums* → *medium* is a regular plural; *media* is the
form the override file keeps). *Recommendation:* drop both;
`data/lemma_overrides.tsv` (moving to `curated/lemma_exceptions.tsv`, step 8) stays the only
exception list, loaded by the map and by the
`AttributeRuler` for lemmas in context; after relearning, check that no target word is mapped
away.

**3.3 How the map is applied.** Today by form alone: one choice per word form, so *leaves* is
*leave* everywhere. The shared pass gives every token its tag, so the map can be applied by form
and tag: *leaves* read as a plural noun → *leaf*, as a verb → *leave*; tokens read as anything
else (*learning* as a gerund) are untouched. *Recommendation:* by form and tag.

**3.4 Spellings.** spaCy's `norm_` table first, filtered to spelling variants (it also holds
informal norms such as *cos* → *because*); the map's own rules as the backstop for derived words
the table lacks (*personalisation*, *operationalise*, and *judgement*). The table is undocumented,
so it is read once, filtered, and saved as a versioned data file the map uses, so a spaCy update
cannot change it silently. *Recommendation:* yes.

**3.5 Phrases.** The static pipeline stays and is rerun (step 1a), so it needs phrases.
*Recommendation:* gensim's `Phrases` (NPMI) on the new word stream, after the lemma map, as now;
spaCy's `PhraseMatcher` for phrase targets in the contextual pipeline. Noun chunks were
considered and set aside: they return whole noun phrases (*the new transformer model*), not
collocations.

*3.5 decided (2026-11-12),* after reading the current output (3,502 phrases: real terms, but
also boilerplate and citations such as *your_browser* and *york_routledge*, other languages,
non-terms such as *lend_itself* and *1980s_and_1990s*, and terms cut to pairs such as
*systolic_blood*) and counting what joined target phrases take from their heads (in AI, *neural
network* is 30% of *network*'s uses in the 1980s and 50% in the 1990s; *artificial intelligence*
is 77–86% of *intelligence*'s):

- Phrases are found by term patterns (adjectives and nouns ending in a noun, up to four words)
  with spaCy's `Matcher` on the shared pass's tags and morphology, scored by C-value with a
  frequency floor in Polars (see the step 3 decision below); not gensim's `Phrases`. Broken-word
  repair stays.
- The static stream never joins a phrase that contains a single-word target, target phrases
  included, so a target keeps all its uses (the current rules contradict each other: learned
  phrases are blocked, target phrases always joined, and the grid's *network* lost the uses that
  changed).
- Phrase targets (*neural network*, *artificial intelligence*) are studied in the contextual
  pipeline: the `PhraseMatcher` marks each usage as inside a compound or not, and a measure runs
  on all usages, only those outside, or only the compound. Step 4's compound question
  (*self-attention*) uses the same mechanism.
- gensim stays only to train the skip-gram vectors behind TWEC (spaCy trains no word vectors);
  whether the static pipeline keeps skip-gram or runs on PPMI-SVD alone is decided after
  detection experiment 4 (step 6b).

**3.6 The review list.** The lemma map's remaining wrong merges (cosmos, thanks, mises,
forceps; merges whose singular means something else, such as novels, currents, logos; names
such as hans, rogers) are listed from the new word stream, where the apostrophe artefacts are
gone, and the approved ones go into the override file.

Decision (2026-11-12), revised under the rule above after measuring spaCy's own lemmas against
the voted map over 22.6 million plural-noun and third-person-verb tokens: where both merge,
spaCy differs on 1,136 types (0.46% of tokens), about half of them spelling only (spaCy keeps
British forms) and the rest systematic errors (*senses* → *sens*, *uses* → *us*, *analyses* →
*analyse*, *motives* → *motif*, *doses* → *dos*, unknown words cut to *identifie*,
*smartwatche*); the other disagreements are the project's own choices (auxiliaries, the
lexicalised plurals of the override file).

- 3.1: spaCy's tokens everywhere; `preprocess_text` and the regex tokenisers retire.
- 3.2 and 3.3: the word stream takes **spaCy's in-context lemma** for plural nouns and
  third-person verbs, found by morphology features (`Number=Plur`; `Person=3`, `VerbForm=Fin`),
  so the form-and-tag behaviour comes with it (*leaves* → *leaf* as a noun, *leave* as a verb).
  Auxiliaries stay as they are (*is*, *has*, *does*); ordinary verbs merge (*goes* → *go*).
  Exceptions go through spaCy's `AttributeRuler` from one file: the reviewed override list
  (lexicalised plurals) and corrections for spaCy's measured errors, generated from the votes and
  reviewed in 3.6. The voted map stops running on the text; its rules survive only as the tool
  that finds spaCy's errors. `TARGET_VARIANTS` and the protected list go; a check confirms no
  target word is mapped away.
- 3.4: spelling by spaCy's `norm_` table applied to the lemma, filtered to spelling variants and
  saved as a versioned data file; the project's rules as backstop for derived words.
- 3.5: as recorded above, with candidates from spaCy's `Matcher` on morphology and part of speech
  and **C-value** scoring (Frantzi, Ananiadou and Mima 2000; to be read and added to
  `related_work.md`), which discounts candidates nested in longer terms. No maintained library
  scores terms: PyATE was archived in June 2023, textacy's last release was April 2023.
- 3.6: the review list (wrong merges and spaCy's lemma errors) built from the new word stream.

## Step 4: usages and the store

Inputs: store draft decisions 1, 2, 4, 5; the data-acquisition draft; `pipeline_map.md` stages
7–8; steps 2 and 3. A usage is one occurrence of a target word in its sentence.

**4.1 Matching a target.** Today `word_pattern` (`\bword(s?)\b`, any case): the word and its
regular plural only, no part of speech. *Recommendation:* spaCy's `Matcher` on the lemma (so
irregular plurals and British spellings match through step 3), the `PhraseMatcher` for phrase
targets; every usage records its part of speech and morphology, and nothing is filtered at
matching: a measure or the sense counter decides whether verb readings count (*prompt* as a
verb), as with shared works in 2.4.

**4.2 Compounds** (from 2.5 and 3.5). Hyphenated words stay whole, so *self-attention* is one
token. *Recommendation:* match it (a `Matcher` pattern on the lower-case form ending in the target)
and record it as a compound usage, with the compound; multi-word compounds (*neural network*)
are marked the same way through the `PhraseMatcher`. Measures choose all usages, only those
outside compounds, or only one compound.

**4.3 One occurrence per abstract** (store decision 1). Today the first match in each abstract.
*Recommendation:* the usages table lists every occurrence (matching is cheap); one per abstract is
drawn at random with a recorded seed for encoding and measures, so abstracts stay independent
and the draw is not tied to the opening sentences, where an abstract's framing plausibly sits.
Other occurrences can be encoded later if a measure needs them.

**4.4 Context** (store decision 2). *Recommendation:* the spaCy sentence (XL-LEXEME was trained on
sentences); a long sentence is cut by the encoder's own window centred on the target, as now.

**4.5 Usage key.** Today defined five times. *Recommendation:* one function, a hash of the
sentence text and the target's offsets (as now), so a usage whose text the cleaning changes gets
a new key and a new vector; the work id is a column, not part of the key.

**4.6 Store format.** *Recommendation:* the usages as one Parquet table (Polars), one row per
usage: key, word, compound, part of speech, morphology, lemma, field, decade, year, work id, its
cells (2.4), sentence and offsets, sentence language and confidence, near-duplicate group;
pandera checks on every rebuild. Annotated documents of target-bearing abstracts as `DocBin`
files per cell. Vectors stored with their keys inside each file and found by key, and encoding
resumed by the keys still missing, not by which files exist, so a changed usages table cannot mix
shards.

**4.7 Scope** (store decision 5). The target words, the control words (step 5), round A′'s
words, the shortlist and the long list (capped at 300 abstracts per cell). The long list came
from `scripts/discovery/` (a scan of the old grid; single-letter names; one input has no
producer). *Recommendation:* keep `data/discovery/longlist.json` as data, delete the discovery
scripts, and regenerate the long list from the rebuilt grid later.

**4.8 Downloads** (the data-acquisition draft). *Recommendation:* the three thin words
(*attention*, *platform*, *alignment*), sized per decade, floor 500 usages; the gap check runs
after the new pass, counting occurrences the way 4.1 matches them; likely under 200 OpenAlex
credits of 10,000 a day; nothing is downloaded without the author's go.

**4.9 Models and running time** (store decision 4). The last full pass (lemma votes, small model,
tagger and lemmatizer) read 1,274,029 abstracts, 206 million tokens, in 30 minutes on 10 CPU
processes; the new pass adds `senter`, lingua and the `SpanRuler`, so under an hour.
*Recommendation:* the small model for the whole pass; the transformer model only if a comparison
on one field shows it changes target sentences' tags, lemmas or boundaries enough to matter (on
lemmas it differed on 63 of 10,395 types); the GPU then for the transformer and the encoders.

Decisions (2026-11-12):

- 4.1: spaCy's `Matcher` on the lemma and the `PhraseMatcher` for phrase targets; every usage
  records its part of speech, morphology and any named entity it sits in. The target list gains a
  column stating each word's intended reading (*medium*: noun, not the adjective of *medium
  effect size*, 22% of its uses; *learning*: noun, 36% verb readings); the usages table keeps every
  match and flags whether it is in scope (the intended reading, not inside a name), measures use
  in-scope usages by default and may include the rest on purpose. The one-field comparison with
  the transformer model (4.9) looks at these words, where the small tagger errs.
- 4.2: hyphenated words stay whole (2.5), so a lemma search alone would miss every compound
  (*self-attention* has the lemma *self-attention*): 2–8% of a target's uses, and among them the
  senses studied (*self-attention*, *cross-attention*) or others (*attention-deficit*,
  *actor-network*, *e-learning*). A second `Matcher` rule finds the target as any part of a
  hyphenated word; each usage records the compound and the target's position (head or
  modifier); the encoder marks only the target part (*self-⟨t⟩attention⟨/t⟩*); compounds are in
  scope by default, and a measure may leave them out on purpose. Multi-word compounds (*neural
  network*) are marked the same way by the `PhraseMatcher` (3.5). Using compounds as evidence of
  a sense is step 7's.
- 4.3: one mention per abstract, drawn at random with a recorded seed (option B), replacing the
  first mention: in abstracts with several mentions (15–44% of them), the first lands in the
  opening sentence about twice as often as a random one (*network* 49% against 23%). The
  usages table lists every mention, so all mentions with each abstract weighted equally (option C)
  stays available without re-reading; it was not chosen now because within one abstract a word
  usually keeps its sense ("one sense per discourse", Gale, Church and Yarowsky 1992), so the
  extra mentions add little about a cell's sense mix, while every measure would have to handle
  same-abstract pairs (J and APD), split the classifier by abstract and resample abstracts in
  bootstraps. If C is switched on, that grouped handling is written once, in the library.
  *Amended by A11:* the draw is among in-scope mentions only, by the smallest hash of (seed, work
  id, content key), not a seeded shuffle.
- 4.4: the store records the usage and each encoder decides how to read it, so XL-LEXEME can be
  replaced later (the author's requirement). Stored, encoder-independent: the sentence text and
  the target's character offsets, the sentence's place in its abstract (the cleaned abstract is
  kept in the abstracts table, 4.6), the usage key (4.5). One encoder table declares each
  encoder's model and pinned revision, its context policy (the sentence by default; sentence
  plus neighbours or the whole abstract possible), its window and centring (XL-LEXEME: 128
  subword tokens) and how it marks or pools the target. Vectors are stored per encoder (name,
  revision, context policy), keyed by the usage key; swapping an encoder is one table entry and
  an encoding run. The 100-word pre-cut is dropped (the window is the encoder's setting);
  sentence length is recorded (sample of 2,462 target sentences: median 25 words, 1.1% under 8
  words, 2.1% beyond XL-LEXEME's window); malformed abstracts with no sentence punctuation (a
  "sentence" over about 150 words) are flagged in cleaning as a new keeping rule (step 2), their
  fate left to the sample audit.
- 4.5: two identifiers, each defined once in the library (they replace five copies of `key_of`
  and two other schemes). The **usage id** (work id, target word, mention number within the
  abstract) is the usages table's row identifier, stable unless a mention itself is added or removed
  (character positions would shift whenever cleaning changed earlier text); positions are
  ordinary columns. The **content key** (sha256 of the sentence text and the target's offsets in
  it, 20 hex characters) finds vectors: identical text is encoded once and shared, for every
  encoder, with no merge-time special case. pandera checks enforce both: usage ids unique,
  content keys equal to the hash recomputed from sentence and offsets, offsets pointing at the
  target.
  *Amended with 4.6:* the vector-cache key is the hash of exactly the text an encoder reads plus
  the target's position in it; for sentence-only encoders such as XL-LEXEME it equals the
  sentence key above, and for an encoder reading neighbours or the whole abstract it differs, so
  identical sentences in different abstracts never share a vector wrongly.
  *Amended by A11:* the usage id is (work id, word, content key), not the mention number, which
  shifted whenever a rule found an earlier mention.
- 4.6: the store is Parquet throughout, written and read with Polars. An **abstracts** table is
  the durable record of the cleaned corpus (work id, cells, year, cleaned text, sentence
  boundaries, language, flags and removal reasons). A **usages** table points into it (usage id,
  content key, word, compound and position, part of speech, morphology, lemma, in scope,
  sentence and target offsets, sentence length, sentence language and its confidence (lingua's
  per-sentence result where the abstract was checked per sentence, 2.3, otherwise the
  abstract's), near-duplicate group) instead of copying text.
  **Vectors** are Parquet tables per encoder and word (`vectors/encoder=…/word=…/`, read by
  Polars' hive partitioning), one row per cache key with the vector as an array of 16-bit floats
  (tested: 1,884 bytes per vector, a filtered read of 100 keys in 0.07 s); encoding resumes by an
  anti-join of usages against existing keys; no numpy shards, sidecar keys or index file.
  spaCy's `DocBin` is only a rebuildable cache of the annotations: it is tied to spaCy's version,
  and target-bearing abstracts are nearly all of them (57% contain one of the 39 target words
  alone; about 2–2.5 GB for the corpus). Unchanged from the store draft: pandera checks on every
  rebuild, the datasheet, versioning (see step 9). The static pipeline's normalised text stays as
  plain text files. Matches common practice: DWUG keeps a usages table per word with the
  context, target offsets and sentence offsets inside the context; Parquet partitioned by column
  is the standard analytical layout; a vector database is not needed, since measures compare
  usages within one word.
- 4.7: one **word list** file for the whole project (word, intended reading from 4.1, role:
  target, control or candidate, pipelines, cap), read by the store, the static pipeline, the
  downloads and the checks; it replaces `words_of_interest.txt`, the control-word files as
  sources of fixed controls, the list typed into `build_vector_store.py` and the scope part of
  `longlist.json` (the static and contextual target lists barely overlap today: 39 against 11).
  Caps by role, not one cap: targets and their controls share one, so they stay comparable;
  candidates keep a lower one (300 now), since one cap of 1,000 would mean up to 12 million
  long-list usages against 1.9 million in the whole store today, where the control word *paper*
  alone has 125,784. Controls are encoded only in the cells of their targets' planned
  comparisons. The static outputs record the word-list version (the phrase rule of 3.5 makes the
  normalised text depend on it; step 9). Same-abstract controls (detection experiment 2) are
  drawn from the abstracts table when needed, not listed. `longlist.json` is kept as data and
  `scripts/discovery/` deleted, the long list regenerated from the rebuilt grid. *component*
  and *having* are stable checks (check 2b's author-judged words, 7.1); the six-word shortlist
  gets a role after checking its history. Cap values are set
  after steps 5 and 6. *Amended by A12 and A5:* the word list becomes three tables (words,
  comparisons, comparison controls; no "pipelines" column), words are (lemma, part of speech)
  pairs, and caps start as provisional values in the parameter files.
- 4.8: downloads are a curated decision, not a pipeline step. The pipeline reports gaps (word,
  field, decade, usable usages, OpenAlex's total of abstracts containing the word, estimated
  cost); the author approves a committed **download plan** (word, cells, reason); the download
  script carries out only the plan, after the author's go, and records what it fetched (counts,
  totals, seed). Order: the plan is drafted from existing counts once steps 5 and 6 fix the words
  and comparisons; downloads run before the full spaCy pass, so one pass cleans the regular
  samples and the downloads together; after the pass, the gap report counts usable usages in
  final terms (4.1, 4.2) and any cell still under the floor gets a small curated top-up.
  Improvements: OpenAlex's total of matching abstracts is recorded per cell (the share sampled);
  when that total is small, the whole set is downloaded, not a sample; fetching continues until
  the floor of usable usages is met, since OpenAlex's stemmed search returns related forms. Kept
  rules: one source per word (a word's cells all regular or all downloaded); downloaded cells
  never feed the static pipeline (the abstracts table marks each abstract's source). The floor
  (500) is provisional until step 6. First candidates: *attention* (AI), *platform*
  (communication), *alignment* (AI).
- 4.9: the small model (`en_core_web_sm`) for the whole corpus: cleaning, sentences, the static
  normalised text (about 30 minutes on 10 CPU processes for the last full pass). The transformer
  model (`en_core_web_trf`) is a candidate for target sentences only, re-annotating tags,
  morphology and entities (and dependencies if step 7 needs them) within the small model's
  sentences, so boundaries stay consistent. spaCy's own scores (general English, not abstracts):
  tags 97.3% against 98.1%, entities 0.843 against 0.899, labelled dependencies 0.899 against
  0.938, sentence boundaries 0.907 against 0.886 (the transformer is not better there). Decided
  by a one-field comparison: if more than a couple of percent of usages change their in-scope
  flag or lemma between the models, target sentences use the transformer, on bazzite's GPU after
  its repository is synced.

Step 4 decided (2026-11-12).

## Step 4b: corpus quality

How we know the corpus is good: four layers, the last two curated. The written record follows
the "Datasheets for Datasets" template (Gebru et al. 2018).

| Layer | What it does | How |
|---|---|---|
| 1. Definition | What a cell is (OpenAlex field label, language, document types, decade, the random sample and its seed), with known weaknesses: field labels come from OpenAlex's classifier, nested fields overlap, HCI's label widened from 915 venues to 3,601 | Written once, reviewed |
| 2. Automatic checks | pandera on every rebuild (4.6); counts per removal reason per cell (step 2) | Automatic |
| 3. Human review of samples | The author reads about 20 random abstracts per cell and records judgements (a real abstract? the right field? clean? did cleaning remove the right things?), plus samples of what each cleaning rule dropped | Curated |
| 4. Sign-off | No rebuilt corpus is used for experiments until the datasheet and the review are read and approved, with the decision recorded | Curated |

Decision (2026-11-12): layer 3 is sized to about an hour, as automatic as possible without
losing quality, built on established audit methods: sampled audits of corpora with a fixed set
of categories (Kreutzer et al. 2022, "Quality at a Glance"), auditing what filters removed as
well as what they kept (Dodge et al. 2021, on C4), stratified random samples, the rule of three
for zero-error samples, Wilson intervals, a codebook written before reading, blind repeats for a
single reviewer, and Data Statements for NLP (Bender and Friedman 2018) beside the datasheet
(these sources to be added to `related_work.md`).

- **Automatic first:** language (lingua confidence), duplicates (MinHash), boilerplate (the
  `SpanRuler` and template counts), malformed or cut-off abstracts (length, missing sentence
  punctuation), and likely wrong-field abstracts (a scikit-learn field classifier trained on the
  corpus flags abstracts it confidently assigns elsewhere; OpenAlex's field score fetched for
  those if useful).
- **Codebook, six categories:** valid abstract; wrong field (defined as clearly another
  discipline, not merely interdisciplinary, since the fields overlap by design); not English or
  mixed; boilerplate or junk; duplicate; truncated or malformed.
- **First rebuild, about 350 judgements (about an hour, in sittings of about 15 minutes):** 130
  random kept abstracts, about 12 per field; about 10 dropped items per removal rule (about
  120); 80 automatically flagged abstracts (finding problems fast and testing whether the flags
  can be trusted); 20 blind repeats. Later rebuilds drop to about 250 (no flagged items, no
  repeats) or lower if the flags proved reliable.
- **Thresholds set before reading:** junk or non-abstracts under 2% overall; wrong field under 5%
  in every field; any removal rule with 2 or more good texts among its 10 dropped items gets a
  closer look. Results are reported automatically (proportions with Wilson intervals) into the
  datasheet; the author reads one summary.
- **Judging page:** field, decade and the first 60 words of each abstract, the six categories on
  keys 1–6, progress saved; adapted from `scripts/review/judge_template.html`.
- **When:** the first rebuild, and again only when cleaning rules or sources change; otherwise
  layers 1 and 2 only.

## Step 5: control words

Inputs: the detection draft's experiments 2 and 3; `pipeline_map.md` stage 11. Four control
files from four scripts today, parsed three ways. Decide one selection method per purpose:
matched on frequency and part of speech, or drawn from the same abstracts (the detection draft
argues for the latter).

Decision (2026-11-12), after a critical pass and a check of established practice:

- The four control-word files are retired as sources and kept as data (past rounds used them).
  They contradict each other (*control* is a target and a rung-3 control) and do not match the
  targets; round A found 4 of 4 changes against shuffles, 2 of 4 against them.
- **Shuffles on for every measure** (scipy's `permutation_test`): the established control
  condition (Dubossarsky et al. 2017, "Outta Control"; Dubossarsky et al. 2019, "Time-Out"). They
  are the floor, not the bar: a word whose contexts moved only with its field's topics passes
  them, as round X's quiet transition did.
- **Matched control words, generated, at least 30 per target** (or pooled by frequency band), so
  a percentile bar is defined: same part of speech, comparable frequency trajectory between the
  two periods (SemEval-2020's criterion: "same POS and comparable frequency development"),
  similar within-period spread (frequency and polysemy both create apparent change: Dubossarsky
  et al. 2017; Keidar et al. 2022), excluding targets, sense-list words and candidates; stored in
  the word list with the role "control" and their target. *Amended by A12:* stored per
  comparison in `curated/comparison_controls.tsv`, matched on frequency in each period, the
  tolerance the smallest that yields 30 candidates.
- **Controls validated against each other, not against their shuffles** (topic drift moves every
  word, so that test would reject them all): outliers among the controls are read briefly by
  hand, a light version of SemEval's manual verification, then replaced. *Amended by A2:* never
  replaced on their change score; only for score-blind reasons, before any target is scored.
- **False discovery rate correction** (`scipy.stats.false_discovery_control`, Benjamini–Hochberg)
  whenever many words are screened at once (Liu, Medlar and Glowacka 2021), noting their caveats:
  permutation tests are costly, and the correction assumes independent tests. *Amended by A10:*
  100,000 shuffles per word, Benjamini–Yekutieli reported beside Benjamini–Hochberg.
- Same-abstract controls are drawn at experiment time from the abstracts table; the static grid
  uses its frequency bands; an automatic check forbids a word being both target and control.
- Which bar decides a change (shuffles, matched or same-abstract controls) is step 6's.
- To add to `related_work.md`: Schlechtweg et al. 2020 (SemEval-2020 Task 1), Dubossarsky et al.
  2019 (Time-Out), Liu, Medlar and Glowacka 2021, Keidar et al. 2022.

## Step 6: detection experiments

Inputs: `docs/design_detection_experiments.md`, steps 2–5, and established practice (checked
2026-11-12). Detection asks whether a word's use changed between periods, and how much.

**Revised list and order (the author, 2026-11-12: established methods first, exploratory ones
after):**

| Order | Experiment | Standing | Needs |
|---|---|---|---|
| 0a | Calibration on the datasets the LSCD Benchmark covers (Schlechtweg, Virk and Arefyev 2024), not only DWUG English: encoder and measures against human graded change, comparable with published numbers. The benchmark's code is not a dependency (Hydra scripts pinned to Python 3.10, no licence stated); its datasets and standard metrics are used with scipy and scikit-learn | Standard practice | Encoders, the published datasets |
| 0b | A systematic synthetic set as the development set: several planted scenarios (gradual or sudden, a sense gained or lost, a frequency change without a change of meaning) (Shoemark et al. 2019) | Established | The rebuilt store |
| 1 | J on every usage: bootstrap intervals (resampling abstracts), permutation significance, false discovery rate correction when many words are screened (Liu, Medlar and Glowacka 2021) | Established | The rebuilt store; the matched controls (step 5), since 6.3's bar is their 95th percentile |
| 2 | Matched controls (step 5) as the bar | Established (SemEval-2020 style) | The word list |
| 3 | Year-by-year time series of a word's change against a reference period, with change-point detection and a shuffle null, replacing half-decade windows (Kulkarni et al. 2015; Shoemark et al. 2019: the full series beats two points) | Established | Years per usage (4.6); `ruptures` (1.1.10, September 2025, maintained), added when the experiment starts |
| 4 | One measure for the static grid instead of the two-threshold rule | The project's own diagnosis | The rebuilt grid |
| 5 | Same-abstract controls, aimed at topic change | Exploratory: the literature has no method that separates topic shift from semantic change (Montanelli and Periti's survey lists it as open) | The abstracts table |
| 6 | Removing the era signal from the vectors (concept erasure, LEACE) | Exploratory | Last, since it alters every vector; `concept-erasure` (EleutherAI, last release January 2024) or LEACE's closed form written with numpy and scipy, decided then |

Checks every experiment reads: the documented changes (*alignment* and *attention* in AI,
2010s→2020s; *network* in AI, 1980s→1990s; *platform* in communication, 2000s→2010s), the quiet
transitions, planted changes, the placebo, shuffles and controls. Exploratory experiments are
judged hard against planted changes and the placebo, since no published result supports them.

**6.1 How results are judged.** Rules written before each run, or open reading on development
cases and confirmation on held-out ones. *Recommendation:* the second, the standard split of
exploratory and confirmatory analysis, with confirmation rules written before the confirming
run (guarding against the "garden of forking paths", Gelman and Loken).

**6.2 Held-out cases.** All known cases were used in round X, and four are too few to split.
*Recommendation:* develop on the synthetic set (0b) and the benchmark (0a); build a fresh set of
known answers for confirmation (changes dated by dictionaries in the project's fields, such as
Wiktionary's dated and field-labelled senses, and cases such as *hallucination* in psychology and
*prompt* in AI), and hold it back. The author judges which cases count as documented.

**6.3 The bar in experiment 1.** *Recommendation:* the 95% interval must clear the matched
controls' 95th percentile (at least 30 controls, step 5), with shuffles as the floor and false
discovery rate correction when many words are screened.

Decisions (2026-11-12): the revised list and order above; and

- 6.1: explore freely on the synthetic set (0b) and the benchmark (0a); confirm once on held-out
  cases under a confirmation plan committed to git before the confirming run (the frozen
  method, the cases, the pass rule; the commit's timestamp shows it came first). Development
  material is not the project's domain (synthetic changes; historical general English), so the
  held-out set from the project's fields is the test that counts.
- 6.2: a fresh held-out set of about 10–20 documented changes and as many stable cases (chosen
  like matched controls), drafted by Claude with evidence (sense, field, dating source, corpus
  counts per decade; sources: Wiktionary through the kaikki.org extracts, well-documented
  recent cases, the corpus), reviewed by the author (reading a few usages per decade is
  annotation, not measuring), committed and marked not for use until confirmation, with its
  hash recorded. Its pass rule (for example, at least 8 of 10 changes found, at most 1 of 10
  stable words flagged) is set before any measuring.
- 6.3: the bar's form is fixed now (the target's 95% interval against its matched controls'
  distribution, shuffles as the floor, false discovery rate correction when screening); its
  level is chosen during development, where the answers are known, and frozen in the
  confirmation plan. Tools: scikit-learn `cosine_similarity` on the store's vectors, scipy's
  `bootstrap` (resampling abstracts, never pairing a usage with its own copy, which would bias J
  upwards), `permutation_test` with at least 1,000 shuffles (the similarity matrix computed once,
  labels reshuffled), `false_discovery_control`. *Amended by A10:* 100,000 shuffles, the same n
  per period for the target and its controls.

## Step 6b: the static models (after detection experiment 4)

Whether the static pipeline keeps skip-gram vectors (TWEC, or cells trained apart with gensim and
aligned with scipy's `orthogonal_procrustes`) or runs on PPMI-SVD alone (the project's own code,
deterministic, no gensim). Skip-gram with alignment is probably the stronger static method for
detecting change (partly checked in A13; skip-gram against PPMI-SVD still to be checked; not yet
in `related_work.md`), and the README's best finding rests on
both agreeing. Experiment 4 tests whether the grid's misses came from its rule, its windows or
the embeddings; if the grid does not earn its place, the question lapses and gensim goes with it.

Decision:

## Step 7a: finding senses

Not settled (the author, 2026-11-12): hand-written lists are the best so far on one benchmark;
each automatic method failed in one configuration, not as a method.

| Method | What was tried | Result | Untried |
|---|---|---|---|
| Hand-written lists | Rounds Y, Q (DWUG), corpus lists | 0.69–0.80 | — |
| Substitutes and neighbours | Round Q: ModernBERT "or even [MASK]", Louvain communities | 0.39–0.59 | Other patterns, encoders, graph methods |
| Dictionaries | Round Z (WordNet); Wiktionary as a gap check | about 0.18 below hand-written; two gaps in four words | Wiktionary lists as such |
| Clustering usage vectors | Rounds C, C′, C″: SciBERT, KMeans, k by silhouette | failed calibration | XL-LEXEME vectors from the store, other algorithms (agglomerative, affinity propagation, graph clustering), clustering only the usages no listed sense fits (gaps) |

What failed in rounds C–C″ was the rule turning clusters into a sense's share on unseen data;
the encoder did place every known sense (`docs/related_work.md`). Clustering may therefore serve
to find senses (build or check a list) even if counting is done another way.

Clustering approaches to consider (the author, 2026-11-12: try different approaches):

| Approach | Source |
|---|---|
| XL-LEXEME vectors from the store (agreement with human clusters 0.27 against BERT's 0.14 and annotators' 0.63) | Periti and Tahmasebi 2024, in `related_work.md` |
| Substitutes grouped by community detection at scale, field by field | Eyal et al. 2022; Lucy et al. 2023 ("not done" in `related_work.md`) |
| Correlation clustering over pairwise XL-LEXEME similarities, mirroring how DWUG's human clusters were built from pair judgements | Claude's suggestion; to be read before it is relied on |
| Algorithms without a fixed number of senses: agglomerative with a distance threshold, affinity propagation, HDBSCAN (which leaves outliers unassigned) | Claude's suggestion |
| Hybrid: hand-written senses as seeds, clustering only the usages no listed sense fits | Claude's suggestion |
| Clustering generated definitions (a comparison arm, since it needs a generative model) | Giulianelli et al. 2023, in `related_work.md` |

Benchmarks: DWUG's human clusters (round Y's bar 0.273; annotators 0.63) and the author's labels
on the corpus.

Decide: which candidates are tried, on which benchmark, and against the hand-written lists.
Settles the step 1b code marked "decide with step 7a".

Decision (2026-11-12), after checking the literature and the tools:

- Literature: under equal conditions APD predicts graded change best, while sense-based
  approaches correlate less but explain more, and XL-LEXEME is the best model for sense induction
  too (Periti and Tahmasebi 2024); DWUG's human clusters come from correlation clustering of a
  usage-pair graph (Schlechtweg et al. 2021), and filling the missing pairs with a model before
  clustering improves agreement with human senses (LChange 2024); K-Means and affinity
  propagation are the most used algorithms, with agglomerative clustering and HDBSCAN also in use
  (Montanelli and Periti's survey); sense induction is still open. The hand-written lists'
  0.69–0.80 (rank correlation with human graded change) and clustering's 0.27 (agreement with
  human clusters) are different measures and are not compared directly.
- 7a.1: automatic methods serve first as an aid to the hand-written lists (proposing senses,
  finding usages no listed sense fits, for the author's review), not as a replacement.
- 7a.2, established first: (1) XL-LEXEME pair scores with correlation clustering, as DWUG built
  its clusters; (2) XL-LEXEME vectors with affinity propagation, agglomerative clustering
  (distance threshold) or HDBSCAN; (3) substitutes grouped by community detection at scale (Eyal
  et al. 2022; Lucy et al. 2023) with **Leiden** (Traag et al. 2019; Louvain's successor, which
  guarantees well-connected groups; in networkx 3.6 already), label propagation as a quick
  check; then exploratory (4) hand-written senses as seeds, clustering only the usages no listed
  sense fits; (5) generated definitions as a comparison arm only.
- 7a.3: scored by agreement with DWUG's human clusters (adjusted Rand index) on the openly
  published DWUG datasets (English, German, Swedish and others), by the same measure as the
  hand-written lists, and on the corpus by the author's labels (step 7).
- Tools: scikit-learn 1.8 (`AffinityPropagation`, `AgglomerativeClustering`, `HDBSCAN`) and
  networkx 3.6 (`leiden_communities`, `label_propagation_communities`), both installed;
  `correlation-clustering` 2.0.0 (April 2026, by the DWUG authors, depending only on networkx,
  numpy and scipy) is the one new dependency, adopted once its licence is confirmed (PyPI states
  none). DWUG's `WUGs` scripts (no package, conflicting licences) and `chinese-whispers` (last
  release July 2024) are references only. If no licence is confirmed: ask its authors (an issue
  on their repository, after the author's go); until then (1) waits and (2)–(3) go ahead; no
  re-implementation.

## Step 7: counting senses

- The sense-counting draft's decisions: which words; the author's labels (about 30 per word,
  across decades); fine-tuning the definition reader or not.
- Labellers: one Claude path (the Batch API) plus JEV; the CLI `Labeller` deleted. Language
  models stay a comparison arm.
- One classifier, one divergence function and one sense-list schema (two today) in the library.
- Cue words move to the spaCy `Matcher`.
- Settles the step 1b scripts marked "decide with step 7".

Decisions (2026-11-12):

- 7.1: the first set is eight words with reviewed lists: documented changes *attention*,
  *network*, *platform*, *alignment* (`corpus_v2.toml` and its working draft); arriving senses
  *prompt* (the anachronism test) and *hallucination* (`round_a_prime.toml`); stable checks
  *component* and *having*, the check 2b words whose pairs the author judged, with no documented
  change. Second wave: *transformer*, *embedding*, *cloud*, and the held-out set of step 6, where
  counting is confirmed (the documented changes are development material here too). Fixes from
  checking the lists against steps 2–4: cues written for split hyphens (*attention*'s `<<self`,
  `<<head`) stop working when hyphenated words stay whole, so compounds recorded in 4.2 become
  cues when cues move to spaCy's `Matcher`; counting uses every reading a list covers (*prompt*'s
  adjective and verb senses), while the intended-reading flag of 4.1 serves detection; the word
  list gains a match-by-form option, since *having* is a form, not a lemma (its reading: verb or
  auxiliary); *network* gains its own actor-network sense (Latour, Callon, Law; 214 uses),
  identified by the compound and kept separable, added to `corpus_v4_draft.toml` with the social
  sense excluding its cues (the frozen version 2 is unchanged).
- 7.2: the author's labels are the calibration set (each counter's error rates for the
  correction of 7.4) and its test; they must be human, blind, random and spread across periods,
  since the earlier correction failed on error rates measured from easy cue-labelled usages.
  About an hour, in sittings of about 15 minutes: refine check 1a (the author's 280 blind,
  random, cross-period yes/no judgements on *attention*, *network*, *platform*, *alignment*,
  *prompt*) by answering "which sense?" for its 183 "no" and "unclear" items (the 97 "yes" items
  are the changing sense already), about 45 minutes; new blind labels for *hallucination*
  (psychology and AI, both periods, about 30) and the stable checks *component* and *having*
  (about 30), drawn at random from the rebuilt store; each list's senses plus "other or
  unclear", on the keystroke judging page of step 4b. Claude's 300 development labels are not
  ground truth; verifying them (about 30 minutes, with about 30 items labelled blind first to
  measure anchoring) is held in reserve for words whose results come out uncertain, and doubles
  as a measure of Claude's labelling. When: after the store rebuild and the consolidation of the
  sense lists (7.5), before the counters are evaluated.
- 7.3: the counters, in order: (1) the cue classifier, the current baseline; (1b) a supervised
  classifier (scikit-learn logistic regression on XL-LEXEME vectors, strongly regularised)
  trained on the author's random cross-period labels, the standard baseline, which removes the
  cue classifier's era bias at its source; (2) nearest sense by examples, in the manner of LMMS
  (Loureiro and Jorge 2019; LMMS Reloaded 2022): sense vectors averaged from XL-LEXEME vectors of
  labelled usages and each list's example, which suits XL-LEXEME's training on usage pairs; (3)
  nearest definition, zero-shot, expected weaker since XL-LEXEME was not trained on definitions;
  (4) cues as spaCy `Matcher` patterns with recorded compounds as cues, alone or cue-first before
  (2); (5) a bi-encoder trained on definitions (Blevins and Zettlemoyer 2020) only if the others
  fall short, trained by the project since the original code is archived (January 2024, PyTorch
  1.2, non-commercial licence); Sonnet and JEV as the comparison arm. Every counter: outputs
  probabilities (for probabilistic adjusted counting, 7.4) and can answer "no listed sense fits"
  (a cut-off set on the author's "other or unclear" items; those usages feed step 7a's gap
  finding); is evaluated out of fold, so labels never build and grade the same counter; is
  scored per period, with sense vectors built from examples balanced across periods; and must
  pass two hard tests, anachronism (no language-model *prompt* before about 2020, no AI
  *hallucination* before the late 2010s) and stability (no invented change in *component* and
  *having*).
- 7.4: proportions by quantification with **QuaPy** (0.2.3, October 2026, BSD; it brings pandas
  back as an installed dependency, which the author accepted; the project's code stays on
  Polars), through a small adapter that passes a counter's probabilities through as a stand-in
  classifier (tested in a throwaway environment with the project's versions). Candidates: PACC,
  ACC and EM, the method chosen per word by QuaPy's artificial-prevalence protocol run on the
  author's labels, since no method wins everywhere (Schumacher et al., JMLR 2025; ACC, PACC and EM
  stand out in Moreo and Sebastiani's re-evaluation). EM only with bias-corrected temperature
  scaling (BCTS), done by QuaPy: uncalibrated EM collapsed in tests and is known to fail with
  several classes or uncalibrated probabilities (Esuli, Molinari and Sebastiani 2020), while EM
  with BCTS is hard to beat (Alexandari, Kundaje and Shrikumar 2020); plain temperature scaling
  still failed. Plain classify-and-count is always reported alongside (near state of the art
  when tuned: Esuli, Moreo and Sebastiani). Bootstrap intervals resample both the cell's usages
  and the author's labels, since error rates estimated from 50–70 labels per word are noisy and
  the corrections amplify that (a synthetic test showed adjusted counts overshooting); where a
  key word's interval is too wide, more labels for that word only. The methods assume a sense
  looks alike in every period; where a word's senses change how they are expressed (computing
  *attention*), error rates are estimated per period (guarded by 7.3's per-period scoring).
  Reference for intervals: *Learning to Quantify* (Esuli, Fabris, Moreo and Sebastiani 2023, open
  access). To add to `related_work.md`: these sources.
- 7.5: one Claude labelling path (the Message Batches API `BatchLabeller`; the command-line
  `Labeller` is deleted); one divergence function (scipy's `jensenshannon`, which returns the
  Jensen-Shannon distance, the square root of the divergence; squared where the divergence is
  meant, and every report names which); one sense-list
  format (the corpus format: keys, definitions, examples, cues, exclusions), into which the
  final lists are written before the author labels (7.2). The step 1b scripts left for step 7
  are superseded and go once their results are in `docs/experiments.md`:
  `score_labels_classifier`, `score_labels_corpus`, `score_labels_head_to_head`,
  `score_labels_known`, `finetune_classifier` and `dev_harness` (its last experiment, decade-
  balanced weights and temperature scaling, committed first as a record); `scripts/review/` goes
  once the keystroke judging page (4b, 7.2) exists, adapted from its template.
- 7.6: the counters are used as they are first; fine-tuning (XL-LEXEME's top layers, or a
  trained bi-encoder) only if they fall short on the author's labels, since labels are few and
  fine-tuning risks learning the era again.

Step 7 decided (2026-11-12).

## Step 8: code layout (follows from steps 1–7)

The code review's critical items: a `usages/` package (which became `corpus/target_matching.py`,
`corpus/usage_selection.py` and the catalog's usages dataset); `corpus/cleaning.py` and an annotation
module holding the spaCy pass; `lemmas.py` split (spelling, map); a `cli/` package. Plus one
neighbour function (from the measured pool, if the explorer stays) and the repeated helpers
extracted (unit rows, divergence, torch device, cell reading, label tables).

Replacement audit (the rule above, applied to existing code; extend as code is touched):

| Hand-rolled today | Library replacement |
|---|---|
| Unit-normalised rows, written seven times (`grid._unit_rows` and others) | `sklearn.preprocessing.normalize` (handles zero rows) |
| `metrics.cosine_distance` | `scipy.spatial.distance.cosine` |
| Sentence regexes (`quality._SENTENCE`, `senses._SENTENCE_END`) | spaCy `senter` with special cases (step 2) |
| Six regex tokenisers | spaCy tokens (step 3) |
| Mojibake table (`classifier.MOJIBAKE`) | affected abstracts dropped, found by a signature list in the patterns file (step 2.2) |
| URL and DOI regexes | spaCy `like_url` and `url_match` (step 2.2) |
| Language checks (`looks_english`, `english_share`, `foreign_share`) | lingua (step 2.3) |
| Duplicate rules | datasketch MinHash (step 2.4) |
| Judgements parsed from markdown with a regex (`gold_pairs.py`) | a TSV file read with Polars |
| Bootstrap intervals (detection experiment 1, to be written) | `scipy.stats.bootstrap` |
| Already library-backed, kept | Jensen-Shannon distance (`scipy.spatial.distance.jensenshannon`), NNLS (`scipy.optimize.nnls`), community detection (`networkx`; Leiden replaces Louvain, 7a.2), KMeans and silhouette (scikit-learn) |

Decision (2026-11-12): one job per package, so code is easy to find and to change; pipelines
compose the jobs; only the catalog reads and writes data (A8).

Layers (one import-linter contract): `catalog/` at the bottom; the job packages side by side,
none importing another (`corpus/`, `embeddings/`, `detection/`, `senses/`, `evaluation/`,
`review/`); `pipelines/` composing them; `cli/` on top, thin. `experiments/` sits outside the
package; `common/` at the bottom only for helpers needed in three or more places.

```
src/interpretant/
├── catalog/      abstracts.py, usages.py, vectors.py, words.py, schemas.py, provenance.py
│                 (one typed load and save per dataset; paths from the catalog file, A8)
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

- Placements: usage selection is a research decision, in `corpus/`, not the catalog; the OpenAlex
  client is `corpus/openalex.py` (corpus compilation includes collecting the texts; a second
  source would become a module beside it), using `pyalex` (0.21, February 2026, MIT, depending
  only on requests and urllib3) in place of the hand-written HTTP client, with the catalog saving
  what it fetches; `evaluation/` holds everything DWUG-specific (the catalog only reads files
  generically and records their versions), planted changes, the held-out known answers and the
  metrics, since detection and senses cannot import each other; the shifted-shares protocol
  stays in `senses/quantification.py`; `review/` replaces `scripts/review/`; model-calling code
  goes by job (substitutes and generated definitions in `senses/induction.py`, the Claude and
  JEV labellers in `senses/llm_labellers.py`, encoders in `embeddings/`), shared GPU and
  batching helpers in `common/`; settings objects belong to `pipelines/`, job functions take
  plain arguments; "static" and "contextual" are flows across packages and CLI command groups,
  not packages.
- Design rules: the catalog hands out streams (iterators) and jobs consume and produce them, so
  memory stays bounded; the catalog is split by dataset; each pipeline reads as a table of contents
  (one call per stage), with a one-page `docs/architecture.md`; interfaces only at the four swap
  points (the encoder table, the counters, the cleaning patterns file, the word tables and
  settings files).
- Names: job modules are nouns, pipelines verb-plus-object phrases (`encode_usages`,
  `detect_change`); the field's standard terms
  (`significance`, after the "statistically significant detection" papers; `inventory`;
  `disambiguation`; `term_extraction`; `provenance`); British spelling, no abbreviations;
  experiments named by their question, not by round letters.
- Data folders: generated data follows the Cookiecutter Data Science convention
  (`data/external/` for DWUG and other datasets, `data/raw/` for downloads, `data/interim/` for
  the annotation cache, `data/processed/` for tables and vectors, `models/` for trained static
  models); hand-curated inputs live in a committed `curated/` folder, never overwritten by a run
  (the word tables of A12, sense inventories, cleaning patterns, lemma exceptions, the spelling table,
  download plans, the held-out known answers).
- `scripts/` disappears once its contents are rebuilt; the replacement audit below applies as
  code moves; the layout is not a separate refactor but the shape the rebuild builds into, phase
  by phase, a commit per phase.
- The composing package is `pipelines/` (the author, 2026-11-12), the most widely recognised name
  (Kedro, ZenML, scikit-learn), chosen over `workflows/` (collides with `.github/workflows/`) and
  `stages/` (DVC's term); each module is one stage, the static and contextual flows are sequences
  of stages, and "the spaCy pipeline" is always named so inside `corpus/`.
- `curated/` changes only through review (the author, 2026-11-12): runs write proposals into
  `data/` (generated control words from step 5, lemma exceptions from spaCy's errors), and a file
  enters `curated/` only after the author approves it, as download plans do; the spelling table
  is committed once after checking and then left alone.
- Everything a person judged lives in `curated/` and is committed (check 1a's answers, the gold
  pairs, the head-to-head choices, the coming labels and reviews): human judgements are the one
  thing that cannot be regenerated. Raw model outputs stay out, as now. The untracked round E and
  round Q label files are sorted under housekeeping.
- The `web/` explorer (a Next.js app, 118 tracked files, reading `web/data/`) is deleted with the
  step 1a deletions, after `docs/experiments.md` records what it showed: good data comes before
  presenting it. In the meantime (the author, 2026-11-12): each experiment runner writes a
  Markdown report, `reports/<experiment>/report.md`, with its key tables (from Polars, as
  Markdown) and matplotlib figures (`reports/<experiment>/figures/`) and the data and settings
  that produced it (step 9); `reports/` follows the Cookiecutter Data Science convention. For
  interactive exploration during development, the most common setup in data projects (chosen over
  marimo and plain-Python cell files): plain Jupyter notebooks in a committed `notebooks/` folder,
  named by order and topic in the Cookiecutter Data Science manner (for example
  `1.0-gd-attention-usages.ipynb`), with `nbstripout` as a pre-commit hook so no outputs or data
  reach git; `ipykernel` and `nbstripout` as development-only dependencies. Notebooks are scratch:
  code needed twice moves into the package, and anything that matters moves into an experiment
  runner, so no result comes from a notebook.
  A presentation (an explorer, a dashboard, shared pages) is designed again once there are new
  results.
- `docs/` is reorganised into `docs/reference/` (living documentation such as `spacy.md`,
  `methods.md`, `related_work.md`, `architecture.md`, `datasheet.md`), `docs/records/` (the
  predictions files, diagnostics, `experiments.md`) and `docs/design/` (drafts and this agenda),
  after a pass over the 23 documents deciding what to merge, what is no longer relevant, and what
  moves where (housekeeping; it touches every link, so it pairs with step 10).

## Step 9: reproducible runs

- A pipeline runner (Makefile targets or a pipeline command) that records flags and the content
  hash of every input (content hashes were agreed in the stack decision).
- Outputs record the settings that made them, so a rerun with other settings cannot silently
  reuse a stale file.

- Data versioning: DVC, the common tool in ML projects for tracking data files by content hash
  and the pipeline stages that produce them, against a small manifest of the project's own
  (file hashes and the settings that made each file). Weighed under the reasonable-dependency
  rule: DVC brings its own workflow and remote storage.

- Outputs that depend on the word list (the static normalised text, through 3.5's phrase rule)
  record its version.

Decision (2026-11-12): **DVC** for pipelines and provenance, after comparing it with Snakemake
(runs steps, no data versioning or lock file), Kedro (a full framework to restructure around),
Hydra (configuration only; pydantic settings already exist), MLflow (experiment tracking at a
scale the Markdown reports cover) and a manifest of the project's own (a re-implementation).

- `dvc.yaml` lists the stages, one per module in `pipelines/`: each stage's command is the CLI
  (`interpretant build-corpus`, …), its dependencies the data, the `curated/` files and the code,
  its outputs what it writes.
- Settings live in parameter files DVC tracks (`params.yaml` or TOML), loaded by the pipelines'
  pydantic settings objects, so a changed seed or cap reruns exactly the stages that use it.
- `dvc repro` replaces every "skip if the file exists" check (downloads, the compass, cells,
  shards, caches).
- `dvc.lock`, committed, is the provenance record: hashes of each output's inputs, settings and
  code; reports cite the commit and the lock entry they came from; it records the word-list
  version, since the word tables in `curated/` (A12) are dependencies of the stages that read
  them (replacing `catalog/provenance.py`'s planned role, which shrinks to reading the lock for
  reports).
- Raw downloads are versioned with DVC: registered with `dvc add`, never produced by a stage (A7).
- Large data: the cache links outputs with reflinks (copy-on-write; APFS on the Mac and btrfs on
  bazzite support them), so it does not double disk use; huge regenerable outputs may be
  `cache: false`.
- No remote at first; an SSH remote (`dvc-ssh`) to move data between the Mac and bazzite is added
  when the rebuild needs the GPU.
- Caveats noted: stewardship moved from Iterative to lakeFS, releases have slowed (last March
  2026); if DVC were abandoned, the pipelines still run as CLI commands and only the rerun logic
  and lock file would need replacing.

## Step 10: terminology pass (later)

The author, 2026-11-12: names should follow common NLP and data-project usage, as *static* and
*contextual* replaced "type track" and "usage track". One pass over code, docs and data paths,
after the rebuild's names settle, listing each project-made term, the standard one, and where it
is used. Candidates noted so far (to be checked against the literature, not assumed):

| Project term | Possible standard term |
|---|---|
| cell (a field in a decade) | corpus slice, time bin, subcorpus |
| grid (both the corpus of cells and the TWEC models) | two names, one per meaning |
| layer (`openalex`, `openalex_full`) | source, subcorpus |
| known answers; quiet transitions | gold-standard changes; stable (negative) cases |
| planted changes | simulated or synthetic semantic change |
| sense shares | sense distribution |
| J (across minus within) | to be checked: a named measure in the literature, or ours |
| word stream (the static pipeline's text) | normalised token stream, normalised text |
| rounds, rungs | experiments; calibration stages |
| usages-table columns | DWUG's names where they mean the same (*context*, target offsets, *grouping* for periods) |

Decision:

## Amendments after review (2026-11-12)

A retroactive review of steps 0–10 (Claude's own pass and an independent agent's cold read,
merged and checked) found contradictions, ordering problems and methodological gaps. Each
amendment below was agreed with the author and overrides the step it names.

- **A1 (steps 6.2, 7.1): held-out words never leak into development.** The word list gains a
  `held_out` role; every development output (screening, reports, gap lists, the static grid's
  ranked lists) leaves held-out words out until the confirmation plan is committed. *prompt* and
  *hallucination* are struck from the held-out candidates (they are development words in 7.1);
  the held-out set is drawn from words used nowhere in development. Labels for held-out words
  are made only after the confirmation plan is frozen, and the labelling procedure is written
  into the plan. Drafted candidates carry their evidence (dictionary dating, corpus counts), never
  change scores. The confirmation plan carries a leakage checklist (what development used, the
  held-out list's hash, that held-out words were filtered from every development output, that
  their labels came after the plan was frozen), after Kapoor and Narayanan's model info sheets
  ("Leakage and the Reproducibility Crisis in ML-based Science", 2023: leakage in at least 294
  papers across 17 fields, undetectable from the papers themselves); SemEval-2020 likewise kept
  its gold labels hidden.
- **A2 (step 5): controls are never replaced on their change score.** Replacing the controls
  that changed most lowers the bar targets must clear (the controls' 95th percentile), an
  outcome-driven exclusion of the kind that inflates false positives (Simmons, Nelson and
  Simonsohn 2011, "False-Positive Psychology"); controls that change a lot measure the method's
  background change and stay (Dubossarsky et al. 2017). Controls are replaced only for reasons
  that do not use their score, written into the step 5 code and the confirmation plan before any
  target is scored: wrong part of speech in practice, mostly a name or fixed phrase, frequency
  mismatch after cleaning, a documented change in a dictionary, or the author's blind reading of a
  few usages per period (as SemEval-2020 verified its stable words), done before scoring. The
  control set is then frozen for that comparison; every replacement is recorded with its reason.
- **A3 (steps 7.2–7.4): the author's labels are split and sampled so they can carry their jobs.**
  The same labels train counter 1b, build counter 2's sense vectors, set the "no sense fits"
  cut-off, measure error rates for the correction, choose the counting method and test it all;
  7.2's plan gave about 15 labels each to *component* and *having*, not the 50–70 per word 7.4
  assumed. (a) Building needs few labels: averaging contextual vectors per sense holds up with
  about three examples per sense (Loureiro et al., *Computational Linguistics* 2021), drawn from
  the author's labels and each list's example. (b) Error rates are the binding constraint: labels
  are drawn by stratified sampling, by period and by the sense a baseline counter predicts, with
  inclusion probabilities recorded and inverse-probability weights in every estimate, so rare
  senses are covered without bias (stratified evaluation: Kumar and Raj; importance-weighted
  evaluation for rare categories under 300 labels: Poms et al. 2021). (c) At least about ten
  labels per sense in each period where it exists, for error rates; below that, pooled error
  rates with honest intervals (replacing 7.4's per-period rates where the minimum is not met).
  (d) A written nested split per word: an outer split only for testing, building and calibration
  inside, so no label builds and grades the same thing (Varma and Simon 2006). (e) The counting
  method is chosen once, across all words (replacing 7.4's per-word choice, too thin per word).
  (f) Period balance applies only where a sense exists (an arriving sense has no earlier
  examples). Intervals rest on Vaz, Izbicki and Stern (*JMLR* 2019). Time: about 200 new labels
  (about 1–1.5 hours, in sittings) beside check 1a's 280, more only for words whose intervals are
  too wide.
- **A4 (step 4b): the corpus review screens fields instead of claiming rates it cannot show.**
  About 12 abstracts per field cannot show "wrong field under 5%" (zero problems still allows
  about 24%), and 130 pooled cannot show "junk under 2%" (rule of three: 2.3%). Following lot
  quality assurance sampling (a fixed sample and decision rule per group, keeping both kinds of
  misclassification tolerable; Hund, Bedrick and Pagano), each field gets 15 random kept
  abstracts and the rule "2 or more wrong-field abstracts means a closer look" (about 17% chance
  of flagging a fine field at a 5% rate and of missing a bad one at 20%); a field that triggers
  gets a second sample of 15. The 165 pooled cover the junk check (zero junk in 165 puts the rate
  likely under 1.8%). Thresholds are screening rules stated in advance, with intervals reported.
  The judging page shows each abstract's end as well as its beginning and highlights removed
  spans, so truncation and over-cleaning are visible. The wrong-field classifier is trained on
  data separate from what it flags and treats nested fields (AI inside computer science,
  cognitive psychology inside psychology) as compatible. First rebuild: 165 random kept, about
  120 dropped items, 80 flagged, 20 repeats, about 385 judgements (about 1 hour 5 minutes, in
  sittings), plus second samples only for fields that trigger.
- **A5 (steps 4.7, 4.8, 5, 6): first guesses, then adjust; nothing waits on later work.** The
  plan asked for answers before the work that produces them (caps, floor and download plan
  "after steps 5 and 6", which need the rebuilt store; controls chosen by within-period spread,
  which needs their encoding). Order instead: (1) provisional caps, the floor (500) and the
  download plan for the three thin words, from existing counts, written in the parameter files
  (step 9); (2) downloads, then the first full pass; (3) candidate controls chosen from the new
  text by part of speech and frequency trajectory, a pool larger than needed (about 60 per
  target) encoded so spread can be measured, the final 30 or more picked from it (encoding
  resumes by missing keys, 4.6); (4) the experiments, which may revise caps and the floor through
  the parameter files, DVC rerunning only what depends on them. A top-up download after the pass
  reruns the corpus-level rules (templates, near-duplicates) in full, not as a patch.
- **A6 (steps 2.2, 4.4–4.6, 4.9): stand-off annotation; the text is never edited.** Cutting removed
  text would shift every stored position, keeping it would leave it in the sentences, and the
  static text written during the pass would miss the corpus-level rules applied after it. The
  established answer is stand-off annotation (brat never modifies the text; the ISO 24612
  Linguistic Annotation Framework addresses annotations by character offsets; STAM likewise), and
  spaCy's document model is one: its tokenization is non-destructive (every token keeps
  `token.idx`), spans carry labels and character positions, the `SpanRuler` writes its matches
  into a span group (`doc.spans`), and `DocBin` saves span groups. So: the stored text is the
  repaired text, never edited after; every removal is a labelled range in a span group (footers,
  URLs, LaTeX from the pass; templates, near-duplicates and author-cap marks added by the
  corpus-level stage), copied into the abstracts table as a Polars list column of (start, end,
  reason); the views are derived after the corpus-level stage, never stored as edited text: the
  contextual view takes the sentences not marked removed (small ranges inside kept sentences stay,
  as 2.2 decided), the static view skips every marked range. Stage order: spaCy pass → corpus-level
  rules → derived views. No new library; the project's own code is the copy into the table and the
  derivation.
- **A7 (steps 4.6, 4.8, 9): DVC keeps what is costly or cannot be remade.** DVC's documentation:
  "Stage outputs are deleted from the workspace before executing the stage commands that produce
  them (unless `persist: true` is used)." Unchanged, that would wipe every stored vector when any
  upstream input changes (defeating 4.6's resume by missing keys), refetch the downloads when
  their code changes (contradicting 4.8's curated, approved downloads; OpenAlex's samples are not
  repeatable and cost credits), and rerun everything when a stage depends on all of `src/`. So:
  (1) the vector store and the annotation cache (`DocBin`) are `persist: true` outputs; encoding
  adds only missing keys; vectors for usages that no longer exist stay unread until a deliberate
  clean-up command removes them. The trade-off is stated in the lock's meaning: these outputs are
  their inputs plus earlier runs, which is safe because a vector is found by the hash of the exact
  text the encoder read. (2) Vectors are stored per encoder name and version
  (`vectors/<encoder>-<version>/`), since the content key covers the text, not the model; a new
  encoder or new weights write a new folder and never read the old one (4.4's encoder table
  carries the version). (3) Downloads are not a stage: an approved plan runs by hand and its
  output is registered with `dvc add data/raw/...`, versioned by hash and never regenerated by
  DVC. (4) Each stage lists the modules it depends on (the spaCy pass: the annotation, cleaning
  and language modules, the patterns file and its settings; not `senses/`), so a change reruns
  only what it can affect; the list sits beside the stage definitions and is checked when stages
  change.
- **A8 (step 8): a data catalog; jobs receive data, never paths.** Four places broke the rule
  that job packages never import each other: `senses/` needed the spaCy pipeline (cues, compound
  cues, the `DocBin` cache's vocabulary) built in `corpus/`; the nearest-sense and definition
  counters needed vectors of sense examples and definitions from `embeddings/`; `evaluation/`
  needed to run detection and counting on benchmarks; and "only storage reads and writes data"
  clashed with library-native loading. One principle fixes all four: jobs are plain functions of
  data, and the pipelines (or experiment runners) build objects and pass them in (dependency
  injection: Fowler 2004; "functional core, imperative shell": Bernhardt, "Boundaries", 2012;
  Percival and Gregory, *Architecture Patterns with Python*, 2020). (1) The spaCy pipeline is
  built in `corpus/annotation.py`; `pipelines/count_senses.py` passes it (or its vocab) to
  `senses/`. (2) The encoding stage encodes the usages and the sense lists' examples and
  definitions into the same per-encoder store (A7), same content key; `senses/` gets those
  vectors from the pipeline. (3) `evaluation/` supplies benchmark data and scoring; the experiment
  runner composes load → measure → score, so `evaluation/` imports neither `detection/` nor
  `senses/`. (4) `storage/` becomes `catalog/`, the data catalog (Kedro's `DataCatalog`; Fowler's
  Repository, 2002; Cockburn's ports and adapters, 2005), responsible for the project's data at
  rest: it knows every dataset by name in one catalog file (path, format, encoder and version),
  which `dvc.yaml` reads through its templating (`vars:`), so each path is written once; it loads
  and saves each dataset with the format's own function (Polars for Parquet, `DocBin`, gensim's
  save and load, TOML and TSV for `curated/`), models the project trains included; it runs the
  pandera checks on read and write; it hands out large datasets as streams. It is not external
  services (pyalex in `corpus/`, the Batch API in `senses/`, the Hugging Face Hub in
  `embeddings/`, whose results are saved through the catalog; model weights are pinned by name and
  revision in the encoder table and kept in Hugging Face's cache), not research decisions, and not
  versioning (DVC's lock; `provenance.py` shrinks to reading it for reports, or goes). The rule:
  jobs never see a path; only pipelines and experiment runners call the catalog, one typed `load`
  per dataset (a grouped load would lose the types under strict mypy; bundle only if three stages
  load the same group); spaCy receives its patterns and exceptions as lists
  (`ruler.add_patterns`, `attribute_ruler.add`).
- **A9 (steps 7a, 7.2–7.5): sense lists are frozen before labelling; check 1a's labels are
  mapped, not assumed comparable.** (a) 7.3 sent "no listed sense fits" usages back to 7a's gap
  finding, which could add a sense after the labels and counts exist: the labels would be against
  the old list, the counters evaluated on it, and a sense added after seeing shares is a forking
  path (Gelman and Loken 2013). Annotation practice revises the scheme in a pilot, freezes it,
  then annotates, and re-annotates what a later change touches (OntoNotes revised sense groupings
  until agreement reached 90% before they were used, Hovy et al. 2006; content analysis pretests
  and freezes the codebook, recoding after any change; Pustejovsky and Stubbs's MATTER cycle
  tracks guideline versions). So: 7a's aids run on the rebuilt store, the author reviews the
  proposed senses, the lists are frozen as a numbered version, then the author labels; a gap
  found after the freeze makes a new list version, the author re-labels only the items marked
  "other or unclear", and the frozen version's counts stay the confirmatory result, the new
  version's reported beside them as exploratory. (b) Check 1a's 280 items stored no work id;
  rerunning its seeded sampling recovered 279 of them (2026-11-12). 71% come from abstracts with
  one mention, where the first mention is the random mention; "unclear" was 2–5% whatever the
  position (7 items in all, weak evidence). The 279 mapped labels serve building and error rates,
  each recording its mention position and whether it was the only mention; once A3's labels
  (random mentions) exist, one comparison on multi-mention abstracts checks first against random
  mentions, and only a gap there removes the 1a labels from the error-rate set (sample selection
  bias: Zadrozny 2004). Items whose abstract the new cleaning drops are excluded; items whose
  sentence reads differently are listed for the author. Side finding: abstracts with several
  mentions carry the changing sense far more often (57% against 26%), so 4.3's unit matters;
  reports state that shares are shares of abstracts.
- **A10 (steps 5, 6.3): the statistics hold when many words are screened.** (1) scipy's
  permutation p-value is (exceedances + 1) / (shuffles + 1), so 1,000 shuffles cannot go below
  about 0.001, while Benjamini–Hochberg at 5% needs the strongest word at 0.05 / m: above 50
  screened words nothing can pass (simulated 2026-11-12: a very large change got p = 0.001, which
  fails at 100 and 300 words). Liu, Medlar and Glowacka (2021) escalate from 1,000 to 100,000
  shuffles; here every word gets 100,000 (the similarity matrix computed once, each batch of
  shuffles one matrix product, timed on one word before relying on it), and the rule is that
  1/(shuffles + 1) is at least ten times below 0.05 / m. (2) The bootstrap keeps scipy's default
  BCa interval (better coverage than percentile: Efron 1987; DiCiccio and Efron 1996; its
  jackknife costs about as much as the bootstrap with a vectorised statistic); a
  `DegenerateDataWarning` stops the run instead of switching method. (3) J's chance level falls
  with the number of usages (simulated with no change: its 95th percentile 0.0030 at 100 usages
  per period, 0.0017 at 200, 0.0006 at 500), so unequal counts would let the controls' noise set
  the bar; Liu et al. found high-frequency words almost always significant and subsampled. Each
  comparison therefore draws the same n per period for the target and every control (n = the cap,
  or the target's count if lower, never below the floor; seed recorded); candidate controls with
  fewer than n usages are not eligible (A5's pool absorbs this); n is a parameter, so a change
  reruns the target and its controls together. (4) Benjamini–Yekutieli (`method="by"`, valid
  under any dependence) is reported beside Benjamini–Hochberg, since words in one field share its
  topic drift. The two simulations in (1) and (3) (two-sense unit vectors, J by
  `permutation_test`) become tests of the statistics module.
- **A11 (steps 3.2–3.6, 4.1, 4.3, 4.5, 4.6): usage selection is fully specified.** (1) The
  mention for an abstract is drawn among its in-scope mentions only (the sampling frame is the
  eligible units: Kish, *Survey Sampling*, 1965); every mention stays in the usages table. (2) The
  draw is by hashing, not a seeded shuffle, so one abstract's choice never depends on other
  abstracts or row order (hash-based assignment as in online experiments: Kohavi, Tang and Xu
  2020): the in-scope mention with the smallest hash of (seed, work id, content key). The usage id
  becomes (work id, word, content key) instead of the mention number, which shifted whenever a
  rule found an earlier mention; the content key's offsets are within the sentence, so the id
  changes only if that mention's sentence does. (3) Matching uses spaCy's full lemma after the
  exceptions file (so *prompted* is found; 7.1 counts verb senses), and each usage records the
  static view's token beside it, so the two pipelines' counts reconcile. (4) No second full
  pass: the order is spaCy pass → corpus-level rules → the author's lemma review → exceptions
  applied to the cached documents by an `AttributeRuler` → matching and usages. Measured
  2026-11-12 on 2,000 abstracts: parsing 25.6 s, loading the cache and applying exceptions 0.2 s
  (about two minutes for the corpus on one process). The cache keeps spaCy's default `DocBin`
  attributes: spaCy 3.8.18's defaults include `NORM`, `SENT_START` and `ENT_ID` (the
  documentation lists older defaults), and a round trip kept sentences from `senter`, token-level
  norms and span groups, while an explicit attribute list without `SENT_START` lost the
  sentences; only extension attributes (lingua's sentence languages) need `store_user_data=True`.
  A round-trip test (sentences, norms, span groups, languages) guards it. (5) The lemma review:
  types where both spaCy and the voted map merge and disagree, spelling-only differences left
  out, numbered 577 (0.024% of tokens) on 2026-11-12, very skewed (the 50 most frequent cover 89%
  of their tokens, 100 cover 93%, 400 cover 99%); the voted map was right on nearly all of the 25
  most frequent (*senses* → *sense*, not *sens*; *motives* → *motive*, not *motif*). Rebuilt after
  the new cleaning, the review covers the 100 most frequent types and every case touching a
  word-list word (about two sittings); for the rest, 30 drawn at random are checked, and if the
  voted map is right on at least 27 (set before reading, as in A4), its lemma is accepted for the
  whole tail as exceptions; otherwise more are reviewed.
- **A12 (steps 4.7, 5, 7.1): the word list becomes three tables, and words carry their part of
  speech.** 4.7's single list could not hold the stable checks (7.1), the held-out words (A1),
  matching by form (*having*), or controls that belong to one comparison (*attention* has three
  comparisons with different frequency trajectories, and A10 fixes n per comparison). Each kind of
  observational unit gets its own table (Wickham, "Tidy Data", 2014): `curated/words.tsv` (lemma,
  part of speech, match by `lemma` or `form`, `held_out`), `curated/comparisons.tsv` (comparison
  id, target, field, the two periods, purpose: documented change, stable check, held-out change or
  held-out stable; n per period) and `curated/comparison_controls.tsv` (comparison id, control,
  status and reason, so A2's score-blind replacements are recorded). Roles follow from the data
  (a target heads a comparison, a control is listed under one), and step 5's "never both target
  and control" check becomes a join; the "pipelines" column goes (the contextual pipeline reads
  the comparisons, the static pipeline its own list). A word is a (lemma, part of speech) pair,
  as in SemEval-2020 Task 1, which joined English targets with their POS tag "because English
  frequently combines various POS in one lemma and many of our target words underwent
  POS-specific semantic changes" (Schlechtweg et al. 2020, §3.2): usages and frequency counts
  cover that part of speech only, so mixed-POS words are narrowed rather than excluded, and a
  control's part of speech is always its target's (nothing for the author to fill in). Controls
  are matched per comparison on part of speech and on frequency in each of the two periods, as
  SemEval matched "the same POS and comparable frequency development between C1 and C2"; the
  frequency tolerance is the smallest that yields at least 30 eligible candidates (SemEval's Latin
  set took the smallest p "for which a control word could be found"), a score-blind rule in the
  sense of A2 and A5. Generated candidates are written to `data/` as proposals and enter
  `curated/` after the author's approval.
- **A13 (step 6b): the static question has three options, and part of its premise is checked.**
  Besides TWEC (whose package, `cade`, is inactive: step 0, row 1a) and PPMI-SVD, the library route
  is skip-gram cells trained apart with gensim and aligned with scipy's `orthogonal_procrustes`
  (Hamilton, Leskovec and Jurafsky 2016). In SemEval-2020 Task 1, 7 of the 10 best systems in
  subtask 1 (8 of 10 in subtask 2) used type embeddings, with Orthogonal Procrustes among the
  usual alignments (Schlechtweg et al. 2020, §§6–7); that predates XL-LEXEME, and how skip-gram
  compares with PPMI-SVD for change detection is still to be checked before step 6b is decided.

## Housekeeping (outside the steps)

- [x] pyarrow: nothing imported it and nothing else depended on it; removed (2026-11-12).
- [ ] Uncommitted loose files: `data/sense_lists/round_a_prime_v2_draft.toml`, the untracked
  round E and round Q label files. (`scripts/dev_harness.py`'s changes were committed as a record,
  `fdf83cc`.)
- [x] Commit this agenda, `pipeline_map.md`, the four design drafts and the code review
  (2026-11-12).
- [ ] Push: the local commits wait for the author's go.
- [ ] Devlog 07: waiting for the author's review.
- [ ] bazzite's copy of the repository: behind the Mac; sync before any GPU run.
- [ ] Docs pass: go through the 23 documents in `docs/`, merge what overlaps, retire what is no
  longer relevant, sort the rest into `reference/`, `records/` and `design/` (step 8), with step 10.

## Reference: what moves to spaCy

| Moves to spaCy | Stays outside |
|---|---|
| Tokens and sentences (`senter` with abbreviation special cases), URL and e-mail detection, possessives and contractions, spellings (`norm_`), lemma votes from the same pass, target matching (`Matcher`, `PhraseMatcher`), cue words (`Matcher`, `DependencyMatcher`), lemma exceptions (`AttributeRuler`), the name filter (entities), removal spans (`SpanRuler` span groups, A6), the annotation cache (`DocBin`), language (lingua as a component) | OpenAlex downloads, template sentences (using spaCy's sentences), near-duplicates (datasketch), mojibake (affected abstracts dropped), static models (gensim), encoders (transformers), the classifier (scikit-learn) |

## Reference: merges

| Today | After |
|---|---|
| 4 cleaning rule sets, 3 language checks, 2 sentence splitters, about 6 regex tokenisers, 2 URL regexes | 1 cleaning step in the spaCy pass, with lingua |
| 6 duplicate and shared-text rules | work id plus MinHash, and template sentences |
| 2 lemma maps | 1 map, plus the `AttributeRuler` |
| 5 copies of the usage key, plus 2 schemes | the usage id and the content key, each defined once (4.5, A11) |
| Divergence 3 times, unit rows 7, logistic regression 4, torch device 4, cell reading 6 | one library helper each |
| 2 Claude labellers | 1 |
| 2 neighbour implementations | 1 |
| 4 control-word files | one set per purpose, from one selector |
| 2 sense-list schemas | 1 |
