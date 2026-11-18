# The preprocessing phase: steps 0–8 to a standard, then a version of record for inference

*Plan of record for agenda 9.2, approved by the author on 2026-11-17. Each step's own document records its pre-registration and results; this file is updated when the plan changes.*

## Context

The author decided on 2026-11-17 (agenda 9.2, `a9017529`) to finish preprocessing and corpus
extraction before any encoding or model. That means the rest of steps 3 and 4 and stages 6–8,
each to a quality bar fixed before it runs.

After step 8, the corpus and the usages table become the version of record that inference starts
from. Later changes stay possible, but each one is deliberate:
- it has a reason in evidence (an audit, the junk gate, a measurement);
- it is recorded as a new version;
- it reruns the steps downstream of it.

**Corpus cleaning and lemma work are not priorities in this phase,** since rounds S1, S1b and L1
found them below the static measurement's noise. They come back if later evidence, for example
from the junk gate or the contextual measures, shows they matter.

Three read-only scouts (code, data on bazzite, old path) and a design pass informed this plan. So
did a check of the libraries' docs and source (spaCy 3.8.18, Polars 2.1.0, pandera 0.36.0) and of
the literature on preprocessing and usage extraction for semantic change; sources are cited in
the steps that use them.
The author settled four questions on 2026-11-17:
- spellings merge into the corpus's more frequent form;
- the contextual view drops sentences fully covered by removals for any reason (D2);
- the cut-off last sentence of a truncated abstract is flagged, not drawn;
- the old grid commands stay until the static-models step.

**Step 7 (junk gate) gets its own planning pass**; below are only the facts and decisions it
starts from. Step 3 had its own pass on 2026-11-18, recorded under step 3.

## Ground rules for every step

- **Commits.** One logical change per commit, with the gates on each: pytest, mypy, ruff,
  lint-imports, vulture.
  - Pre-registrations go in their own commit, before the data they judge is read.
  - Each data change gets its own `chore(dvc)` lock commit.
- **Fingerprint modules.** Never edit `corpus/annotation.py`, `corpus/attribute_rules.py` or
  `corpus/repair.py`; they are the cache's fingerprint. The transformer annotator must not call
  `build_pipeline`.
- **Runs.**
  - Measured runs happen on bazzite only, since the Mac's tables are stale. Sync with
    `git push bazzite main`, then run DVC with `source .venv/bin/activate`.
  - After a `catalog/config.py` change, run `dvc commit -f` on the stages that change cannot
    affect.
  - Live progress, with an update every 2 minutes.
- **Blinding.** No target's drift is read. Held-out words are sealed: never printed, only hashed.
- **Judging.** Audits use calibrated agents (two Sonnet judges and an Opus adjudicator, with the
  prompt and codebook saved under `curated/judging/<step>/`). The author judges a blind
  calibration set, and the agents are accepted at ≥90% agreement. Otherwise the author judges.
- **Audit bars are sampling plans.** Each is a pair (n, c): n items, at most c errors. It is chosen
  from two error rates, one acceptable (AQL) and one that must fail (RQL), each with about 5–10%
  risk of a wrong verdict. Each plan is checked by its curve of pass chance against true error rate
  (NIST/SEMATECH e-Handbook 6.2.2), and rates are reported with Wilson intervals (Brown, Cai and
  DasGupta 2001). Per-word checks are screening: a failed word gets reviewed, not the whole step
  failed.
- **spaCy use** (checked against spaCy 3.8's source and docs):
  - rulers and matchers are built on the documents' own `spacy.blank("en").vocab`, since a bare
    `Vocab()` silently breaks `LOWER` matching and the recogniser;
  - cached NORM is never overwritten, since the small model's recogniser reads it;
  - persistent keys use `hashlib` sha256 (`Expr.hash` is not stable across Polars versions).

## Step 0: leftovers (Mac, docs)

- The datasheet limits and the dated notes in the living docs are already done (`c5ad4171`).
- **`docs(agenda)`:** soften 9.2's wording as in the Context above. After step 8 the corpus is the
  version of record for inference; changes stay possible when evidence calls for them, versioned
  and rerun downstream. Cleaning and lemma work are deprioritised, not closed.
- **`docs(lemma_review)`:** a dated correction. The review's "voted map" was
  `data/processed/lemmas.tsv`, the simplemma-era map (36,726 rows), not the spaCy-voted
  `lemmas_voted.tsv`. It shaped the `disagree` group and the tail check, and is not rerun (round
  L1).
- **Author's decisions:**
  - the push of 40+ commits;
  - `~/interpretant-old` on bazzite;
  - devlog 07;
  - where the loose label files go (`data/sense_lists/round_a_prime_v2_draft.toml`, rounds E and
    Q).
- **Before step 1 on bazzite:** check `dvc status` is clean (R_B `daadb4ed`), and copy R_B aside
  as `data/interim/static_view_r_b` for step 4's alignment check.

## Step 1: the word table (`curated/words.tsv`, 4.7)

1. **`feat(scripts)`:** `scripts/evidence/word_table_draft.py` runs on bazzite in about 5 min. Per
   candidate word it reports:
   - part-of-speech shares (coarse and fine) over kept works, with the exceptions applied;
   - counts and part-of-speech shares per decade, since a change in a word's syntactic role gets
     read as semantic change (Kutuzov, Velldal and Øvrelid 2022);
   - 10 seeded examples of each minority reading at ≥5%;
   - the phrase targets' spaced and hyphenated counts.
2. **`feat(catalog)`:** `catalog/words.py`, with `Word(lemma, pos, match: Literal["lemma","form"],
   held_out: bool)` and `load_words(catalog)`.
   - Validators: lowercase; letters, space or hyphen only; unique lemma; a UPOS tag.
   - Plus the catalog entry, the `CatalogConfig` field and tests.
3. **`data(curated)`:** `curated/words.tsv` after the author's approval, then the exceptions
   invariant test pointed at it.
4. **Held-out set,** if drafted now (recommended): Claude drafts about 40 words with evidence, as
   in 6.2. The author reviews them. They are sealed with a hash.

**Bar:** the author approves the table. The held-out column is fixed before step 6 runs and
before any target result is read.

**Decisions at this step:**
- **Which words.** The 32 single-word and 7 phrase static targets; the contextual words
  (*attention*, *prompt*, *platform*, *alignment*, *hallucination*, *transformer*, *cloud*,
  *embedding*); the stable checks (*component*, *having*); and the six-word shortlist (open
  question in 4.7).
- **The part-of-speech rule: one tag per word, or a set of accepted tags.** Data to decide on:
  - *embodiment* is 55% ADJ, which is tagger error;
  - PROPN appears on capitalised common nouns: *intelligence* 17%, *sign* 10%;
  - *learning* is 37% VERB, *control* 23%, *experience* 19%, *medium* 21% ADJ.
- **Phrase-target forms** (*information-processing*, 465 hyphenated).
- **`data/words_of_interest.txt`** stays frozen as the list that rounds S1, S1b and L1 used.

## Step 2: spellings (3.4), and retiring the safe part of the old path

**Design.** `corpus/spelling.py::spell(word, spellings)` maps each hyphen segment one-for-one after
`fold`, so token starts stay aligned.
- `static_tokens` applies it to the written token, whether that is a lemma or the text.
- Step 6 applies it to the folded lemma used for matching.
- The table is spaCy's `lexeme_norm`, recovered by probing with the corpus's types, filtered, and
  saved as `curated/spellings.tsv`, with `corpus/lemmas.py::american` as the backstop for derived
  words.
- The filter drops: Title duplicates; keys or values with stray spaces; about 16 informal norms
  (*cos*, *thx*); wrong pairs (*gases*, *buses*, *storey*, *tonne*, *analyses → analyzes*);
  values without a letter; and any key that is a word-table lemma.
- **Direction (the author):** each pair merges into the corpus's more frequent form, so *dialogue*
  stays and *behaviour* becomes *behavior*. Targets keep the word table's spelling.
- **Compound variants** (spaced, hyphenated, solid) move to step 3.

**Commits:**
1. `corpus/spelling.py` with `spell`, `build_spelling_candidates` and `find_mapped_words`, plus
   tests.
2. `scripts/evidence/spelling_table.py` on bazzite, in 3–5 min.
3. `docs/spellings.md`, the pre-registered bar.
4. `catalog/spellings.py` with `Spelling(form, standard, source)` and validators (no chains, the
   standard has a letter), plus the catalog entry.
5. `curated/spellings.tsv` once the bar passes.
6. A test that no word-table lemma is a spelling form and that no standard merges two lemmas.
7. `static_tokens(doc, spellings)`. `_PartJob` carries the spelling tuple; `dvc.yaml` gets the new
   dependencies. The rebuild runs in step 4.
8. The exceptions builder drops the 10 spelling-only `hyphen_rule` rows (e.g. *edge-colourings*)
   and *hypothesises*, and reads its targets from `words.tsv`. The TSV is regenerated in a separate
   commit.
9. **Retirement,** in order, each with its tests:
   - (a) Rewrite `tests/test_pipeline.py` to start at the grid commands.
   - (b) Move `sample_chunks` into `embedding/grid.py`.
   - (c) Remove `corpus phrases` and `corpus/phrases.py`.
   - (d) Remove `corpus lemmas votes|learn|apply` and `LemmaMap`, `normal_form`,
     `learn_lemma_map` (the protected list). Keep `voted_lemma`, `lemma_votes`, `read_overrides`,
     `american` and `_lemmatizer`.
   - (e) Remove `corpus audit` and all of `audit.py` except `load_target_words`.
   - (f) Remove `corpus openalex ingest` and `files.py`.
   - (g) Update `CLAUDE.md`, the README, the dated notes and the agenda ticks.

   **Kept:**
   - the grid commands, until the static-models step (the author);
   - `openalex download`, `OpenAlexSource`, `drift/senses.py`;
   - **`preprocess_text` and `TARGET_VARIANTS`.** Live sampling scripts and `lemma_votes` use
     them, and removing them changes seeded draws in 3 of 96 raw cells. They retire with those
     scripts (agenda 1b).

The map applies to the written form after the lemma, because spaCy's lemmas are not
spelling-normalised (*organisations* → *organisation*).

**Evidence:**
- **Merging is supported.** Unmerged variants get separate, noisier vectors and split counts
  (Nguyen and Grieve 2020); merging variants improved term recognition (Nenadic, Ananiadou and
  McNaught 2004).
- **The British/American mix drifts across decades** (Gonçalves et al. 2018), which is a confound
  for diachronic work.
- **No study settles the direction,** but one corpus-wide choice is what matters, and the author's
  rule is one.
- **Pairs that differ in meaning between varieties** (*programme*/*program*, *cheque*/*check*) go to
  the census for a decision. Whether to also use the `breame` list's flags for such pairs (a small
  Apache-2.0 package) is the author's call before anything is installed.

**Bar (pre-registered):**
- A seeded 150 pairs (75 weighted by tokens, 75 uniform over types), passing with ≤3 wrong.
  AQL 1% passes 94% of the time; RQL 5% passes 5%. (The earlier 100/3 plan passed a 5% error rate
  26% of the time.)
- A census, with errors fixed one by one: every pair touching the word table, the top 50 by
  tokens, and the about 150 reversed-direction pairs.
- Judged by agents calibrated on 30 pairs the author judges blind.
- The "no target mapped away" test passes.

## Step 3: compound variants (3.4), and no phrase list (3.5)

*Planned in its own pass and decided by the author on 2026-11-18. Three scouts (code, data on
bazzite, literature and library docs) and a design pass informed it. The audit's pre-registration
and results go in `docs/compounds.md`.*

**Decisions:**
1. **No phrase list.** Single words stay unjoined in the static view. The 7 phrase targets are
   studied only in the contextual pipeline (agenda 3.5). The evidence:
   - The main semantic-change studies and SemEval-2020 join no phrases; they change only the target
     tokens (Schlechtweg et al. 2020; Dubossarsky et al. 2019).
   - Joining changes the vectors of the words inside the phrases (Camacho-Collados and Pilehvar
     2018), and no study shows it helps diachronic measures.
   - A list would need a large review: 13,940 noun-phrase candidates occur at least 100 times, and
     their ranking mixes terms with stock phrases (*present study*, *other hand*). C-value
     precision falls past the top few thousand (Zhang et al. 2008); the best TermEval 2020 system
     reached F1 46.7.
2. **Compound variants are merged, all three forms.** The hyphenated, solid and spaced forms of one
   two-part compound become the form the corpus uses most often, counted over kept works for the
   whole corpus (Nenadic et al. 2004; spelling normalisation as in VARD2 and CCOHA). A word-table
   lemma keeps its own spelling. Each merged family's share per decade is reported, since the
   spelling shift is a finding in its own right (the Shorter OED dropped some 16,000 hyphens in
   2007; Sun and Baayen 2021 find hyphenated compounds persist).
3. **Spaced forms merge only as noun-noun compounds.** A spaced pair counts and merges only where
   both tokens are tagged NOUN or PROPN, are adjacent in the document and in one sentence:
   - *data base*, *web site* and *user interface* merge;
   - *is a*, *would be* and *how to* never do;
   - nor do adjective-first pairs (*long term*, *high level*, *real time*): before a noun the
     hyphen is grammar, not spelling.

   Hyphen and solid pairs merge whatever their first part (*on-line*, *online*).
4. **Targets are guarded.** A spaced pair containing a word-table lemma never merges, and no lemma
   is written away. Splits that add target tokens are allowed and reported per target
   (*user-interface* becomes *user interface*).
5. **One entry per spaCy token.** This replaces the span column first planned, so the view stays
   aligned with R_B:
   - a merged spaced pair writes its standard in the first token's entry and `""` in the second's;
   - a token whose standard is spaced writes it with its space;
   - a hyphen and solid pair rewrites the entry.

   Lines for the static models drop `""` and split entries on spaces
   (`corpus/cell_versions.segment_words`).
6. **The map is audited like the spellings:**
   - a seeded sample of 150 pairs passes with at most 3 wrong;
   - the census is fixed one by one;
   - two Sonnet judges and an Opus adjudicator use a saved prompt, judging at the word level;
   - the author judges 30 sample pairs blind, and the agents are accepted at 27 or more of 30.

**Design.**
- **Order in `static_tokens(doc, spellings, compounds)`:**
  - each token is folded, spelled per hyphen segment, then rewritten whole by the one-token
    compound map;
  - spaced pairs then merge left to right without overlaps, looked up on the spelled words and
    gated on the two tokens;
  - accented tokens are left alone by both passes.
- **`corpus/compound_spelling.py`** holds the gate (`is_noun_compound`), the map and the invariant
  check, shared by the view, the builder and the audit so all three count the same thing.
- **The map** is `curated/compounds.tsv` (`form`, `standard`, `source`), loaded by
  `catalog/compounds.py`. Its validators:
  - two letter-only parts joined by a hyphen, a space or nothing;
  - the same letters on both sides;
  - no chains, and one standard per family.
- **The builder** `scripts/evidence/compound_table.py` counts each family's forms over kept works
  per decade (2–4 min on bazzite). Its filters are recorded with their reasons:
  - letter codes (fewer than 4 letters, Roman-numeral or vowelless segments);
  - three-part tokens, reported but not paired;
  - spaced forms whose share passing the gate is under 0.5;
  - anything touching a word-table lemma.
- **The audit's scripts copy the spelling audit's** (the second occurrence is copied, not
  abstracted).

**Facts (data scout, 2026-11-18, kept works):**
- 1,296 compounds occur both hyphenated and solid (each form at least 20 times); 2,845 both
  hyphenated and spaced, the spaced counts swollen by syntax; 290 in all three forms.
- The largest hyphen and solid pairs are line-break remnants (*re-search* 42 against *research*
  173,445; *meth-od*, *in-formation*).
- A seeded sample of 30 such pairs held 28 real variants (*pre-test*, *non-profit*) and 2 letter
  codes. About 50–60 of the 1,296 are letter codes.
- None of the 7 phrase targets has a solid form.

**Left out, counts reported:**
- three-part compounds (*state-of-the-art*);
- forms under the floor of 20;
- accented tokens;
- adjective-first spaced and hyphenated pairs.

Solid forms are not spelled per segment, so *colourblind* and *color-blind* fall in separate
families.

## Step 4: the full static view

**Commits:**
- `scripts/evidence/static_view_check.py`, comparing the new view with the R_B copy.
- The bar, committed first.
- `dvc repro build_static_view` (5–9 min with the check), then the lock commit. Then the results
  and the datasheet updated (spellings, compounds, the entry convention of step 3).

**Bar:**
- **Alignment:** 100% of works have the same starts as R_B, so the same entry counts.
- **Changes explained:** 100% of changed entries come from the maps (a spelling, a one-token
  compound, or a merged standard followed by `""`). A rewrite the maps call for but the new view
  did not apply is allowed only in accented tokens, as the text shows.
- **Targets:**
  - Each single-word target's word count equals R_B's plus the tokens spelled or compounded into
    it.
  - No target token of R_B changes.
  - Zero merged pairs hold a word-table lemma; held-out words are reported as a count.
- **Straddles** (a merged pair cut by a removal) are reported.
- **Spot check:** 80 changed entries in context, at most 1 wrong. A 0.5% error rate passes 94% of
  the time; 5% passes 9%.
  - The entries are drawn from kept works, outside removed ranges, by the smallest sha256 of
    (seed 20261126, work id, start).
  - They are judged by the two Claude Sonnet judges and the Claude Opus adjudicator, with
    `curated/judging/static_view/prompt.md` (added 2026-11-18, before the rebuild). The author
    calibrated these judges on the compound audit, where they agreed with the author on 30 of 30.
  - The items quote corpus text and stay out of git; the key and the verdicts go in
    `data/review/static_view/`.

**Results (2026-11-18).**
- **The run.** The view was rebuilt on bazzite at code `e937c9f0` in 132 s: 705,283 documents and
  104,176,115 entries, the same as R_B. The lock is `8fad6e8f`, the run record `43e4897f`. The
  other stages were re-committed unchanged first, since `catalog/config.py` only gained the map
  paths. The check (`scripts/evidence/static_view_check.py`, code `e62fb8f4`; summary in
  `data/review/static_view/summary.json`) ran in 25 s.
- **Alignment passes.** Every work is in the same part with R_B's starts.
- **Changes explained passes, with one source the bar did not name.**
  - 263,582 entries are rewritten (a spelling or a one-token compound), 29,415 are split and
    34,276 pairs are merged, all as the maps say.
  - All 437 rewrites left unapplied are in accented tokens.
  - 14 entries change for another reason: the seven exception rows dropped in step 2
    (`docs/lemma_review.md`, 2026-11-17) had kept a plural or *-s* verb unlemmatised. Examples are
    *four-colourings* and *re-singularises*. Without them, the view writes spaCy's lemma, spelled
    (*four-coloring*), as its rule for plural nouns and *-s* verbs says. The change was intended in
    step 2; the bar named only the maps.
- **Targets pass.**
  - No target entry of R_B changes, and no merged pair holds a word of the table. No held-out word
    is affected.
  - Over kept works, with removed ranges skipped, the targets gain words: *behavior* 16,876 (from
    *behaviour*), *information* 710, *learning* 460, *user* 367, *interface* 316, *image* 213,
    *attention* 167, *system* 152, *feedback* 146, and 10 more under 100.
- **Straddles:** none.
- **Reconciliation** (kept works, removed ranges skipped):
  - Merges: 33,432 against the builder's 33,461. The 29 short are overlaps, where an earlier merge
    took the first word (*amyloid beta* before *beta protein*, *family work* before *work life*).
  - One-token rewrites: 126,767 against 126,765.
- **The spot check passes.** Both judges called all 80 entries right: 64 rewrites, 8 merges and 8
  splits. With no disagreement, the adjudicator was not needed.

## Step 5: the contextual sentence view (2.2)

**Rule (the author):**
- A sentence of a kept work is excluded when every non-space character lies inside removal ranges
  (D2: 125,398 sentences, 2.9%).
- Partly covered sentences stay whole, so offsets don't move.
- A mention inside a removed range is never a usage (115 target tokens).
- The last sentence of a truncated work is marked `cut_off` (64,813 works; 1.18% of target
  mentions). It stays in the view and in the usages table, but is not drawn.
- Sentences of undetermined language (37,141) and the 1,603 malformed works are kept, with their
  language and length recorded.

**Commits:**
1. The rule and the audit's bar, as docs.
2. `corpus/sentence_view.py`, with `build_sentence_view(abstracts) -> pl.DataFrame`. One row per
   sentence: `work_id, sentence_number, start, end, language, confidence, word_count, exclusion
   (null|removed|cut_off)`. Plus `is_inside_removal(start, end, removals)` and tests.
3. `catalog/sentence_view.py`, with a pandera check (unique `(work_id, sentence_number)`) and the
   entry `data/processed/sentence_view`.
4. `pipelines/` and `cli/build_sentence_view.py` and the DVC stage. Run on bazzite (1–2 min), then
   the lock commit.
5. `scripts/review/sentence_audit.py`, then the results.

The input is the abstracts table, which already holds sentence boundaries and per-sentence
languages identical to the sentences table (4,584,443 rows).

**Bar:**
- **Cut precision:** 50 abstracts (6–7 per decade), every cut judged, precision ≥98%.
- **Boundaries:** 150 eligible target sentences (about 19 per decade), ≤5 with a wrong boundary.
  AQL 2% passes 92% of the time; RQL 7% passes 4%.
- **Why the boundary audit matters.** spaCy's general model has done poorly on abbreviations in
  scientific text: 88% of GENIA sentences right against 97% for scispaCy (Neumann et al. 2019);
  Sadvilkar and Neumann 2020 report the same pattern. The pass added abbreviation special cases,
  so the audit decides. If it fails, the failing cases are compared with pySBD or scispaCy (a
  dependency decision for the author), applied to the contextual view only, not the cache.
- **"Fully covered"** is computed on non-space characters against the merged removal ranges
  (Polars: merge the ranges per work, then `join_asof`), because a sentence can span two adjacent
  ranges.
- **Judges:** agents calibrated on run 3's 50 abstracts the author judged
  (`data/review/run3/verdicts.jsonl`), at ≥45/50 agreement. This also closes 4b's postponed
  sentence re-check.

**Results (2026-11-18)** are in `docs/sentence_view.md`. The audit failed at first (cut precision
97.8% on the author's curtailed inspection); merge rules found by scanning every cut, and a split
rule for breaks the pass missed after capitals, brought it to 98.3% precision, 0 of 80 wrong joins,
1 of 80 wrong splits and 0 of 150 wrongly bounded target sentences over five rounds of fresh agent
audits. The agents' verdicts stand by the author's decision after two recalibrations fell one short.
OCR correction was left to step 7.

## Step 6: matching and the usages table (4.1–4.6), in `src/interpretant/usages/`

**Layers:** `pipelines > review > usages > corpus > catalog`. `usages` goes in the catalog's
forbidden list, and agenda 8.1 is updated (it named `corpus/target_matching.py`).

**Commits:**
1. **Docs.** `docs/usages_audit.md`, plus the saved judge prompt and codebook. It pre-registers the
   model comparison (4.9) and the matching audit.
2. **The layer** in import-linter.
3. **`usages/identifiers.py`:**
   - `build_content_key(sentence, start, end)` = sha256 of `"{start}:{end}:{sentence}"`, first 20
     hex characters (the old `key_of` scheme);
   - `build_draw_key(seed, work_id, word, content_key)`, sha256 over all four. The word is in the
     key so different words' draws are independent: Kohavi et al. (2009) found correlations unless
     the experiment's id was hashed with the unit's. A chi-square check confirms the draws are
     uniform.
4. **`usages/target_matching.py`:**
   - `TargetMention(word, token_index, start_char, end_char, compound, position)`;
   - `build_target_matcher(vocab, words, spellings)` and `match_targets(doc, matcher)`;
   - a word matches by (spelled lemma = word) OR (spelled folded form = word), so VBG *learning*
     is kept as a row, except words matched by form only (*having*);
   - a `LOWER` regex rule for hyphen parts (4.2);
   - token patterns for phrase targets (modifier `LOWER` plus head lemma, and the hyphenated
     form), which handle inflected heads;
   - tests: *media* stays apart from *medium*, *behaviour* is found as *behavior*, *self-attention*,
     *neural networks*.
5. **`usages/entities.py`:**
   - before recognition, `doc.set_ents([], default="missing")`, since existing O labels block every
     prediction;
   - the small model's `ner` on `span.as_doc()`; its tok2vec is its own, so no re-tagging;
   - **sentence context.** Entities are recognised on the target sentence alone, as agenda 4.1
     decided from its own measurement (99.87% agreement with the whole abstract on 15,018
     mentions). The literature finds that document context helps (Luoma and Pyysalo 2020;
     Schweter and Akbik 2020), so the model comparison re-checks sentence against abstract on its
     sample;
   - **the transformer gets one sentence per `Doc`,** since it has no sentence splitter (`doc.sents`
     raises E030 without one);
   - the transformer path: `spacy.load("en_core_web_trf", exclude=["parser"])` on a `Doc` built
     from the cached tokens, then the exceptions ruler (it has no `senter`, so it never goes
     through `build_pipeline`);
   - `require_entity_annotation(doc)`, which raises `MissingEntityAnnotationError`.
6. **`usages/usage_selection.py`:**
   - `is_in_scope(pos, word, entity_label, name_labels)`;
   - `draw_mentions(usages, *, seed)`: the smallest draw key per (work, word), among in-scope
     mentions in eligible sentences (not `cut_off`);
   - invariant tests: adding or removing abstracts, reordering rows or adding words changes no
     other draw.
7. **`catalog/usages.py` and `usages/checks.py`:**
   - the columns of 4.6, plus `sentence_exclusion, is_in_scope, is_drawn, model, static_token`;
   - a pandera schema: unique (work_id, word, content_key), ordered offsets, 20-hex keys;
   - `check_usages_against_text`.
8. **The stage.** `pipelines/` and `cli/build_usages.py` (`--field`, `--decade`); the
   `params.yaml` section `build_usages: {seed, model, name_labels}`; the DVC stage depending on the
   cache, abstracts, sentence view, static view, words, spellings and exceptions. The small model
   runs in workers; the transformer runs in the main process on the GPU.
9. **`scripts/evidence/model_comparison.py` (4.9):** 300 mentions per word, seeded and stratified
   over decades, plus 50 disagreements judged. Then the decision, set in `params.yaml`.
10. **`scripts/review/usages_audit.py`** on 8 development parts, then the results and fixes. Any
    `words.tsv` change needs the author.
11. **After step 7's pre-registration:** the full `dvc repro build_usages`, then the lock and
    counts per word × decade (matched, in scope, drawn), with no drift read.

**Bars (pre-registered):**
- **Model:** switch to the transformer if more than 2% of sampled usages, pooled, change their
  in-scope flag or lemma.
- **Precision:**
  - pooled over 20 drawn in-scope usages per word, about 800 items, passing with ≤23 errors. AQL 2%
    passes 96% of the time; RQL 4% passes 6%;
  - per word, ≤2 errors in 20 as screening (a word with a true 5% error rate still passes 93% of
    the time), so a failed word is reviewed;
  - 10 out-of-scope matches per word, where 3 or more wrongly excluded sends that word's tag rule
    back.
- **Recall:**
  - an independent regex (forms, plurals, British spellings, hyphen parts) is compared with the
    matcher;
  - 20 regex-only hits per word are judged;
  - missed mentions ≤2% per word and ≤1% pooled.
- **Judges:** calibrated agents, with 60 items the author judges blind (about 2,500 agent
  judgements).

**Evidence from the literature that the step uses:**
- **Part of speech.** The SemEval-2020 and DWUG benchmarks join each English target with its
  part of speech and take sentence-level uses (Schlechtweg et al. 2020, 2021).
- **Names.** A part-of-speech tag alone lets name uses through (Martinc et al. 2020, *lane* in
  *Rose Lane*), and names inflate change scores (Kutuzov, Velldal and Øvrelid 2022). So the entity
  check stays, with a sensitivity check with and without name-filtered uses.
- **One use per abstract.** Benchmarks don't do this, but corpus linguistics supports it: repeat
  mentions in one text are strongly dependent (Church 2000; Lijffijt et al. 2016).
- **Compounds.** No published precedent marks only part of a compound, so compound uses are
  flagged and results reported with and without them (4.2 already allows this).

**Decisions at this step:**
- **Short sentences.** The benchmarks require at least 10 tokens per use. The usages table records
  sentence length, and whether very short sentences leave the draw is decided with the data in
  step 6's pre-registration;
- a stratified comparison sample or one field (4.9 says one field; the sample is recommended);
- the name labels (proposed: PERSON, NORP, FAC, ORG, GPE, LOC, PRODUCT, EVENT, WORK_OF_ART, LAW,
  LANGUAGE);
- the seed.

**Retires here:**
- the two `key_of` copies (`scripts/build_vector_store.py`, `scripts/review/build.py`);
- `pilot_labellers.usage_id`;
- the old store layout once nothing reads it.

`drift/senses.py`, `preprocess_text` and `TARGET_VARIANTS` go when their scripts are replaced
(agenda 1b).

## Step 7: the junk gate (its own planning pass, committed before step 6's full run)

**Facts:**
- Corpus junk is 8% (5–13%; 16 of 200).
- No per-record junk labels exist except run 3's 250 rows.
- The calibrated judges' setup was never saved (no prompt, no codebook; the 7 overturned verdicts
  exist only as prose).
- The unit is the record behind each drawn usage. 57% of abstracts hold a target, so each unique
  record is judged once.
- The `anthropic` SDK (in the `labels` extra) allows a scripted judge with saved prompts and raw
  answers.
- **Why the gate is per decade.** A 2026 OpenAlex study (Kim, Holst and Ginis, a preprint) flags
  about 21% of English abstracts before 2000, 12% for 2000–2019 and 6% after. In the old decades
  web-scrape artefacts and bibliographic metadata dominate. Junk that rises in old decades can
  look like semantic change. The study released a classification prompt that the judge setup can
  adapt.

- **OCR damage (the author, 2026-11-18).** No OCR correction before this step. The gate scores
  OCR damage per target usage (for example, the share of its words no lexicon knows) and tests
  whether OCR-heavy usages move J; correction is considered only if they do.
  - OCR errors harm sentence splitting most (van Strien et al. 2020, with spaCy) and degrade
    embeddings (Todorov and Colavizza 2022). No study measures their effect on semantic change
    detection itself.
  - Post-OCR correction gains are modest and uneven: the ICDAR 2019 competition's best English
    result cut character errors by about 11%, and LLM correction helps English more than other
    languages (Kanerva et al. 2025).
  - OCR'd works gather in the early decades, so a correction would itself differ by decade.

**To decide:**
- the threshold (absolute, e.g. ≤10% per word and decade, or relative to the corpus);
- n per cell and the sampling rule;
- what failing does (judge the whole cell and leave out the junk, or drop the cell from its
  comparisons);
- the judge setup: scripted or agents, and the calibration rebuilt from run 3 plus the author's 7
  corrections;
- storage: `curated/judging/junk_gate/` and `data/review/junk_gate/`;
- sealed handling of held-out words;
- rerunning after a top-up.

## Step 8: the gap report (4.8)

1. **`curated/comparisons.tsv`** (4.7), with `catalog/comparisons.py` (each target in the word
   table, valid fields and periods), approved by the author.
2. **The `params.yaml` section `report_gaps`:** floor 500 (provisional), caps by role (candidates
   300), and words sized by year (*attention* AI, *platform* communication, *alignment* AI).
3. **`usages/…count_usable_usages`:** kept regular abstracts with at least one drawn in-scope
   usage, per word × field × decade or year.
   - The floor applies to these usable usages, not to abstracts in general. That is how the
     literature sets floors: 100 uses per period in SemEval-2020 and DWUG, at least 50 in
     Montariol et al. 2021. So 500 is generous.
   - The report adds each word's dispersion over abstracts (Gries 2008, DP) per cell.
4. **`fetch-word-totals`** (by hand): OpenAlex's count per cell under the floor, at 10 credits
   each, guarded by the daily budget, registered with `dvc add`.
5. **The `report_gaps` stage** (`review/gap_report.py`), writing `data/processed/gap_report.md` and
   `.parquet`. Held-out rows are sealed.
6. **`curated/download_plans/<date>.tsv`**, approved by the author. Then `download-abstracts
   --plan`, reusing `OpenAlexClient.download_cell`, registered with `dvc add`.
   `scripts/fetch_sense_words.py` is deleted.
7. **Downloaded cells as a second source** through stages 2–3: parts per word, the `source` column,
   skipped by the static view, one source per word in the usages. This is the heaviest piece and
   gets a short plan of its own when reached.
8. **Rerun stages 2–7** for the new abstracts, gate them again, and record the version of record
   for inference.

**Bar:**
- The plan covers only comparison cells under the floor.
- After the top-up, every planned cell is at or above the floor, or its shortfall is documented.
- The regular cells' kept works and removals change by <0.1% (`compare-corpus`).

**Decisions:**
- the comparisons;
- the floor and the caps;
- the 11 old downloads in `data/raw/openalex_words/` (register them, or download again under a
  plan);
- a full rerun of the corpus-level rules (4.8) or applying the regular corpus's decisions to the
  downloads.

## Order, gates and runtimes

0 → 1 → 2 → 3 → 4 → 5 → 6 → 7's pre-registration → 6's full run → 7's gate → 8.

- The held-out words are fixed before step 6.
- Step 6's code can be written while step 3's audit runs. Its full run needs step 4 (each usage
  records the static token).

**Runtimes on bazzite (estimates):**

| Run | Time |
|---|---|
| Step 1 draft | 5 min |
| Step 2 spelling table | 3–5 min |
| Step 3 compound builder and audit items | 2–4 min each |
| Step 4 build and check | 5–9 min |
| Step 5 sentence view | 1–2 min |
| Step 6, small model | 6–10 min |
| Step 6, model comparison | 5 min (GPU) |
| Step 6, transformer, full run | 1–1.5 h (GPU; cache annotations by sentence hash if chosen) |
| Step 8 report | <2 min |
| Step 8 rerun after a top-up | 25–40 min |

## For the inference phase (noted here, not in scope)

- **A shuffled-decade control:** a claimed effect must vanish on a corpus with its time labels
  shuffled (Dubossarsky et al. 2017).
- **Feed raw, cased sentences to XL-LEXEME** (Kutuzov and Giulianelli 2020; Kutuzov et al. 2022),
  with an ablation that replaces only the target with its spelled lemma, since the target's surface
  form biases contextual vectors (Laicher et al. 2021).
- **A small test of sentence against abstract context** (Martinc et al. 2020 raise it, untested).
- **Document-level bootstrap intervals** (Lijffijt et al. 2016).

## Verification

- **Each step:** the gates on every commit; new unit tests per module (named above); end-to-end
  pipeline tests on the synthetic corpus (`tests/corpus_catalog.py`); `tests/test_dvc_stages.py`
  for every new stage's dependencies.
- **Data:** each step's pre-registered bar, run on bazzite, with its results appended to the
  step's doc. `dvc status` clean after each lock commit, and `git diff --stat` empty on the three
  fingerprint modules before each run.
- **Invariants carried through:**
  - no target mapped away (exceptions, spellings and compounds);
  - static views aligned with R_B;
  - draws independent of other abstracts;
  - matching refuses documents without entity annotation;
  - held-out words never printed.
