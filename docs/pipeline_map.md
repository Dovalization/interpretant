# Pipeline map: what every step does today

*Draft, 2026-11-12, for review. Read from the code at commit `5250d0b`, not from the docs; each
step lists what it reads, how it works, what it writes, and what is fragile or inconsistent. It
is the input for choosing a strategy per step (the store rebuild, `docs/design_store_rebuild.md`)
and for documenting the chosen one. Nothing here is decided.*

## The two pipelines

The project has two pipelines that share only the downloaded abstracts:

- **Static:** one vector per word per cell (a field in a decade), learned from all of that
  cell's text (static word embeddings: TWEC, PPMI). It scans every word cheaply and gives
  readable neighbours, but a word with several senses gets one blended vector.
- **Contextual:** one vector, or one sense label, per occurrence of a word in its sentence
  (contextual embeddings: XL-LEXEME). It sees senses separately, at a higher cost, so it covers
  chosen words only, and it needs real sentences with case and punctuation.

```
                 OpenAlex ──► raw cells  data/raw/openalex[_full|_words]/<field>/<decade>.jsonl.gz
                                  │
          ┌───────────────────────┴────────────────────────┐
  STATIC (one vector per word per cell)          CONTEXTUAL (one per occurrence)
  ingest: rule set 1 + preprocess_text           cased_abstracts: rule set 1 ids, raw cased text
     → data/processed                            (+ quality v2 in the store only)
  lemmas apply → data/lemmatised                 one occurrence per abstract, its sentence
  phrases apply → data/phrased                   tidy_usage → store manifest
  twec grid / ppmi grid → models/                XL-LEXEME / DeBERTa → float16 shards
  drift grid, perplexity, heldout,               J and APD; sense lists, labellers,
  separability, direction → data/drift           cue classifier, adjusted counts
          └──────────────► export web, figures ◄───────┘ (static only)
```

The lemma votes sit between the two: they read the contextual pipeline's cased text and feed
the static pipeline's map.

No Makefile target or script runs either pipeline; the order is written only in `README.md`
(131–157) and `docs/corpus.md`, and several outputs need flags recorded only there.

## Stage 1: acquisition

| Step | How it works | Reads → writes |
|---|---|---|
| `corpus openalex download` (`openalex.py:392`) | Server-side filter: field (`primary_topic`, 11 slugs; ai and hci sit inside compsci, cogpsych inside psychology), years of the decade, `language:en`, `has_abstract`, research types. OpenAlex `sample` with seed 42, up to 10,000 works per cell, pages of 200; `--full` walks every work; `--word` adds `abstract.search` (stemmed). The abstract is rebuilt from the inverted index; the title is never fetched. Retries with backoff; stops cleanly when the daily budget runs out. | API key → `data/raw/openalex/<field>/<decade>.jsonl.gz` (`id, year, type, authors, abstract`) |
| `corpus openalex sources` (`openalex.py:427`) | Venue of each work, 50 ids per request, appended. | raw cells → `sources_<decade>.jsonl` |
| `scripts/fetch_sense_words.py` | For each planned word and cell, counts lines of the processed text that match the singular; any cell under 500 sends every planned cell of that word to a word-targeted download of 1,000 works. | processed text → `data/raw/openalex_words/<word>/…` |

Fragile: a cell is skipped whenever its file exists, whatever the seed, size or word; the gap
count (exact singular on processed text) and the download (OpenAlex's stemmed search) define an
occurrence differently.

## Stage 2: cleaning

Four rule sets, applied on different paths.

| Rule set | Where it runs | What it does |
|---|---|---|
| **Rule set 1** (`openalex.clean_cell`, :270) | `corpus openalex ingest` (static); also re-run by `cased_abstracts` to recover kept ids | Drops catalogue and citation records, web-page scrapes, abstracts under 5 words, non-English by `looks_english` (31 English function words against a French/Spanish/German/Portuguese list, at least 5%), exact duplicates within the cell; strips JSTOR and copyright footers; caps any author at 1% of a cell (seed 42; the cap is 1 work in cells under 200). |
| **`preprocess_text`** (`preprocessing.py:42`) | Inside rule set 1, so every processed file; also on single tokens in the lemma votes | NFKC; URLs and DOIs removed; lowercase; NFKD and non-ASCII deleted (Greek, CJK and curly apostrophes vanish); `$…$` math removed; every other non-letter, non-digit, non-hyphen becomes a space; `TARGET_VARIANTS` (26 hand-written forms) mapped to their lemma. |
| **Quality rules v2** (`quality.py`) | Only `scripts/build_vector_store.py` | Sentences (regex split, 40–300 characters) shared by 20 or more abstracts of the grid are "template" sentences: an abstract mostly made of them is dropped, otherwise they are removed; exam-dump spam dropped; non-English by sklearn's stop words (under 15%, only with 20 or more words). |
| **Usage tidying** (`classifier.tidy_text`, :159) | The store and the scoring scripts, after a usage is cut | A hand-written mojibake table, control and invisible characters, soft hyphens, spaces. |

Plus, in the contextual pipeline only: `cased_abstracts` strips the footers and a leading
"Abstract"/"Summary" label but keeps case and punctuation; `Sampling(quality=True)` drops
abstracts with more than 20% letters beyond Latin, after its own URL regex.

Fragile and inconsistent:

- **Quality v2 never reaches the static pipeline:** the processed, lemmatised and phrased text, the
  grid models and every static measure use rule set 1 only.
- **Three language checks** (`looks_english`, `english_share`, `foreign_share`), each with
  different evidence and bars; `looks_english` runs on ASCII-folded text, so non-Latin abstracts
  are counted as empty, not non-English.
- **Curly apostrophes fuse words** (542,162 tokens; `docs/spacy.md`).
- **Inline math** `\$[^$]+\$` also deletes prose between two dollar amounts.
- **The shared-sentence cache** is reused whenever the file exists, counted from
  `data/raw/openalex` only, double-counts works in nested fields, and is counted before footers
  are stripped but matched after.

## Stage 3: duplicates and shared text

| Mechanism | Where | Key |
|---|---|---|
| Exact duplicate in a cell | rule set 1 | the preprocessed text |
| Same work in two cells | lemma votes | work id, first seen wins |
| Same work in two fields of a decade | `senses.drop_shared`: rounds X, 1a, 1b, 2b | work id |
| Same text in two cells | `perplexity.remove_shared`: perplexity, separability, direction | exact text |
| Template sentences | quality v2: the store | sentence in 20+ abstracts |
| Repeated usage text | `classifier.boilerplate`: rounds T, U, A | text seen 5+ times |
| Nothing | `data/processed`, the audit's merged counts, the store (no `drop_shared`) | — |

Near-duplicates (a preprint and its published version) are removed nowhere.

## Stage 4: normalisation (static)

| Step | How it works | Reads → writes |
|---|---|---|
| `TARGET_VARIANTS` | Inside `preprocess_text`: 26 hand-written plurals and British forms of target words mapped to the lemma. | — |
| `corpus lemmas votes` (`lemmas.py:260`) | spaCy `en_core_web_sm` (tagger, lemmatizer) over each cased abstract once; counts `(type, tag, lemma)`, with types normalised by `preprocess_text`. | raw cells via `cased_abstracts` → `data/processed/lemma_votes.parquet` |
| `corpus lemmas learn` | Rules in `docs/spacy.md`: merges a type into its singular or base form when 80% of 5+ readings are plural nouns or third-person verbs; British → American by rules; reviewed overrides; protected target words. | processed type counts, votes → `data/processed/lemmas_voted.tsv` |
| `corpus lemmas apply` | Token lookup per line. **Defaults to the frozen grid's map**, `data/processed/lemmas.tsv`, not the one `learn` writes. | processed → `data/lemmatised` |
| `corpus phrases learn` / `apply` (`phrases.py:83`) | gensim `Phrases`, NPMI, `min_count` 20, threshold 0.5, connector words; 300,000 tokens per file in shuffled 1,000-token chunks (seed 42); phrases with one-letter parts dropped, two-part phrases whose joined form is more frequent turned into joins, target phrases forced. | lemmatised (`openalex` only by default) → `data/phrased` |

Fragile: two lemma maps (`TARGET_VARIANTS` before, the learned map after) overlap; votes and
counts use different tokenisations, so hyphenated types never get votes; `phrases apply` must be
run by hand for `openalex_full`; the phrase model sits in `data/lemmatised` while
`docs/corpus.md` says `data/processed`.

## Stage 5: static models

| Step | How it works | Reads → writes |
|---|---|---|
| `twec grid` (`embedding/grid.py:72`, `twec.py:85`) | Compass: word2vec skip-gram on 200,000 tokens per main cell (shuffled chunks, seed 42), 100 dimensions, window 5, `min_count` 5, 10 epochs, 5 negatives, 8 workers. Each cell: own vocabulary (`min_count` 1), vectors and context vectors copied from the compass, context vectors reset to the compass after every epoch (alignment by a frozen context matrix, no Procrustes). | `data/phrased/openalex` (compass and cells), `openalex_full` (cells only) → `models/twec_grid/` |
| `ppmi grid` (`ppmi.py:102`) | Per cell: vocabulary of words with 20+ uses, symmetric unweighted ±5 window, PPMI with smoothing 0.75, truncated SVD to 100 dimensions (seed 42). Cells are not aligned. | `data/phrased` → `models/ppmi_grid/` |

Fragile: `TWECTrainer`'s own defaults (300, 10, 10, 4) disagree with the CLI's; resuming reuses
an existing compass and skips finished cells whatever the current settings; the compass sample
is always seed 42, so the seed-43 grid shares it; 8 gensim workers are not deterministic; words
missing from the compass train their context vectors freely; `models/twec_grid_unlemmatised`
was built from `data/phrased_unlemmatised`, recorded nowhere.

## Stage 6: static measures

| Step | How it works | Reads → writes |
|---|---|---|
| `drift grid` (`drift/grid.py:228`) | Pairs: consecutive decades of a field, and every two fields in a decade. Scores every word with 150+ uses in both cells: cosine drift (aligned vectors) and neighbourhood shift (Jaccard distance of the 25 nearest, drawn from words with 50+ uses in both cells). Up to 10 equal-count frequency bands; a word is a candidate when it is above the 95th percentile of its band on both measures. | `models/twec_grid`, word lists → `data/drift/within_field.parquet`, `cross_field.parquet` |
| `drift perplexity` / `decompose` (`perplexity.py`) | Per decade and field: 1,000 abstracts held out, Kneser-Ney bigram (discount 0.75) trained on an equal token budget; excess = cross-entropy under another field's model minus under the field's own. `decompose` splits the 1990→2020 fall into mix and rate, retraining its models. | `data/phrased/openalex` → `cross_perplexity*.parquet` |
| `drift heldout` | Held-out minus training cross-entropy per cell (10% held out, up to 5,000,000 training tokens). | `data/phrased` → `heldout.parquet` |
| `drift separability` | TF-IDF and logistic regression telling fields apart (2,000 abstracts per field, 5-fold). | → `separability*.parquet` |
| `drift direction` | A classifier trained on 1980 and 1990, tested on later decades: relative gain and movement. | → `direction_assignments.parquet` |

Fragile: running `drift grid` on PPMI models would put unaligned cosines into the flags (no
method switch); the measure thresholds are duplicated between CLI defaults and the constants the
export imports; `drift perplexity`'s defaults do not produce the corrected file the export and
figures read (that needs flags in the README); `decompose` and `perplexity` differ in default
seeds (1–5 against 42); the vocabulary is per decade, so unknown words pool differently across
the decades compared; `heldout` and `perplexity` both define `MAX_TRAINING_TOKENS`, at 5M and 1M.

## Stage 7: usage extraction (contextual)

| Step | How it works |
|---|---|
| Reading (`senses.cased_abstracts`, :87) | Re-runs rule set 1 on the cell for its kept ids, then re-reads the raw file for cased text with footers and the leading label stripped. |
| Matching (`word_pattern`, :115) | `\bword(s?)\b`, any case: the word and its regular plural only; matches inside hyphenated compounds. |
| Sampling (`sample_occurrences`, :146; `sample_many`, :178) | Shuffles each cell with a seed per word and cell, takes the **first** match in each abstract, up to a cap per cell. |
| Context (`context_around`, :122) | The sentence holding the match, by the regex `(?<=[.!?])\s+(?=[A-Z0-9("“])` (it breaks after *e.g.*, *et al.* or *Fig.* before a capital); cut to 100 words around the target. |

Each round then applies its own mix of `drop_shared`, quality v2, `tidy_usage` and `boilerplate`
(Stage 3), so the store's usages and the earlier rounds' usages are not the same population.

## Stage 8: the store and encoding (contextual)

| Step | How it works | Reads → writes |
|---|---|---|
| `build_vector_store.py export` | Per part (a field, or the word downloads): `cased_abstracts` → quality v2 → `sample_many` → `tidy_usage`. Scope: 11 named words and the rung-3 controls uncapped, the shortlist uncapped, the rest of the long list capped at 300 per cell; seed 1. Key: sha256 of `start:end:text`, 20 characters. | raw cells, word lists, `longlist.json`, shared-sentence cache → `data/store/part_*.jsonl.gz` |
| `merge`, `encode`, `index` | One manifest; duplicate keys marked; shards of 50,000 float16 vectors (finite values checked), resumed by file existence; key → shard and row index. | → `data/store/usages.jsonl.gz`, `vectors*/`, `index*.json` |
| Encoders (`pairs.py`) | XL-LEXEME (pinned revision): target marked `<t>…</t>`, centred in 128 subword tokens, mean of the whole sequence, 1,024 dimensions. DeBERTa-v3-large (pinned): 256-token window, mean of the last layer over the target's tokens. | — |

Fragile: shards are named by position, so a manifest that changes between runs would mix them;
the usage key is defined five times across scripts (and two other schemes exist); `drift senses`'
SciBERT encoder never uses CUDA, while XL-LEXEME does and fine-tuning requires it.

## Stage 9: contextual measures

- **Pair measures** (`pairs.py`): APD (1 minus the mean cross cosine), within-set distance, J
  (APD minus the mean within), PRT.
- **Round U** (DWUG EN, Spearman against graded change, 20 period shuffles) and **round X** (the
  documented, weak and quiet cases on the corpus; found when above the 95th percentile of 200
  shuffles and the 90th percentile of the 30 control words; 150 usages per cell, 100 minimum).
- **Checks:** 1a (markers and the author's reading of 20 usages per cell), 1b (planted
  *attention* usages at known shares, and a placebo), 2b (the author's 0–4 judgements of usage
  pairs against XL-LEXEME, parsed from markdown).

## Stage 10: senses (contextual)

| Step | How it works |
|---|---|
| Sense lists (`data/sense_lists/*.toml`) | Hand-written senses with definitions and cue words; two schemas (DWUG cues, corpus lists with `out_of_scope`); six drafts used only by the review scripts. |
| Labellers (`labels.py`) | Claude Code CLI in blocks with repeats (`Labeller`), the Message Batches API one usage per request (`BatchLabeller`, Sonnet 5.5), TypeSafe JEV (`JevLabeller`, yes at 0.7, no at 0.3); about 100 words of context with the target marked. |
| Cue classifier (`classifier.py`) | Cues in a five-token window (`<`, `>`, `!` syntax) label the easy usages; a logistic regression on the vectors labels the rest (cue first); counts corrected with a confusion matrix (adjusted classify-and-count, NNLS). |
| Induction (`induction.py`) | ModernBERT substitutes ("or even [MASK]"), Louvain communities, neighbours and Wiktionary glosses: the round Q evidence. |
| Fine-tuning (`finetune_classifier.py`) | The top four XL-LEXEME layers and one head per word on cue seeds; CUDA only. |

Fragile: the logistic regression and Jensen-Shannon divergence are each written three or four
times; two parallel Claude labellers do one job; context differs per step (a 5-token cue window,
100 words for the labellers, 128, 256 or 512 subword tokens for the encoders).

## Stage 11: control words

Four sets from four scripts: `control_words.txt` (frequency-matched everyday words, for the
grid), `control_words_cross_field.txt` (lowest variation across fields), `control_words_senses.txt`
(30 words, round X and check 1b), `control_words_rung3.txt` (10 words, round A and the store; no
generating script). The files are parsed three ways.

## Stage 12: outputs

| Step | How it works | Reads → writes |
|---|---|---|
| `corpus audit` | Documents, tokens and target counts per cell; a transition is reliable when both decades have 150+ target uses. Sums nested fields, so shared works count twice. | `data/phrased` (README) or `data/processed` (default) → `audit.json` |
| `export web` (`export.py:158`) | Explorer JSON: neighbours per target and cell, drift tables, perplexity, predictions parsed from markdown with regexes. | models, drift tables, audit → `web/data/*.json` |
| `figures` | The five README figures. | `models/twec_grid`, perplexity, audit → `figures/*.png` |

Fragile: the explorer's neighbours come from one cell's pool while the measured neighbourhood
shift uses the pool of words frequent in both cells, so the displayed neighbours are not the
measured ones (and the neighbour code is written twice); two figures embedded in
`docs/known_answers.md` have no generator; figure footnotes hard-code result claims.

## Cross-cutting problems

1. **No single cleaning path.** Four rule sets (Stage 2) and six duplicate rules (Stage 3), each
   applied on some paths only; the static pipeline never gets quality v2 or near-duplicate removal.
2. **Many tokenisations.** `preprocess_text` plus whitespace split, spaCy, and six regexes
   (`[a-z]+`, `\w+`, `[a-z][a-z-]{2,}`, `[a-z']+`, `[A-Za-z]+`, `[a-z]{4,}`).
3. **Two sentence splitters,** both regexes, both breaking on academic abbreviations.
4. **Three language checks** with different evidence.
5. **Two lemma maps** and two URL regexes.
6. **Occurrence counted differently** by the gap check, the sampler and the audit (singular on
   processed text, word plus regular plural on raw text, exact processed token).
7. **Different usage populations** per round (Stage 7).
8. **Reproducibility rests on conventions:** no pipeline runner, flags recorded only in the
   README, resumes decided by file existence (downloads, compass, cells, shards, the
   shared-sentence cache), defaults that disagree between a class and its CLI.
9. **Code placement** (`docs/review_codebase_2026-11-12.md`): corpus cleaning in `drift/`,
   scripts imported as libraries, helpers copied across scripts.

## Strategy questions, by stage

| Stage | Question |
|---|---|
| Acquisition | Should a cell's file record the parameters that made it, so a changed seed or word cannot silently reuse it? |
| Cleaning | One cleaning path for both pipelines (the spaCy pass of the store design), with which of the four rule sets kept? |
| Duplicates | One rule (work id, plus near-duplicates by MinHash) applied once, before either pipeline? |
| Normalisation | One lemma map (drop `TARGET_VARIANTS`), learned on the same tokens it is applied to? |
| Static models | Rebuild the grid on the new processed text, or keep it frozen as a record? |
| Static measures | Which of perplexity, held-out, separability and direction stay in use? |
| Usage extraction | One occurrence rule and one sentence rule (spaCy), shared by every round? |
| Store | Keys by content hash and shards named by key range, so changes cannot mix shards? |
| Senses | One labeller interface; one classifier and one divergence function in the library? |
| Controls | One control-word set and one selection script, or a set per purpose, documented? |
| Outputs | Explorer neighbours from the measured pool? Generators for every committed figure? |
| Everything | A pipeline runner (Makefile targets or a script) that records flags and inputs? |
