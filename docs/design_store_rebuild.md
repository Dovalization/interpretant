# Design draft: rebuilding the usage store

*Draft, 2026-11-12, for review; nothing here is decided. It proposes how the next store is built
on the stack settled on 2026-11-11 (README, Stack; `docs/spacy.md`), and lists the decisions it
needs.*

## The store today

`scripts/build_vector_store.py` exports usages per part (a grid field, or the word-targeted
cells), merges them into one manifest (`data/store/usages.jsonl.gz`), encodes them on the homelab
GPU into float16 shards, and indexes key to shard and row. Each record holds the key, word,
source, field, decade, year, work id, the context text and the target's start and end.
Quality rules version 2 apply to every cell; each abstract gives one occurrence of a word, with
its regular plural; usages are tidied as every round tidies them.

The 2026-11-11 audit found it sound (four cells re-derived identically) but limited:

- **Thin cells.** *attention* in AI has 186 and 174 usages in the 1990s and 2000s, *platform* in
  communication 142 in the 2000s, *alignment* in AI 105 in the 2010s.
- **Controls.** Only 15 of round X's 30 control words are in the store.
- **Text handling by regex.** Sentences are split by a regex that breaks on *e.g.*, *et al.* and
  *Fig. 2*; targets are found by `word_pattern` and `TARGET_VARIANTS`. The cleaner,
  `preprocess_text`, deletes curly apostrophes instead of spacing them, which fuses 542,162
  tokens (*Kant’s* → *kants*, *one’s* → *ones*, *it’s* → *its*), and it keeps hyphenated words
  whole while spaCy splits them, so the lemma votes never reach hyphenated types
  (`docs/spacy.md`, Cleaning text with spaCy).
- **Duplicates.** Only exact duplicates are removed; a preprint and its published version both
  stay, which shrinks distances within a period.
- **Language.** Checked per abstract, so the non-English half of a bilingual abstract can be a
  usage's sentence.
- **Format.** A gzipped JSON lines manifest, read whole.

## Proposed pipeline

1. **Sources, each abstract once.** The grid's sample cells, the full cells and the word-targeted
   cells (`data/raw/openalex_words`, plus any new downloads; see
   `docs/design_data_acquisition.md`), de-duplicated by work id the way the lemma votes are.
2. **Cleaning.** Quality rules version 2 as now, then near-duplicates: MinHash with
   locality-sensitive hashing (datasketch) on word 5-shingles, one abstract kept per group (the
   earliest published, else the first seen), every removal recorded with its group.
3. **One spaCy pass.** Over each kept abstract with case and punctuation, after Unicode
   normalisation and math stripping: sentences, tags and morphology, lemmas in context, named
   entities. The tokenizer gets special cases for academic abbreviations (*cf.*, *Fig.*,
   *Sect.*, *Eq.*, *et al.*, *approx.*) and loses the infix rule that splits hyphens; `senter`
   sets sentences (the parser only if the dependency cues need it). The small model on the CPU
   for the whole pass; optionally the transformer model on the GPU for the sentences that hold
   targets. An `AttributeRuler` loads `data/lemma_overrides.tsv`, so lemmas follow the same
   reviewed list as the lemma map. lingua, as a pipeline component, gives each sentence a
   language and a confidence.
4. **Processed text from the same pass.** The text the static models and the lemma map read is
   written from the tokens, replacing `preprocess_text`: punctuation, URLs, e-mail addresses and
   the possessive *'s* dropped, each other token's `norm_` lowercased and folded to ASCII. The
   lemma votes and the type counts then share one tokenisation, and the votes are taken from
   this pass instead of a separate one.
5. **Targets.** A `PhraseMatcher` (phrase targets) and a `Matcher` on lemma and lower-case form
   (single words, their inflections and spellings) replace `word_pattern` and
   `TARGET_VARIANTS`. Each match becomes a span in a `targets` span group, with the word, its
   tag, morphology and lemma, and whether it sits inside a named entity.
6. **Annotated documents kept.** The target-bearing abstracts are saved as `DocBin` files per
   cell, so later rounds read tags, dependencies and entities without parsing again (cues as
   `Matcher` patterns, dependency collocations).
7. **Manifest.** One Parquet table, written with Polars, one row per usage: the current fields
   plus the sentence (and its character offsets in the abstract), tag, morphology, lemma, entity
   label, sentence language and confidence, near-duplicate group, and a content hash that serves
   as the key.
8. **Encoding.** As now: XL-LEXEME and DeBERTa on the homelab GPU, resumable float16 shards,
   keyed by the content hash, so an unchanged usage keeps its vector across rebuilds.
9. **Checks.** pandera schemas on the manifest, run on every rebuild (below).
10. **Datasheet and versioning.** `docs/datasheet.md` with the counts per cell and per removal
   rule, regenerated with the store; a committed manifest of each built file's content hash and
   the configuration that made it.

## Checks (pandera)

- Keys unique; every offset inside its sentence; the sentence slice equals a form of the word.
- Year inside the cell's decade; field in the grid's taxonomy; source one of the known sources.
- No work id in two cells of the same decade unless the cells are a sample and its full cell.
- Share of target sentences not read as English under a set bound, per cell.
- Usages per cell at least the floor a measure needs (the thin cells above should fail until
  the new downloads arrive).
- Every control word present in every cell where its target is.

## How the new store is checked against the old

On the usages both stores hold (same abstract, same occurrence), compare the text, the cell and
the target offsets; count what each change of rule removed or added (near-duplicates, language,
sentence boundaries); then re-baseline round X on the new store before any new experiment.

## Decisions needed

1. **One or every occurrence per abstract.** One keeps abstracts independent; every occurrence
   triples some cells and lets a measure weigh abstracts unevenly.
2. **Context window.** The sentence (XL-LEXEME was trained on sentences) or a fixed span of
   words around the target.
3. **Near-duplicate threshold.** A Jaccard similarity of 0.8 on 5-shingles is a common start;
   a sample of pairs around it should be read before fixing it.
4. **The transformer model for target sentences.** About 5.5 hours of GPU time for the whole
   corpus at the measured 11,000 words a second; less for target sentences alone.
5. **Scope.** The target words, the 30 control words, round A′'s words, the shortlist and the
   long list (capped as now at 300 abstracts per cell).
6. **Hyphenated words.** Kept whole as now (*self-organising*, *meta-analysis* are terms), or
   split as spaCy does by default.
7. **Numbers.** Kept as tokens, replaced by one placeholder, or dropped; today they are kept and
   split at decimal points (*3.5* → *3 5*).

## Order of work

1. Near-duplicate study on a sample, to fix the threshold (decision 3).
2. The spaCy annotation pass, the processed text written from it and target matching, on one
   field, compared with the current store and the current processed files; then the lemma map
   relearned on that text and its audit rerun.
3. The manifest, the pandera checks and the datasheet.
4. The full rebuild, then encoding on the homelab.
5. Round X re-baselined on the new store.
