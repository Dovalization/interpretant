# Plan: language detection before the word repairs and the spaCy pass

*Proposed 2026-11-16, awaiting the author's approval. Measured on the corpus after run 8
(`abd16663`); the scripts are read-only and their numbers are below.*

## Why

Language data is read by many steps, but today it exists only after the spaCy pass, the
longest stage. Three problems follow:

- **The word repairs cannot see language.** Runs 7, 7c and 8 change foreign text as if it were
  English: run 8 splits Spanish *considera* into *consider a* and German *Themen* into *the
  men*, and English words put into a foreign sentence can make lingua read it as English, so it
  is kept instead of removed. Most of run 8's remaining errors are foreign words ending in *a*
  (*performa*, *playa*, *conducta*; 462 words, 781 uses), and 9 foreign words sit on the
  kept-joined list for this reason only.
- **The spaCy pass annotates works that are dropped as not English** (18,791 in the last build,
  2.7% of the corpus).
- **Guards that need language before the pass have nothing to read.**

## What the measurements show

| Question | Result |
|---|---|
| Early check (lingua with the stage's logic, on the text after `repair_text`) against the final sentence languages: whole abstract mostly English | agree on 704,930 of 705,283 works (99.95%) |
| The same, on the 6,083 damaged texts (run 8's guard) | mostly English 5,824 early, 5,825 final: glued words do not confuse lingua |
| Sentences with the same boundaries early and final | every one has the same language label (3,858,808 of 3,858,808) |
| Sentence boundaries: spaCy's plain `sentencizer` | 92.7% of the final sentences identical; it cuts after *Vol.*, *No.*, *pp.* |
| Boundaries with the project's tokenizer settings and sentence rules in front of the `sentencizer` | 93.9% identical, 347,132 sentences against 348,148 final (55,184 texts) |
| Final sentences spanning early sentences of two languages | 5,433 of 4,160,537 (0.13%) |
| Works where only the early check finds foreign text (2,797) | citation fragments from the plain sentencizer ("51, No.", "1, pp."): median 6 characters, 690 of 749 read as Italian at confidences of 0.15-0.43 |
| OpenAlex's `language` field | not usable: "en" for 20,818 of the 21,205 works the final check reads as not English |
| lingua's low accuracy mode | no faster on this corpus (14.1 s against 14.2 on 58,774 works) and 97.0% agreement on the whole-abstract verdict: not used |
| Early check's cost | 196 s on ten workers with the plain sentencizer (today's stage: 121 s); the project's sentence rules add about 30 s |

## Design

1. **Stage order.** `build_linked_tables` → `repair_texts` → `detect_languages` →
   `count_word_repairs` → `annotate_corpus` → `clean_corpus` → `build_corpus` →
   `report_corpus`. `repair_texts` is split from today's word-repair stage: it merges the cells
   and runs `repair_text`, writing the base texts per part.
2. **`detect_languages`, moved.** Per part in worker processes: the project's tokenizer settings
   and sentence rules with spaCy's rule-based `sentencizer` (no models) split the base texts;
   lingua reads them as today (19 candidates, full accuracy, one thread per worker). A sentence
   too short to be read on its own takes its abstract's verdict instead (the threshold to be
   measured; the fragments above are mostly under 20 characters). It writes each sentence's
   language on base-text offsets.
3. **`count_word_repairs`, guarded.** Runs 7, 7c and 8 count and change only sentences read as
   English (or undetermined in an English abstract). The stage records each text's edits and
   writes the language spans moved to the final text's offsets, so later stages read them on the
   text they annotate.
4. **`annotate_corpus` skips works with no English sentence.** Their rows are still written,
   dropped as `not_english`, from the language table alone.
5. **`clean_corpus` gives each spaCy sentence a language.** A spaCy sentence takes the language
   of the early sentences it overlaps; one that spans two languages (0.13%) is read again by
   lingua. Removals and the `not_english` rule work as now.
6. **Today's late language stage goes;** its code is reused by the early one.

## For the author to decide

- The fragment rule's threshold (characters or words), measured before it is set.
- Whether undetermined sentences in an English abstract get the word repairs (proposed: yes).
- The skip rule: only works with no English sentence at all (proposed), or the full
  `not_english` rule moved early, which would need word counts without the cleaning patterns'
  removals.
- Whether the 9 foreign words leave the kept-joined list once the guard runs (proposed: yes,
  each checked in the run).

## Steps (one commit each)

1. The tokenizer settings and sentence rules shared between the spaCy pass and the early
   sentence split.
2. `repair_texts` split from `count_word_repairs`, outputs identical.
3. `detect_languages` on the base texts, with the fragment rule; measured against today's.
4. The guard in the word repairs and the language spans moved to final offsets.
5. `clean_corpus` gives spaCy sentences their language, with the re-check.
6. `annotate_corpus` skips works with no English sentence; `clean_corpus` writes their rows.
7. `dvc.yaml` reordered, tests, a full profiled run, the agenda and `docs/performance.md`.

## Expected effect

Foreign sentences stay as written; the 462 content-word-plus-*a* errors and the foreign keep-list
entries go; language data is available to every stage; the run takes about the same time (the
early stage costs about a minute more than today's, the spaCy pass and cleaning skip about 3% of
the works).
