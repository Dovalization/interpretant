# Plan: language detection before the word repairs and the spaCy pass

*Approved 2026-11-16 by the author, with the changes below the design. Measured on the corpus
after run 8 (`abd16663`); the scripts are read-only and their numbers are below.*

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
   measured with the project's tokenizer settings, which stop the cuts the fragments came from;
   if one is needed it reads lingua's confidence, 0.15-0.43 for the fragments, not a length). It
   writes each sentence's language on base-text offsets.
3. **`count_word_repairs`, guarded.** Runs 7, 7c and 8 count and change only sentences read as
   English (or undetermined in an English abstract), and run 8's damaged-text guard counts only
   splits in English sentences. The stage records each text's edits (a position and a change of
   length) and moves the language spans through them, so later stages read them on the text they
   annotate; tests check that every sentence the repairs leave alone maps to the same text,
   after joins that delete characters and splits that add them. *Changed after the second check
   run (2026-11-16, the author):* counting English sentences only made more kept abstracts worse
   than better (1,025 kept abstracts' final text differed from run 8's; about 20 of 30 samples
   worse). Decisions at the margins flipped ("roboticand", 10 uses apart to 1 joined in run 8,
   9 apart now) and foreign words lost the joined uses that kept them whole (*forma*, 497 joined
   uses in run 8, 92), so they were split in Latin terms of English texts. The counts and the
   damaged-text guard read every sentence again, as in run 8; only the changes are limited to
   English sentences.
4. **The spaCy pass annotates every work, as now.** Skipping works with no English sentence
   would save about 20 s: 21,633 works (3.1%) have none in the early check, but 410 of them are
   kept in the final build, so the rule needs cleaning's own `not_english` decision moved early;
   left for a change of its own.
5. **`clean_corpus` gives each spaCy sentence a language.** A spaCy sentence takes the language
   of the early sentences it overlaps; one that spans two languages (0.13%) is read again by
   lingua. Removals and the `not_english` rule work as now. *Changed after the check run
   (2026-11-16, the author):* a spaCy sentence inside a longer early sentence took that
   sentence's language, so foreign titles in English text were kept and some English sentences
   next to foreign text removed (about 14 of 20 sampled changes of 80+ characters were worse).
   Now a spaCy sentence takes an early sentence's language only when their boundaries are the
   same (93.9% of the 4,584,546 sentences); lingua reads the other 278,657 in their abstracts
   (88,652, 12.6%) as the language stage reads sentences, the whole abstract first.
6. **Today's late language stage goes;** its code is reused by the early one.

## Decisions

- A fragment rule only if the improved sentence split still needs one (measured in step 3).
- Undetermined sentences in an English abstract get the word repairs (decided: yes).
- No skipping in the spaCy pass for now (decided, see design point 4).
- The 9 foreign words leave the kept-joined list once the guard runs (decided: yes, each checked
  in the run). *Reversed after the check run (2026-11-16, the author):* off the list, they were
  split 54 times in sentences read as English, 33 of them foreign words in quotes or in
  sentences that run into foreign text and 11 broken English words, against 10 real repairs.

## Steps (one commit each)

1. The tokenizer settings and sentence rules shared between the spaCy pass and the early
   sentence split.
2. `repair_texts` split from `count_word_repairs`, outputs identical.
3. `detect_languages` on the base texts (a fragment rule if step 3 shows one is needed),
   measured against today's.
4. The guard in the word repairs and the language spans moved to final offsets.
5. `clean_corpus` gives spaCy sentences their language, with the re-check.
6. `dvc.yaml` reordered, tests, a full profiled run that compares all three word-repair decision
   tables as well as the outputs, the agenda and `docs/performance.md`.

## Expected effect

Foreign sentences stay as written; the 462 content-word-plus-*a* errors and the foreign keep-list
entries go; language data is available to every stage. A run takes about a minute longer (about
22 minutes): the early stage costs a little more than today's late one.

## Keep or revert (written 2026-11-16, before the third check run's results)

The third check run counts the word repairs over every sentence, as run 8 did, and applies them
only to English sentences. The change stays only if it is objectively better than run 8's
pipeline (the author):

- **Population:** abstracts kept by cleaning in both run 8 and the check run whose final text
  (the text outside the removals) differs.
- **Test (automatic):** each region where an abstract's repaired text differs from run 8's and
  that is kept in either run's final text, located in the check run's sentence languages. The
  change stays if (1) at least 90% of those regions lie in sentences the repairs may not change
  (read as foreign, or undetermined in a mostly foreign abstract), so the change does only what
  it is for; (2) run 7's split decisions equal run 8's; and (3) run 7c's joins and run 8's
  lower-case splits differ from run 8's by under 1% (words decided differently, over run 8's
  count). Otherwise revert to run 8's pipeline: the late language stage, no guard, the
  kept-joined list as it is.
- *Amended 2026-11-16 before the results (the author):* this test replaces a blind comparison of
  40 samples by the author, which would have cost the author's time to measure what the test
  measures directly.
- **Reported, not decisive:** the decision tables and drop verdicts against run 8, the size of
  the population, and the abstracts whose foreign text cleaning now removes.
