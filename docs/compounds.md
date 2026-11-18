# The compound map (agenda 3.4; `docs/preprocessing_phase.md`, step 3)

Pre-registered 2026-11-18, before any pair of the sample or the census was judged.

## What the map is

A two-part compound is written open (*data base*), hyphenated (*data-base*) or closed
(*database*), and the share of each way drifts across decades (the Shorter OED dropped some 16,000
hyphens in 2007; Sun and Baayen 2021 find hyphenated compounds persist). Unmerged, one compound's
counts and vectors are split between its forms (Nenadic et al. 2004). The author's decisions
(2026-11-18):

- **The three forms become one,** the family's most frequent form over kept works for the whole
  corpus. A word-table lemma keeps its own spelling.
- **A spaced pair counts and merges only as a noun-noun compound:** both tokens tagged NOUN or
  PROPN, adjacent in the document and in one sentence
  (`corpus/compound_spelling.is_noun_compound`). So *is a* never merges, and neither does an
  adjective-first pair (*long term*), whose hyphen before a noun is grammar, not spelling.
- **Targets are guarded.** No spaced form containing a word-table lemma merges, and no lemma is
  written away. Splits may add target tokens, reported below.
- **One entry per spaCy token.** A merged pair writes its standard then `""`, a form whose standard
  is open writes it with its space, and a hyphen and closed pair rewrites the entry. Accented tokens
  are left alone. Lines read the entries as words (`corpus/cell_versions.segment_words`).

## The proposal (`scripts/evidence/compound_table.py`, code `0f9bd349`)

- **Counting.** Over kept works, with the lemma exceptions and the spelling map applied as in the
  static view, the builder counted every two-letter-part compound's forms (2 min 14 s on bazzite).
  Removed ranges are skipped; a merged pair cut by a removal ("straddle") was never found.
- **Families left out, with their reasons:**
  - letter codes: 56 (fewer than 4 letters: *a-b*, *i-e*);
  - Roman-numeral segments: 20 (*dsm-iv*, *wais-iii*);
  - vowelless segments: 117 (*bi-lstm*, *hs-crp*, *mm hg*);
  - one-letter second segments: 52 (*gaba-a*, *cbt-i*, OCR splits such as *group s*).
- **Spaced forms left out:**
  - 1,373 whose share passing the gate is under 0.5 (mostly syntax; the largest pairs only the
    ungated count finds are *for a*, *may be*, *a new*);
  - 28 whose split differs from the hyphenated form's.
- **Left as they are:** three-part tokens (61,492 tokens: *state-of-the-art*, *face-to-face*),
  hyphenated tokens with digits or other characters (193,501 tokens), and forms under the floor.
  Every pair needs both forms at 20 or more uses.
- **The proposal:** 2,001 pairs over 162,118 tokens (0.16% of the view's 98.7 million):

  | Kind | Pairs | Tokens | Examples |
  |---|---|---|---|
  | rejoin (one token to one token) | 1,207 | 100,106 | *on-line* → *online*, *posttest* → *post-test* |
  | merge (two tokens to one) | 320 | 34,544 | *health care* → *healthcare*, *data set* → *dataset* |
  | split (one token to two) | 474 | 27,468 | *information-processing* → *information processing* |

  Six standards are word-table lemmas (*feed-back* → *feedback*, *net-work* → *network*).
- **Targets gaining tokens from splits:** *information* 670, *learning* 460, *user* 367,
  *interface* 316, *image* 213, *attention* 167, *system* 152, *model* 96, and 10 more under 75.

## The audit, set before reading

- **What is judged.** Whether a pair's two forms are one word or compound written open,
  hyphenated or closed, at the level of the word, as for the spellings (2026-11-17).
  - **right:** one compound written another way, even when the sentences show different senses;
    a stray line-break hyphen (*meth-od*, *in-formation*).
  - **wrong:**
    - the way of writing marks a different word (*re-creation* against *recreation*,
      *re-present* against *represent*);
    - the open form is a free combination rather than the compound;
    - either form is a letter code, an abbreviation, or a chemical or gene name;
    - either form is not English.

  The direction is the rule above and is not judged. Names and other languages are judged as the
  English word the pair stands for.
- **The sample.** 150 pairs, seed 20261123: 75 drawn with probability proportional to tokens and
  75 uniform over the other pairs. By kind: 90 rejoin, 28 merge, 32 split. The map passes if at
  most 3 are wrong. Passing chances are:
  - 94% at a true error rate of 1%;
  - 65% at 2%;
  - 5% at 5%.
- **The census, fixed one by one.** Its errors do not count against the bar. It covers 265 pairs,
  34 of them also in the sample:
  - every pair touching a word-table lemma (68);
  - the 50 with most tokens (49 after overlaps);
  - every *re-* pair (64), since *re-* can mark a different word;
  - every one-letter first segment (49), where codes and OCR splits gather (*m age*, *t hat*);
  - *co-*, *de-*, *un-*, *pre-* and *pro-* pairs whose hyphenated form has at least 100 uses (35).
- **The judges.**
  - Two Claude Sonnet judges and a Claude Opus adjudicator for their differences, with the prompt
    saved in `curated/judging/compounds/prompt.md`.
  - Each pair is shown with up to two corpus sentences for each form, from kept works, the
    occurrence marked; spaced forms only from occurrences that pass the gate.
  - The author judges 30 pairs drawn from the sample (seed 20261124), blind. The agents' verdicts
    stand if they agree with the author on at least 27 of the 30, a "not sure" counting as a
    disagreement; otherwise the author judges the 150.
  - The items, which quote corpus sentences, stay out of the public repository. The key
    (`data/review/compounds/key.json`) and the verdicts are kept in `data/review/compounds/`.
- **If the sample fails,** the agents review every pair of the failing kind (rejoin, merge or
  split), its errors are fixed, and a new sample (seed 20261125) is judged once against the same
  bar.
- **Then.**
  - `curated/compounds.tsv` is written from the proposal minus the pairs judged wrong. A pair
    judged wrong takes along its pairs of the same kind whose form only adds *-s*, *-d*, *-ed* or
    *-ing*.
  - The invariant tests run on the map: no word-table lemma written away.
  - Each merged family's share of its forms per decade is reported, overall and per field.

## Results

To be written after judging.
