# The contextual sentence view (agenda 2.2; `docs/preprocessing_phase.md`, step 5)

Pre-registered 2026-11-18, before the view was built or any sentence judged.

## What the view is

The contextual pipeline draws its usages from sentences, so it needs one table of every kept
work's sentences, with what each may be used for. The sentence boundaries are the spaCy pass's
(`en_core_web_sm` with the project's abbreviation special cases), already stored in the abstracts
table with each sentence's language. Nothing is split again.

- **One row per sentence of a kept work** (`corpus/sentence_view.py`): work id, sentence number,
  start, end, language, confidence, word count and exclusion. The word count is the number of
  whitespace-separated pieces holding a letter.
- **`removed`:** every non-space character of the sentence lies inside the work's removed ranges,
  merged. A sentence can span two adjacent ranges, which is why the ranges are merged and spaces
  between them are ignored.
- **Partly covered sentences stay whole,** so offsets never move. A mention inside a removed range
  is never a usage; that is step 6's matching.
- **`cut_off`:** the last sentence of a kept work flagged `truncated`, whose text ends without
  sentence punctuation. Its words are real uses up to the cut, so it stays in the view and in
  step 6's usages table, but no usage is drawn from it.
- **Kept, with their values recorded:** sentences of undetermined language (`und`) and the
  sentences of works flagged `malformed`.
- **Expected, from the phase plan's scout:** about 125,000 removed sentences (2.9%), about 65,000
  cut-off sentences, about 37,000 of undetermined language and about 1,600 malformed works. The
  build reports the exact counts.

## The audit, set before reading

### The judges and their calibration

- Two Claude Sonnet judges and a Claude Opus adjudicator for their differences, with the prompt
  saved in `curated/judging/sentences/prompt.md`.
- **Calibration** uses the 50 abstracts whose sentence breaks the author judged in run 3's review
  (`data/review/run3/verdicts.jsonl`, items `b000` to `b049`).
  - The items are rebuilt from run 3's table (`data/processed/abstracts_run3`), exactly as the
    author saw them: each sentence's text, in order.
  - Each judge marks wrong cuts (by the number of the sentence after the cut) and missed breaks
    (by the sentence and the word that should start the next one).
  - An abstract agrees when the agents' final marks give the same wrong cuts as the author's and
    the same sentences holding a missed break.
  - The agents' verdicts stand if they agree with the author on at least 45 of the 50 abstracts.
    Otherwise the author judges the two samples below.
- The items quote corpus text, so they stay out of the public repository. The keys and the
  verdicts go in `data/review/sentence_view/`.

### Cut precision

- **The sample:** 50 kept abstracts with at least one cut, 6 per decade, then the 2 smallest keys
  among the rest. Abstracts are drawn by the smallest sha256 of (seed 20261127, work id).
- Every cut is judged; missed breaks are marked too and reported.
- **Bar:** wrong cuts are at most 2% of the sample's cuts (precision at least 98%).

### Boundaries of target sentences

- **The sample:** 150 eligible sentences that hold a single-word target outside removed ranges, by
  the static view's entries.
  - Eligible means no exclusion.
  - 18 per decade, then the 6 smallest keys among the rest. Sentences are drawn by the smallest
    sha256 of (seed 20261128, work id, sentence number).
- **Each is judged with the sentences before and after it, as cut.** It is right when both its
  cuts fall where a sentence ends and it holds one sentence. Otherwise it is wrong: a wrong start,
  a wrong end, or two sentences.
- **Bar:** at most 5 wrong. At a true rate of 2% this passes 92% of the time; at 7%, 4%.

### If a bar fails

The failing cases are compared with pySBD or scispaCy, the splitters built for scientific text
(Sadvilkar and Neumann 2020; Neumann et al. 2019). Adding either is a dependency decision for the
author. The chosen splitter applies to the contextual view only, never to the cached documents.

### The calibration failed; the author judges both samples (2026-11-18)

- **Result.** The agents agree with the author on 40 of the 50 run-3 abstracts, so the calibration
  fails (`scripts/review/score_sentences.py`, code `854691ea`; verdicts in
  `data/review/sentence_view/verdicts/`).
  - In 5 of the 10 disagreements the author had marked a break after a heading glued to the first
    words of a sentence (*ABSTRACT ‖ Meta-analyses*, *Methods: ‖ A hospital*). The run-3 page, and
    the judges' prompt copied from it, said to leave headings however they were cut.
  - The two Sonnet judges' answers are nearly identical, reasons included (146 of 150 sentence
    verdicts word for word), so with one prompt and one model they act as one judge.
- **The author's decisions.**
  - The author judges both samples on a page, the agents' verdicts hidden, as the rule above says.
  - A heading glued to the start of a sentence is not a boundary error: it adds a word or two of
    context, and the sentence and its words stay right. Only a cut inside a sentence, or two
    sentences in one, counts.
- **Set before the author judges.**
  - A target sentence the author marks "not sure" counts as wrong.
  - An abstract's marks count as given, with "not sure" reported.
  - The bars are unchanged: at most 2% wrong cuts, at most 5 wrong target sentences.
- **Curtailed inspection (the author asked for a shorter review, 2026-11-18).** The precision
  sample holds 314 cuts, so 7 wrong cuts fail the bar, and judging more abstracts can only add
  errors.
  - The agents marked 9 wrong cuts, all in 4 abstracts (A060, A070, A071, A086). In the
    calibration they marked fewer errors than the author, not more.
  - The author first judges those 4 abstracts, blind to the agents' marks.
  - If the author's wrong cuts there reach 7, the precision bar fails and judging stops. Step 5
    then moves to the comparison with pySBD or scispaCy, and the boundary sample is judged once,
    on the splitter kept.
  - Otherwise the author judges the other 46 abstracts and the 150 target sentences.

### The merge rules and the fresh audit (pre-registered 2026-11-18, before the rules were applied)

**How the rules were found.** `scripts/evidence/sentence_cuts.py` scanned all 3,708,932 cuts
between consecutive sentences of kept works, in two passes (`c8b27460` broad patterns, `236bafce`
narrowed rules), with 40 seeded examples of each pattern. Every pattern is 5 to 12 times more
common per cut in the 1950s than in the 2020s, so wrong cuts gather where OCR damage does. A
pattern became a rule only if its examples were nearly all wrong cuts.

**The rules** (`corpus/sentence_view.py`) join two consecutive sentences of a kept work, neither
removed, when:
1. **closing punctuation:** the second starts with *, ; : ) ] }* (1,223 cuts in the scan);
2. **a period:** the second starts with *.* (13,029);
3. **spaced dots:** the first ends with spaced dots (*. .*) and the second starts lowercase (part
   of 4,857);
4. **a list number:** the first ends with a list number after a colon or semicolon, not after a
   digit (*were: 1.*, *;(2)*; 4,633);
5. **a function word:** the first ends with a word such as *the*, *of*, *and* or *a*, with no
   punctuation after it (106);
6. **inside brackets:** the first leaves a bracket open and the second closes one it never opened
   (2,501);
7. **an abbreviation that never ends a sentence:** *ed., eds., trans., tr., cf., e.g., i.e.,
   viz., vs., esp., Univ., Dept., Dr., Prof., Mr., Mrs.* in their usual case;
8. **a reference abbreviation and a number:** *p., pp., vol., vols., ch., chap., no., nos., fig.,
   figs., eq., ref., refs., sec.* followed by a digit, a Roman numeral or a bracket, and *St.*
   followed by a capital;
9. **no letters:** a row without letters joins the row before it (the first row joins the next).

Left out because their examples also caught real sentence ends: *etc.*, *et al.*, *Jr.*, *Inc.*,
*Co.*, *sec* as seconds, all-capital acronyms (*CA.*, *NO.*, *IE.*), and a second sentence that
merely starts lowercase. The rules catch 6 of the author's 7 wrong cuts; the seventh, a stray OCR
period before a capital, looks like a sentence end to any rule.

**Amended the same day, after the first build and before any audit item was drawn.** The first
build joined 13,411 cuts but only 4 of the author's 7 wrong cuts. Two were next to a stray *.* or
*2* that the template rule had removed (a lone mark is common across abstracts), and removed
sentences were never joined. So:
- a removed sentence with no letters joins like any row without letters (a removed sentence with
  letters, such as a citation line, still never joins);
- a row without letters that holds a digit, before a row starting with *.* or *)*, joins that
  next row instead of the one before it: it is a list number cut from its item (*2* | *. Subjects
  using*).

A joined sentence takes the language and confidence of its longest part. The view gains a column,
`merge_rules`, listing the rule of each join, and its sentence numbers count joined sentences.

**The fresh audit.**
- **The judges' prompt** now says that a heading glued to a sentence is not an error, and covers
  a third kind of item, a joined cut.
- **Recalibration:** 30 abstracts drawn fresh from the new view (seed 20261201; 3 per decade,
  then the 6 smallest keys among the rest), the author judging blind. The agents' verdicts stand if they agree with the author on at least 27, with
  agreement defined as before (same wrong cuts, same sentences holding a missed break). Otherwise
  the author judges the three samples below.
- **Cut precision:** 50 new abstracts (seed 20261202), drawn as before, apart from the
  recalibration set. Bar: at least 98%.
- **Joined cuts:** 80 joins drawn from kept works (seed 20261203), each shown with its two parts.
  Joins by the no-letters rule are left out of the draw (amended before drawing: they are 35,101
  of the 46,888 joins, and a row without letters cannot be a second sentence).
  A join is wrong when the two parts are two sentences. Bar: at most 2 wrong. At a true rate of
  1% this passes 95% of the time; at 7%, about 8%.
- **Boundaries:** 150 new target sentences (seed 20261204), drawn and judged as before. Bar: at
  most 5 wrong.
- The agents judge the recalibration abstracts mixed in with the precision sample, so they cannot
  tell them apart.

## Results

To be written after the build and the audit.
