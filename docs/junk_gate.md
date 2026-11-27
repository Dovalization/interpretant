# The junk gate (phase step 7; decision agenda 4b)

*The pre-registration, committed on 2026-11-27, before step 6's locked `build_usages` run and
before any record of the gate's population is judged. The text above "Results" is not edited
after that; a change made before the data is read goes in as a dated amendment, in its own commit.*

## What the gate asks

How much of each target word's drawn usages, per decade, comes from text that is not the work's
own prose (junk)? The main measure, J, reads the corpus only through drawn usages, so junk reaches
it only there. Junk that differs between the periods of a comparison can look like semantic change.

- **Why per decade.** A 2026 study of OpenAlex abstracts (Kim, Holst and Ginis) rejects about 21%
  of abstracts before 2000, about 12% for 2000–2019 and about 6% for 2020–2025.
- **Why per word.** Matched controls absorb junk that a whole decade shares. They can't absorb one
  word's own excess.
- **How large an excess matters.** J is half the squared distance between the two periods' mean
  vectors (agenda 6.3). If the periods' junk shares differ by d and nothing else differs, J equals
  d² times the J between junk and real uses: 1% of it at 10 points, 4% at 20 points.

## What was read before this was written

- **Counts on the working usages table** (the table before the locked run), with no word printed:
  99 words, 788 word-by-decade cells, 1,055,201 drawn usages from 445,687 records. The median cell
  has 471 drawn usages; 613 cells have 100 or more and 71 have under 30.
- **Non-LLM signals there:** records with no finite verb are 0.55% of those records;
  repository-only records are 8–15% of drawn usages per decade; list-like sentences about 1%.
- **Run 3's 200 record verdicts** (`data/review/run3/verdicts.jsonl`) and the earlier reviews'
  figures (agenda 4b): 30 non-abstracts and 3 unsure of 200 in run 3; 16 failures of 200 in the
  provisional check after the cleaning batch.
- **Not read:** any junk label on the gate's population, and any drift.

## The decisions behind the design (the author, 2026-11-27)

1. Checks that need J are fixed here and run at the start of inference, before any target's drift
   is read. Nothing encodes the usages table yet, and the phase ends before encoding.
2. A trained classifier scores every record. Judges are not used at scale: judging 32,000 to 44,000
   records was priced and set aside.
3. The Claude judges label through the API (`judge-api`).
4. A failing cell keeps its place in the comparisons; its junk is left out.
5. The judges are calibrated on run 3's 200 records, on the author's second look at the disputed
   ones, and on 30 fresh records the author judges blind.
6. Local models are trialled as judges, as a comparison. The trial changes no verdict of the gate.

## Definitions

- **A record** is a kept work's abstract. Its text here is the text outside removed ranges
  (`corpus/removals.kept_text`), which is what the measures read.
- **A drawn usage** is the one usage drawn per work and word (`usages/usage_selection.py`).
- **A cell** is a word and a decade, over all fields.
- **The codebook** is `curated/judging/junk_gate/prompt.md`. It follows agenda 4b's failure
  definition: junk is text that is not the work's own prose. Article openings, book reviews,
  editorials and comments count as abstracts. Its categories draw on Kim, Holst and Ginis's.
- **An answer** has a note, a verdict (`abstract`, `mixed` or `junk`), the kind of junk, whether
  the marked sentence is the work's own prose (`own`, `other`, `none`), and whether it is damaged.
- **A junk usage** is a drawn usage whose record's verdict is `junk` or whose sentence is `other`.
  The measure reads one sentence per usage, so a real abstract's stray citation line counts.
- **A junk record** is a record whose verdict is `junk`. The classifier learns this.

## The items

- **The text shown** is the record's kept text. Up to 6,000 characters are shown whole. A longer
  text shows its first 3,500 and last 1,500 characters, and the marked sentence with 300
  characters on each side if it falls between; `[… cut …]` marks each gap.
- **The marked sentence** is the drawn usage's sentence, between ⟦ and ⟧. No word is shown, and
  the item carries no field, decade, source or year.
- **Order.** Items are shuffled with seed 20261405 and numbered in that order.
- Items quote corpus text. They stay where they are drawn, out of git; the key holds work ids and,
  for words, `Word.reported_name`.

## The sets

Each draw takes the smallest values of `sha256(seed:work:word:content_key)`
(`usages/identifiers.build_draw_key`), so it never depends on row order or on other records.

- **L, the labelled sample: 4,000 usages.** In each decade, the 500 drawn usages with the smallest
  keys under seed 20261401, over all words and fields. L is drawn by usage, so its junk share
  estimates the share of the decade's drawn usages that are junk.
- **E, the enrichment set: 600 records,** seed 20261402, from records likely to be junk. A record
  already in L is skipped.
  - 200 whose text has no finite verb outside removed ranges (the static view's `finite_verbs`);
  - 200 with a drawn usage in a list-like sentence (`usages/traits.py`);
  - 200 flagged `repository_only` whose kept text is under 500 characters.

  E is used to train the classifier and for nothing else.
- **Run 3: 200 records,** with the text they had in run 3. The author judged them on 2026-11-14,
  with the removed ranges highlighted and the field and decade shown. Cleaning has changed some
  texts since, so the judges see run 3's own kept text. No sentence is marked.
- **The confirmation samples: 40 usages per flagged cell,** seed 20261404 (below).

## The Claude judges

- **Who.** Two `claude-sonnet-5-5` judges label every item on their own. A `claude-opus-5-5`
  adjudicator settles the items where one says `junk` or sentence `other` and the other does not.
  On other differences the first judge's answer stands.
- **How.** `judge-api`: the Message Batches API at high effort with adaptive thinking, one item
  per request, the codebook as the whole system prompt, the answer constrained to its schema.
- **The budget.** At most 60 US dollars in all. If the confirmation samples would pass it, the
  judging stops and the author decides.

## The calibration, in three steps

**Step 1: run 3.** The judges label run 3's 200 records. Against the author's verdicts, with r018
counted as junk (the author's decision of 2026-11-15) and the 3 unsure records left out, all three
bars must hold:

| Bar | Passes when the judges are good | Passes when they are not |
|---|---|---|
| At least 178 of 197 agree | 99.8% of the time at a true 95% | 1.8% at 85% |
| At least 27 of the author's 31 junk are found | 98.2% at 95% found | 8.3% at 75% found |
| At most 6 of the 166 abstracts are called junk | 95.0% at a 2% false-junk rate | 5.0% at 7% |

Agreement alone can't see missed junk: judges that find half of it and raise no false alarm agree
on 92%. `mixed` counts as an abstract here, since run 3 was judged by record.

**Step 2: the second look.** The author looks again at the records where the judges' final verdict
differs from theirs, on a page that shows the text and no verdict. The answers are saved
(`data/review/junk_gate/run3_second_look.json`). They stand in for the corrections of 2026-11-16,
which were never saved by record. The bars of step 1 use the first verdicts; the second look gives
the reference for the judges' error rates below.

**Step 3: the fresh 30.** After L and E are judged, the author judges 30 of their records blind,
drawn with seed 20261403: 12 of L's that the judges call junk, 8 of L's that they call abstracts,
and 10 of E's that they call abstracts (where missed junk would gather). The judges are accepted
at 27 or more of 30. Judges agreeing 95% of the time reach that in 94% of draws, at 80% in 12%.

- **What each step can show.** Run 3 shaped the codebook, so it is a development set, and the
  fresh 30 are the only blind test. At 30 they are a weak test of missed junk; the second bar of
  step 1 carries that evidence.
- **If a bar fails.** The codebook is revised once and the step is repeated. For step 1 that is
  before L and E are judged. For step 3 a new 30 is drawn (seed 20261409), and every label made
  under the old codebook is judged again. A second failure goes to the author.

## The decade rule

It reads L alone, with no classifier.

- **The bar.** A decade passes with at most 52 junk usages in its 500. It passes 97.7% of the time
  at 8% junk and 65% at 10%, and fails 95.5% of the time at 13%. Eight decades at 8% all pass 83%
  of the time.
- **The comparison.** Each decade's junk count is tested against the other seven decades together
  (Fisher's exact test, one-sided, at 0.05/8). This is the pattern the gate exists to catch.
- **If any decade fails either.** Every drawn usage from a record the classifier calls junk goes
  into the exclusions table, in every decade, and the measures are reported with and without it.
  The exclusion covers all decades so that both sides of a comparison are treated alike.
- **Also reported:** each decade's share with its Wilson interval, and the share corrected for the
  judges' error (Rogan and Gladen 1978) from their sensitivity and specificity against the
  author's second-look verdicts (posterior draws with seed 20261407, as in the usages audit). The
  correction is a check, not part of the bar.

## The classifier

- **The vectors.** Each record's kept text is encoded by `BAAI/bge-base-en-v1.5` at revision
  `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`: the first 512 tokens, the vector of the first token,
  normalised to length 1 (768 numbers). The model is not trained further.
- **The surface features.** The encoder lowercases its input, and capitals are the main sign of
  lists and titles. So 13 features of the kept text are added, each standardised on the training
  records:
  1. the logarithm of its length in characters;
  2. the logarithm of its number of words;
  3. the logarithm of its number of sentences;
  4. the share of its words that start with a capital;
  5. the share of its words written all in capitals;
  6. the share of its characters that are digits;
  7. the share of its characters that are punctuation;
  8. its list separators (`.-`) per 100 words;
  9. whether it has a finite verb outside removed ranges;
  10. the share of its lower-case words that are unknown (the OCR score below, on the whole text);
  11. the share of the record's characters that cleaning removed;
  12. whether it ends with a full stop, question mark or exclamation mark;
  13. the mean number of words per sentence.
- **Never a feature:** the decade, year, field, source or word.
- **The model.** Logistic regression (scikit-learn's `LogisticRegression`, C = 1, balanced class
  weights, the default solver), trained on the junk records of L and E against the rest.
  Nothing is tuned.
- **Cross-fitting.** L's records are split in 5 folds by the hash of the work's source (its
  ISSN-L, or its source name, or the work itself when it has neither), so one journal's template
  never sits on both sides. Each fold is scored by a model trained on the other four and on E.
- **The cut-off.** Inside each training set, an inner cross-fitting of the same kind scores its L
  records, and the cut-off is the lowest score that still clears 97% of the abstracts there. It is
  applied to the held-out fold. The final model is trained on all of L and E, with its cut-off set
  the same way.
- **The quality bar.** On L's held-out verdicts, against the judges' junk records: sensitivity of
  at least 0.70 and specificity of at least 0.97, as point estimates. With about 200 junk records
  the sensitivity's interval is about 7 points either side.
- **If the bar fails.** The decision model's junk probability from the local trial takes the
  classifier's place if, with its cut-off set by the same rule, it meets the bar. Otherwise the
  author decides.
- **Reported:** the false-alarm rate by decade and by field, and for each decade the mean
  difference on L between the judges' label and the classifier's, with its interval. That
  difference is the classifier's bias in the decade (the correction term of prediction-powered
  inference, Angelopoulos et al. 2023).

## The cell rule: the classifier screens, the judges confirm

- **Why the classifier does not decide alone.** Correcting a cell's share for the classifier's
  error assumes it errs equally in every cell. It does not: the kinds of junk differ, and early or
  humanities abstracts look more like junk. A clean cell where the classifier's false-alarm rate
  is 10% instead of 3% would fail about 4 times in 10.
- **The screen.** For each cell with at least 100 drawn usages, take x, the number of its drawn
  usages from records the classifier calls junk, and q, that share over the whole decade. The cell
  is flagged when a count of x or more has a probability under 0.01 in a binomial draw of the
  cell's size at rate q.
  - With a classifier at the quality bar and a decade at 6% junk, a cell at 25% junk is flagged
    91% of the time at 100 usages and almost always from 200; a cell at 20%, 69% and 97%.
- **The confirmation.** For each flagged cell the Claude judges label its 40 drawn usages with the
  smallest keys under seed 20261404; a usage already labelled keeps its label. The cell fails at 9
  or more junk usages.
  - A cell fails 0.4% of the time at 8% junk, 7% at 13%, 41% at 20%, 70% at 25% and 89% at 30%.
- **The remedy.** When a cell fails, every drawn usage of that word from a record the classifier
  calls junk goes into the exclusions table, in all the word's decades. The cell stays in its
  comparisons, and each comparison of the word is reported with and without the exclusion.
- **Why the whole word.** At the quality bar about a third of what the classifier removes is real
  abstracts, of kinds that look like junk. Cleaning one cell and not the one it is compared with
  could itself look like change.
- **Cells under 100 drawn usages** are reported with their classifier share and its interval, not
  gated. Step 8's top-ups add to them, and the gate runs again then.
- **The exclusions table** holds a work and a word per row. The measures leave those usages out.
  The usages table of step 6 is not changed.

## OCR damage and the flags

No OCR correction was made before this step (the author, 2026-11-18). The gate scores damage and
leaves the question of its effect to the checks below.

- **The score.** For each drawn usage's sentence: the share of its lower-case words of 3 or more
  letters, the target left out, that are unknown. A word is known if
  `corpus/dictionary.is_dictionary_word` knows it, or if it appears in the kept text of at least 20
  kept works, so that technical terms do not count as damage. A sentence with fewer than 5 such
  words has no score. A dictionary share is the usual estimate of OCR quality without a reference
  text (van Strien et al. 2020).
- **Its check.** On L, the score must separate the sentences the judges marked `damaged` from the
  rest, with an area under the ROC curve of at least 0.75. If it does not, it is revised once,
  before any drift is read. If it still does not, the OCR check below is reported as not testable.
- **OCR-heavy usages** are those at or above the score that best separates the two groups on L
  (the largest sensitivity plus specificity).
- **The flag report,** from the locked table, per decade and per cell: the exact shares of
  capitalised targets, mentions, list-like sentences, modifier and compound roles, and sentences
  under 10 words.

## The checks left for inference

They need J, so they run when the usages are encoded, before any target's drift is read. Their
results are ratios over all words, with held-out words under their sealed names.

- **The subsets:** capitalised targets; modifier and compound roles; list-like sentences; mentions;
  sentences under 10 words; OCR-heavy usages; usages from records the classifier calls junk.
- **The test,** in the form of rounds S1 and S1b (`docs/predictions_cleaning.md`). For each
  comparison of `curated/comparisons.tsv` where the subset holds at least 10 usages of a cell:
  - D(subset) is the absolute change in J when the subset is removed;
  - D(random) is the mean absolute change over 100 draws (seed 20261406) that remove the same
    number of usages from each cell at random.

  The subset **moves J** when both hold: the mean D(subset) is at least 1.5 times the mean
  D(random), and it is at least half the mean bootstrap standard error of J (the document-level
  bootstrap). The second condition keeps a ratio of two tiny numbers from passing.
- **What each outcome does:**
  - **capitalised targets** move J: the name classifier of `docs/usages_audit.md` is trained on
    about 1,000 judged capitalised uses. Otherwise the names stay as they are;
  - **roles:** the measures are reported with and without modifiers and compound parts either way,
    as step 6 requires;
  - **list-like sentences, mentions, short sentences:** a subset that moves J is left out of the
    main measure, and both versions are reported;
  - **OCR-heavy usages** move J: OCR correction is considered (the author). Otherwise there is
    none;
  - **predicted junk** moves J: the usages of every record the classifier calls junk are left out
    of the main measure, in every cell.

## The local trial

A comparison, set before any local model has judged a record. It changes no verdict of the gate.

- **The question.** Can models that run on our own machines do the judges' work, and at what cost
  in time? Rao and Callison-Burch (2026) found that decision models can often stand in for LLM
  judges, and that both tend to err on the same items.
- **The job.** `judge-ollama` sends the same items and the same codebook to an Ollama server.
  - **Chat models** answer through `/api/chat` with the answer's schema, thinking off, temperature
    0 and seed 20261408.
  - **Decision models** answer through `/v1/systemone`, which returns probabilities and no text.
    They get three questions about the item: whether the text is the work's own prose (yes or no,
    with the codebook's definitions as criteria), the kind of junk (a choice), and whether the
    marked sentence is the work's own prose (yes or no). The record is junk at a probability under
    0.5 for the first.
- **The setups.** A setup is a model, its precision and its machine.
  - On the runner's graphics card (12 GB): `qwen3.5:9b` at 4 bits and at 8 bits, `gemma4:12b`,
    `bonsai-2:27b` and the decision model `pplx-decider-flash:4b`.
  - On the runner, partly in system memory: `gemma4:26b` and the decision model `d3:9b`.
  - On a second machine with 24 GB of shared memory, serving the runner over the private network:
    `gemma4:26b` and `qwen3.8:27b`. Records pass through its memory only; none is stored there.
- **Stage 1: accuracy against speed, on run 3.** Every setup labels run 3's 200 records. For each,
  the run records its agreement with the author on the 197, how many of the 31 junk it finds, its
  records per minute, and the share of the model on the graphics card.
- **Combinations,** scored from stage 1's saved answers with no further runs:
  - **a panel:** two setups label every record and a third settles the records where they differ
    on junk. Every choice of three is scored;
  - **a cascade:** a decision model labels every record, and those with a probability between 0.2
    and 0.8, or between 0.1 and 0.9, go to a chat model. Every pairing and both bands are scored;
  - **a jury:** the majority of any three setups.

  A combination's time comes from its parts' measured speeds and the share of records each
  handles: parts on different machines run together, parts on one machine in turn.
- **Stage 2: the full sets.** A setup or combination goes on to L and E when no other is both more
  accurate and faster on run 3, and when its estimated time for them is under 12 hours. Accuracy
  is the agreement on the 197, then the junk found. Stage 2 records the real times.
- **Reported for stage 2:** agreement, sensitivity and specificity against the Claude judges'
  final verdicts on L; agreement with the author on the fresh 30; the decision model's probability
  against the classifier's score on L; and the gate's decade and cell verdicts when the classifier
  is trained on a local setup's labels.
- **Two cautions.** Many combinations are tried on 197 records, so the best one looks better than
  it is: the choice is made on run 3 and its accuracy is read on L. And the fresh 30 were chosen
  by the Claude judges' verdicts, so they test a local setup less fairly than run 3 does.
- **Fit for labelling.** A local setup or combination is fit to label records in this project if
  it meets the three bars of step 1 and reaches a sensitivity of 0.80 and a specificity of 0.97
  against the Claude judges on L. What it is then used for is the author's decision.

## Storage

- `curated/judging/junk_gate/prompt.md`: the codebook.
- `data/review/junk_gate/`: the key, the counts, the author's answers, the labels and the results.
  A label is an item's verdict, kind, sentence and damage, with no note. Notes can quote the
  corpus, so they stay where they were written.
- `data/processed/junk_gate/`: each record's score and verdict and each usage's OCR score, the
  exclusions table, and the record vectors, written by DVC stages with a `junk_gate` section in
  `params.yaml`.
- Held-out words appear only under their sealed names.

## After a top-up (step 8)

- The classifier, unchanged, scores the new records. The screen and the confirmation run on every
  cell the top-up changes.
- The Claude judges label 300 drawn usages of the downloaded records (seed 20261410). If the
  classifier's sensitivity or specificity there falls under the quality bar, the author decides
  before the new cells are gated.
- The decade rule is not rerun: downloads are drawn by word, not by decade.

## What this design cannot do

- **Junk the classifier does not recognise is not flagged.** The screen sends the judges where the
  classifier sees an excess. A cell full of a kind of junk it misses passes. The checks left for
  inference are the second line.
- **Run 3 shaped the codebook,** and the fresh 30 are a small blind test.
- **Text that is the work's own but not the work's time** (a modern summary attached to an old
  work) reads as an abstract here.

## Sources

- Angelopoulos, Bates, Fannjiang, Jordan and Zrnic 2023, "Prediction-powered inference", *Science*
  382:669–674.
- Kim, Holst and Ginis 2026, "One in Eight OpenAlex Abstracts Has Integrity Issues",
  arXiv:2605.20168.
- Rao and Callison-Burch 2026, "Decision Models vs. LLMs as Rubric Judges: Cheaper, Faster, and
  Wrong in the Same Places", arXiv:2609.29769.
- Rhoda, Fernandez, Fitch and Lemeshow 2010, "LQAS: User Beware", *International Journal of
  Epidemiology* 39:60–68 (on which conclusion a small-sample rule should need strong evidence
  for).
- Rogan and Gladen 1978, "Estimating prevalence from the results of a screening test", *American
  Journal of Epidemiology* 107:71–76.
- van Strien et al. 2020, "Assessing the impact of OCR quality on downstream NLP tasks", ICAART.

## Amendment, 2026-11-27, before L and E are judged

- **What run 3 showed.** Under the codebook as first committed, the judges passed all three bars
  of step 1: 192 of 197 agreed with the author, 28 of the 31 junk records were found, and 2 of the
  166 abstracts were called junk. The author's second look at the 5 disputed records
  (`data/review/junk_gate/run3_second_look.json`) called all five abstracts: three the author had
  called junk on 2026-11-14, where the judges were right, and the two the judges called junk.
  Against the second look the judges found all 28 junk records and called 2 of 169 abstracts junk.
- **The pattern.** Both false junk records are a passage of an older philosophical or theological
  text standing as the record's text. The codebook listed "a quotation standing alone" as junk
  (`other_work`), and both judges followed it. The author called both abstracts, on 2026-11-14 and
  again on the second look: such a work is usually an edition, a translation or a study that opens
  with the text, so the text is the work's own, in line with the rule that an article's opening
  counts as an abstract (agenda 4b).
- **The change** (the author, 2026-11-27). The codebook now says that a passage of an older or
  primary text standing as the record's text is the work's own text, and junk only when the record
  plainly belongs to something else. `other_work` keeps a quotation used as an epigraph, a
  chronology and an interviewee's biography.
- **What follows,** as "If a bar fails" requires for a revised codebook. Run 3 is judged again
  under the amended codebook and step 1's three bars are checked again on it; the second look
  stays the reference for the judges' error rates. L and E have not been judged, so no other label
  is affected. The first judging's verdicts are kept on the runner, apart from the new ones.

## Results

*To be appended.*
