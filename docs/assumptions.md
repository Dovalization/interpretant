# Assumptions audit

*Started 2026-11-10, while round X ran. Every instrument in this project has been tested against
predictions; the assumptions underneath the instruments mostly have not. This document states
them, says which results each could explain, and says how each could be tested or removed. Where
a consequence is a hypothesis rather than a measured fact, it says so.*

Each entry gives: the assumption; where it enters; whether the project ever stated it (**stated**
as a README limitation, or **unstated**); what it could explain; and a test or a removal.

## A1. A field is what OpenAlex's classifier says it is

**Where it enters.** Every cell. A work belongs to a field because its primary topic maps to that
field or subfield (`cell_filter` in `corpus/openalex.py`). OpenAlex assigns topics with a model
that reads a work's title, abstract, source name and citations, and maps each of about 4,500
topics to a subfield, field and domain through Scopus's ASJC categories (OpenAlex help pages).
One present-day taxonomy and one model are applied to every decade.

**Stated?** In part. The README says topic labels are noisy in small early cells and that a
label's scope can widen; it does not say that the labels are drawn partly from the text being
measured, or that they are applied anachronistically.

**What it could explain.**
- *Circularity.* Fields defined partly by their vocabulary will look linguistically distinct. A
  classifier tells every pair of fields apart (round S), and every field's language model scores
  its own text best (D0, round M); both are partly built in.
- *Anachronism.* A 1985 paper is filed by what a 2024 model thinks it resembles. A field's early
  cells may hold the papers that look like the field's present, not the community it was.
- *Widening labels.* The HCI cell spans 915 sources in the 1990s and 3,601 in the 2020s (round K):
  the classifier files papers that read like HCI as HCI, wherever they appeared.
- *Nesting and mislabels.* Parent fields contain subfields (shared abstracts in the first
  cross-perplexity run; most label-noise flags); some abstracts are plainly misfiled (engineering
  under cognitive psychology).

**Test or removal.** Define communities independently of their text: by venue, a fixed list of
journals and conferences that belong to a field in every decade compared (round K′'s
venue-stable cell, generalised). Rerun a robust finding and a known answer on venue-defined cells
and compare with the label-defined ones. Measure, per cell, how often the topic label and the
venue agree.

## A2. What changes in a cell is meaning, not who is writing

**Where it enters.** Every comparison between decades. A cell is treated as one community whose
use of a word may change; its membership is assumed to be stable enough not to matter.

**Stated?** Unstated, except for the HCI label's widening.

**What it could explain.** A change in how a word is used in a cell can come from four sources,
and every change measure here adds them together:
- meaning change, the quantity of interest;
- composition change: different subcommunities and venue types within the field;
- demographic change: who writes, from where, in what first language. *Hypothesis:* formulaic
  phrases such as "has attracted increasing attention" are typical of particular groups of
  writers whose share of AI abstracts grew; round C's stock-phrase sense of *attention* may be
  partly this. Not measured;
- coverage change: OpenAlex indexes more of recent literature than of older literature.

Round K′ showed one instance: inside venues HCI had in both decades, one model's gain on HCI was
half its whole-cell size. Round U showed that a pairwise measure is dominated by a word's variety
of uses; variety plausibly grows with a field's size and heterogeneity (*hypothesis*).

**Test or removal.** Measure composition per cell: the share of the top venues, venue diversity,
and the countries of authors' institutions (OpenAlex records them). Check whether a measured
change survives on samples matched for venue mix. Report composition shift beside any change.

## A3. Equal-sized random samples make cells comparable

**Where it enters.** The grid samples up to 10,000 abstracts per field and decade.

**Stated?** Stated as a trade-off ("equal cells trade volume for comparability").

**What it could explain.** A field that grew a hundredfold is represented by a much smaller and
more heterogeneous fraction of itself in later decades. Equal size equalises training budgets,
not what the sample stands for. Small early cells were the ones the held-out check flagged
(round M).

**Test or removal.** Report each cell's sampling fraction and field size. Where a result spans
decades, check it on samples stratified by venue (A1, A2).

## A4. Decades are the unit of change

**Where it enters.** Every cell is a decade; every change is measured between adjacent decades.

**Stated?** Stated as a limitation ("decade granularity is coarse"), never treated as a reason a
known answer could be unfindable.

**What it could explain.** A change dated 1986 (neural networks) or around 2010 (digital
platforms) sits inside a decade cell, not between two. Round 5 missed *network* at AI's
connectionist turn and *platform* in communication; round 1 missed *network* too (P2). Round X
tests both again.

**Test or removal.** Use years or three- to five-year windows around a documented date when
testing a known answer, and locate where the change falls with a simple count (A5) before
choosing the cells.

## A5. Known answers are present in the cells

**Where it enters.** Rounds 5, C, C′, C″ and X: an instrument is judged by whether it finds a
change documented in a field's history.

**Stated?** Unstated.

**What it could explain.** A change documented for a research community need not appear in an
OpenAlex label's cells at decade resolution (A1, A4). When an instrument misses a known answer,
either the instrument is blind or the change is absent from the data; the project has only ever
tested the first. For *attention* the presence was checked indirectly (marked mechanism uses are
about 55% of AI's 2020s occurrences and almost absent before); for *platform* and *network* it
was never checked.

**Test or removal.** A manipulation check before any known-answer round: count, with simple rules
a reader can verify (marker phrases, hand-read samples), how much of each cell carries the known
sense, and commit that count with the predictions. A known answer enters the round only if the
change is visibly present.

## A6. Abstracts carry a field's use of a word

**Where it enters.** The whole corpus (apart from the books layer).

**Stated?** In part ("the corpus is not the discourse").

**What it could explain.** Abstracts are a formulaic genre; stock phrases (*particular attention
is given*) form their own clusters of use (round C). Theoretical words (*intentionality,
embodiment*) do much of their work in arguments that abstracts compress. Abstracts are
reconstructed from OpenAlex's inverted index, and those before the 1970s are often fragments.

**Test or removal.** Scope the claims to the abstract register explicitly. For a few words,
compare with full texts from an open-access subset.

## A7. A word's meaning is the distribution of its contexts

**Where it enters.** Every instrument: neighbours in an embedding, bigram predictability,
contextual vectors.

**Stated?** Stated as the project's theory (Firth, Wittgenstein, Peirce's interpretant); its
measurement consequences are not.

**What it could explain.** Context-based measures register the topics a word occurs among, its
syntactic environment and its variety of uses as well as its sense (Kutuzov et al. 2022; round U,
where shuffling the periods left APD's agreement at 0.842 of 0.864). A word used in the same sense
amid new topics looks changed.

**Test or removal.** Net out variety (J); compare against control words on the same transition;
validate against human judgements of usage pairs (the author's gold set in item 8c).

## A8. Five resamples measure the uncertainty

**Where it enters.** The presence rule (same sign in five seeds, mean at least twice the standard
deviation) and most cross-perplexity and classifier rounds.

**Stated?** Stated as a method; what it covers is not.

**What it could explain.** Resampling within a cell captures sampling noise only: not label noise
(A1), composition (A2), or the choice of corpus. In small cells every seed draws the same
abstracts and the spread collapses (round C). Across 22 rounds and 118 verdicts there is no
correction for the number of tests; thresholds were set by judgement round by round.

**Test or removal.** Say what each uncertainty estimate covers. Resample by venue or author where
composition matters (a cluster bootstrap). Prefer fewer, decisive predictions with thresholds
justified in advance.

## A9. The researcher's choices of words are neutral

**Where it enters.** Target words, marker phrases, word sets for neighbour tests, and known
answers were chosen by the author.

**Stated?** Partly: control words have been drawn at random since round C.

**What it could explain.** Targets came from prior intuitions about which words would differ;
marker phrases decided round C″ (*prompt engineering* was both a marker and a sense); known
answers for round X were chosen knowing round 5 had missed two of them.

**Test or removal.** Draw from lists nobody on the project made (DWUG, Wiktionary's field labels,
random draws), validate marker phrases on a hand-labelled sample before they are used, and keep
the author's choices for open questions, not for calibration.

## A10. Smaller ones

- **English only.** OpenAlex's language tag and the corpus's English check; bilingual abstracts
  leak through (round C's crash). Scope, not bias, for most questions.
- **One occurrence per abstract** in the sense and pair rounds. Repeated uses within an abstract
  are dropped; harmless for most words.
- **The interpretant is measurable from text.** Peirce's interpretant includes habits of
  inference and action that no corpus records; the project measures the textual trace. A scope
  statement, not a testable assumption.

## Which results each assumption could explain

| Result | Assumptions that could explain it |
|---|---|
| Classifier separates every field; own model always best (S, D0) | A1 (circularity) |
| HCI label widening; venue-stable effect half the whole-cell one (K, K′) | A1, A2 |
| Spurious parent–subfield asymmetry before shared abstracts were removed | A1 (nesting) |
| Grid missed *network*, *platform*, *attention* (round 5, P2) | A4, A5, A2 |
| Sense clusters split stock phrases from content uses (round C) | A6, A7, A2 |
| APD keeps its agreement with the periods shuffled (round U) | A7, and plausibly A2–A3 |
| Spread across seeds near zero in small cells (round C) | A8 |
| Round C″ failed on a marker that formed its own sense | A9 |

## What to test first

1. **A5, the manipulation check.** Cheap, and it decides whether any known-answer verdict means
   what it says. Before another calibration round, count each known sense's presence per cell.
2. **A1 and A2, communities defined by venue and composition measured.** The largest structural
   question: whether "a field" in this corpus is a community or a classifier's category. A
   venue-defined version of one robust finding (*embodiment*) and one known answer would show
   how much the label matters.
3. **A4, finer time windows** around documented dates, after A5 has located the change.

The rest are scope statements to add to the README, or practices for future rounds (A8, A9).
