# Datasheet: the rebuilt abstracts corpus

The written record of agenda step 4b: "Datasheets for Datasets" (Gebru et al. 2018), with a Data
Statement (Bender and Friedman 2018) after it. It describes the build of record `75f36a82`
(the cleaning batch of runs 9, 10, 12 and 14) and the static view recorded at `8fad6e8f` (phase
step 4: the lemma exceptions, the spelling map and the compound map; first recorded at
`2c547e7a`). The
counts come from `metrics/corpus.json` and the corpus report (`interpretant report-corpus`,
`data/processed/corpus_report.md`), which holds the full layer-2 tables.

## Summary for the sign-off

**Status: approved by the author on 2026-11-16**, for agenda stages 5–13, under the stage 7
junk gate (4b, layer 4).

- **What it is.** 632,843 English research abstracts from OpenAlex, kept from 705,283 sampled in
  88 cells: 11 fields × 8 decades (1950s–2020s), each cell a seeded random sample (seed 42) of up
  to 10,000 works. 23,452 kept works belong to more than one field. After the character and word
  repairs the text is never edited: every removal is a labelled character range.
- **What the cleaning did.** 72,440 abstracts dropped (10.3%), each with one reason, and 354,259
  ranges removed, in kept and dropped abstracts, each with its reason (tables below). Every rule is
  documented with its evidence in `curated/cleaning_rules.yaml` and agenda 2.3.
- **How good it is.**
  - The author's review of run 3 found 30 non-abstracts and 3 unsure in 200 kept records (15%).
  - After the cleaning batch, calibrated agent judges found 16 failures in a fresh 200 (8%,
    Wilson 95% interval 5–13%), so the 2% target is not met. The judges agreed with the author's
    corrected verdicts on 99.0% of 200 and missed no failure.
  - The residual junk is a long tail of kinds with no clean pattern (below).
- **Why the cleaning stops anyway.** Rounds S1 and S1b (`docs/predictions_cleaning.md`): removing
  the junk the rules can find moves static drift no more than removing as much text at random,
  or than drawing the corpus again.
- **What guards the main measure.** J reads the corpus only through target usages, so a gate at
  stage 7 counts the share of each target's sampled usages that come from junk records, against a
  threshold pre-registered before the usages are drawn (agenda 4b, decided 2026-11-16).
- **The decision asked for.** Approve this corpus, with the limits listed under "Known limits",
  for agenda stages 5–13, under the stage 7 gate.

## Motivation

- **Purpose.** To measure how the meanings of terms change across decades within fields of
  research (cognitive science, computing, the social sciences and the humanities), by comparing
  the same field's cells across decades. The project's main measure is J on contextual usage
  vectors, against matched controls (agenda step 6); a static grid is an experiment (6b).
- **Created by** the author, with the pipeline in this repository; no external funding.

## Composition

- **Instances.** One OpenAlex work's abstract each, with its id, year, decade, fields, author ids
  and type. A work's text is the abstract OpenAlex holds (rebuilt from its inverted index); for
  older works this is often the opening of the article or a catalogue description.
- **Cells.** One field in one decade; a work counts in each of its fields' cells. The fields:
  philosophy (`philosophy`), history and philosophy of science (`histphilsci`), linguistics
  (`linguistics`), psychology (`psychology`), cognitive psychology (`cogpsych`), neuroscience
  (`neuroscience`), computer science (`compsci`), artificial intelligence (`ai`),
  human-computer interaction (`hci`), sociology and political science (`sociology`), and
  communication (`communication`).
- **Sampling.** Up to 10,000 English works per cell, a seeded random sample (seed 42) of research
  prose with abstracts: articles, reviews, book chapters, books, dissertations and conference
  papers; book reviews, editorials, letters, preprints and datasets excluded at download
  (`docs/corpus.md`). Early cells hold every matching work, so they are smaller.

Kept works per cell (kept / sampled):

| Field | 1950s | 1960s | 1970s | 1980s | 1990s | 2000s | 2010s | 2020s |
|---|---|---|---|---|---|---|---|---|
| ai | 234 / 292 | 1,770 / 1,907 | 4,642 / 4,937 | 9,457 / 10,000 | 9,690 / 10,000 | 9,859 / 10,000 | 9,826 / 10,000 | 9,825 / 10,000 |
| cogpsych | 783 / 1,031 | 2,711 / 3,180 | 6,893 / 7,406 | 9,467 / 10,000 | 9,491 / 10,000 | 9,579 / 10,000 | 9,623 / 10,000 | 9,591 / 10,000 |
| communication | 822 / 1,769 | 1,513 / 2,713 | 3,756 / 5,569 | 5,824 / 8,144 | 8,338 / 10,000 | 9,263 / 10,000 | 9,423 / 10,000 | 9,417 / 10,000 |
| compsci | 3,079 / 3,538 | 9,139 / 10,000 | 9,328 / 10,000 | 9,369 / 10,000 | 9,571 / 10,000 | 9,830 / 10,000 | 9,805 / 9,999 | 9,717 / 9,999 |
| hci | 21 / 21 | 179 / 189 | 452 / 486 | 2,640 / 2,945 | 7,221 / 8,002 | 9,580 / 10,000 | 9,716 / 10,000 | 9,651 / 10,000 |
| histphilsci | 3,690 / 8,354 | 4,897 / 10,000 | 6,718 / 8,579 | 8,071 / 10,000 | 8,659 / 10,000 | 9,273 / 10,000 | 9,355 / 10,000 | 9,480 / 9,997 |
| linguistics | 1,475 / 1,982 | 2,556 / 3,874 | 5,942 / 8,260 | 7,781 / 10,000 | 8,171 / 10,000 | 8,977 / 10,000 | 9,059 / 10,000 | 9,115 / 9,997 |
| neuroscience | 5,575 / 6,490 | 9,145 / 10,000 | 9,584 / 10,000 | 9,680 / 10,000 | 9,687 / 10,000 | 9,737 / 10,000 | 9,787 / 10,000 | 9,838 / 10,000 |
| philosophy | 2,001 / 2,402 | 4,606 / 5,382 | 8,809 / 10,000 | 8,880 / 10,000 | 8,664 / 10,000 | 9,308 / 10,000 | 9,487 / 10,000 | 9,590 / 9,998 |
| psychology | 5,070 / 6,913 | 8,437 / 10,000 | 9,092 / 10,000 | 9,376 / 10,000 | 9,583 / 10,000 | 9,595 / 10,000 | 9,607 / 9,999 | 9,436 / 9,999 |
| sociology | 4,385 / 5,870 | 7,965 / 10,000 | 8,351 / 10,000 | 8,897 / 10,000 | 9,158 / 10,000 | 9,519 / 10,000 | 9,582 / 10,000 | 9,550 / 9,999 |

- **Overlap.** The fields overlap by design: psychology contains cognitive psychology, and
  computer science contains AI and HCI. 23,452 kept works sit in more than one cell.
- **Flagged, not dropped.** `repository_only` 96,267 (primary source a repository, listed in none
  of Crossref, PubMed, arXiv or DOAJ); `truncated` 79,091 (the abstract stops mid-sentence);
  `malformed` 2,117 (a "sentence" over 150 words); `retracted` 147; `paratext` 9.
- **Length.** Median characters by decade from 734 (1950s) to 1,214 (2020s); the share under 250
  characters from 15.7% (1950s) to 2.9% (2020s).

## Collection process

- **Source.** The OpenAlex API, filtered by each work's primary topic; the raw cells are dated
  9–11 November 2026. Crossref and PubMed records were fetched for the record-kind rules and the
  linked tables.
- **What the source changed, and its coverage** (agenda 4b, layer 1):
  - OpenAlex replaced its topic classifier in October 2026; the raw cells appear to carry the new
    assignments (an open question, checked from the download records).
  - Two large publishers withdrew closed-access abstracts in November 2022 and November 2024
    (Kramer 2024).
  - OpenAlex holds an abstract for over 60% of 2022 works and about 45% before 2000.
  - HCI's topic label widened from 915 venues to 3,601.

## Preprocessing and cleaning

The stages (agenda 9.1; `dvc.yaml`): character and word repairs (mojibake, glued and split words),
one spaCy pass (`en_core_web_sm` 3.8.0) cached as `DocBin`s, sentence languages (lingua 2.1.1),
the cleaning patterns (spaCy's `SpanRuler`) and keeping rules, then the corpus-level rules
(template and repeated sentences, near-duplicates, the author cap, the record kinds).

**The static view** (`corpus/static_view.py`; checked against the previous view in
`docs/preprocessing_phase.md`, step 4) holds one entry per word token of every cached document,
104,176,115 in all, each with its character offset:
- the token lowercased, its accents folded, the en dash written as a hyphen;
- plural nouns and third-person singular verbs written as their lemma, with the lemma exceptions;
- each spelling variant written one way by the audited spelling map (1,946 pairs,
  `docs/spellings.md`), per hyphen segment;
- each two-part compound written one way, open, hyphenated or closed, by the audited compound map
  (1,971 pairs, `docs/compounds.md`).

Both maps leave accented tokens alone. A spaced compound merges only as two adjacent nouns in one
sentence; it writes its standard in the first entry and an empty string in the second. A token
whose standard is open holds that standard with its space (*user interface*). Readers take the
entries as words (`corpus/cell_versions.segment_words`), so the empty entries drop out. Single
words are not joined into phrases (agenda 3.5).

Dropped abstracts, by reason:

| Reason | Abstracts |
|---|---|
| not_english | 18,742 |
| citation_record | 13,957 |
| template | 8,789 |
| catalogue_record | 8,037 |
| web_scrape | 7,604 |
| near_duplicate | 5,581 |
| too_short | 4,825 |
| citation_line | 2,391 |
| letter | 1,082 |
| obituary | 573 |
| news | 252 |
| mojibake | 228 |
| author_cap | 190 |
| erratum | 128 |
| exam_spam | 61 |

Removed ranges inside kept and dropped abstracts, by reason:

| Reason | Ranges |
|---|---|
| not_english (sentences) | 112,951 |
| template_sentence | 108,407 |
| abstract_label | 37,815 |
| web_scrape | 15,688 |
| citation_record | 14,937 |
| url | 14,209 |
| latex | 10,772 |
| catalogue_record | 8,037 |
| footer | 7,418 |
| doi | 5,768 |
| copyright | 5,323 |
| repeated_sentence | 4,815 |
| citation_line | 3,018 |
| review_header | 1,404 |
| email | 1,164 |
| letter | 833 |
| exam_spam | 715 |
| thesis_notes | 599 |
| event_header | 198 |
| download_stamp | 156 |
| encoded_link | 32 |

Each rule's scope and evidence are in `curated/cleaning_rules.yaml` and agenda 2.3; the runs that
added or sent back rules are recorded there with their records.

## Quality review (4b, layer 3)

- **Run 3, the author (2026-11-14 to 15).** 200 kept records, 25 per decade: 30 not abstracts and
  3 unsure (15%, Wilson 11–21%); sentence breaks 98.2% precise on 338 cuts. Most failures were not
  the work's own text: citation lines and publisher pages (10), archive and repository items (10),
  tables of contents (4).
- **The calibrated judges.** Two Sonnet judges, blind, with 4b's failure definition, calibrated
  on run 3's verdicts. A blind Opus adjudicator agreed with the author on 19 of 20 controls and
  overturned 7 of the author's verdicts where all three models disagreed with them. Against the
  corrected verdicts, the two judges with the adjudicator on their splits agree on 99.0% and
  miss no failure.
- **After the cleaning batch (2026-11-16).** A fresh 200 kept records, 25 per decade, none of
  run 3's: 16 failures (8%, Wilson 5–13%), by decade 3, 3, 1, 0, 3, 1, 5, 0. Their kinds: lists
  of reviewed books (2), archive descriptions (2), an obituary and death notices (2), cut-off
  fragments (2), citation lines in forms the rules miss (2), a thesis note (1), and one each of a
  contents list, a keyword jumble, a lone quotation, interviewee biographies and a list of verb
  examples.
- **Does the junk matter?** Rounds S1 and S1b: the identifiable junk (0.8% of the text) moves
  static drift no more than random removal or a redraw; the static flags themselves are not
  reproducible (79% change between bootstrap samples, 51% between two averages of 9 models).

## Known limits

- **Residual junk** of about 8% (5–13%), mostly kinds with no clean pattern; the 2% target is
  not met, and the check moves to stage 7's gate on target usages.
- **Field labels** come from OpenAlex's topic classifier: nested fields overlap, sparse early
  cells are noisy (the 1950s "HCI" works were studies of eye movements in the November probe),
  and the October 2026 reclassification may have moved works between fields.
- **Small early cells**: 6 cells hold fewer than 1,000 kept works (HCI 1950s–1970s, AI 1950s,
  cognitive psychology and communication 1950s), too few for some measures.
- **Old decades** carry OCR damage (partly repaired), catalogue-style records and short
  abstracts; flagged truncated abstracts (79,091) are kept.
- **Repository-only works** (96,267) are flagged, not dropped: theses, reports and preprints mixed
  with articles, most in the early humanities cells.
- **Lemmas** are spaCy's small model's in context, with the 3,287 exceptions of the lemma review
  (2026-11-17, `curated/lemma_exceptions.tsv`, `docs/lemma_review.md`): *senses* becomes *sense*,
  *data* and *media* keep their form, and a plural whose singular is mostly an adjective keeps
  its form (*professionals*).
- **Spellings and compounds** are written one way in the static view only; the text keeps the
  author's. The maps' audits passed with 3 and 1 wrong pairs in samples of 150, so a few wrong
  merges remain. The map also hides how each compound's ways of writing drift across decades
  (in the families that vary, closed forms gain about 20 points from the 1950s to the 2020s), so
  that drift is reported
  separately in `docs/compounds.md`. Accented tokens (mostly French words) are left unmapped, and
  three-part compounds (*state-of-the-art*) are left as written.
- **Sentences in other languages** remain: about 1% of the sentences sampled for the lemma review
  (33 of 3,551) read as another language, mostly French. A likely cause, not confirmed: in text
  whose accents were stripped, French *à* becomes *a*, which the language stage's list of English
  function words counts, so a sentence of an abstract read as English is not checked on its own
  (`corpus/language.py`). Not fixed, since cleaning stopped (rounds S1 and S1b); the stage-7 gate
  on target usages covers what reaches the measures.

## Uses

- **Intended.** Within-field comparisons of word use across decades: J on contextual usages
  with matched controls (agenda step 6), and the static grid if it earns its place (6b).
- **Not intended.** Counting a field's output, bibliometrics, judging authors or venues, or
  comparisons across fields that ignore their overlap.

## Distribution

The corpus is not redistributed. The repository holds the code, the curated rules, the run
records and the content hashes of every built file (`dvc.lock`); the data live in the DVC store
and on the run machine. OpenAlex's records are CC0; the abstracts' own rights stay with their
publishers.

## Maintenance

Every rebuild runs from `dvc.yaml` and is recorded by a commit of `dvc.lock` and
`metrics/runs.jsonl`. `interpretant compare-corpus` reports what changed against any recorded
build. This datasheet is rewritten when the sources or the cleaning rules change; the layer-2
counts are regenerated with every build in the corpus report.

## Data Statement

- **Curation rationale.** Research abstracts sampled per field and decade, so each field's
  history can be followed with comparable samples; English only, since the measures are English
  models.
- **Language variety.** English (`en`), the academic register of published research; American
  and British spelling mixed in the text (the static view writes each spelling one way); many
  authors write English as a second language, more so in recent
  decades. Sentences lingua reads as another language are removed, and abstracts mostly in
  another language are dropped.
- **Speaker demographic.** The authors of indexed research works, recorded by OpenAlex author ids
  only; the author cap keeps any one author's share of a cell small (190 abstracts dropped).
- **Annotator demographic.** The corpus text has no annotation. The quality review was made by
  the author (run 3) and by language-model judges (Claude Sonnet, with Claude Opus adjudicating),
  calibrated on the author's verdicts.
- **Speech situation.** Written, edited, published abstracts (for older works often an article's
  opening or a catalogue description), addressed to researchers of the field.
- **Text characteristics.** Short expository texts (median 734–1,214 characters by decade), with
  field terminology, citations and boilerplate partly removed as labelled ranges.
- **Recording quality.** Digital text from publishers and from OCR of older printed works:
  character and word repairs fix most of the damage the repair rules know; the rest stays.
- **Provenance.** OpenAlex (CC0 metadata), with Crossref and PubMed records used for the record
  kinds.

## Sign-off

Approved by the author on 2026-11-16: the corpus of the build of record `75f36a82` is used for
agenda stages 5–13, with the known limits above, under the stage 7 junk gate (agenda 4b). The
review behind it is the calibrated judges' check in place of a fresh manual review, as decided
the same day.
