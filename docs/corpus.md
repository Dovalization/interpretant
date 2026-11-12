# Corpus

The corpus decides what the drift metrics can mean. This page describes the current
sources, how to audit them, what the November 2026 audit found, and the corpus design the
project's goals require.

## Sources

| Layer | Content | Coverage | Loader |
|---|---|---|---|
| OpenAlex grid | Research abstracts in 11 fields, up to 10,000 per cell | 1950s–2020s | `OpenAlexSource` (`corpus openalex download|ingest`) |
| OpenAlex full cells | Every abstract of one field, for rare targets | philosophy 1980s–2020s, HCI 2000s–2020s | the same, with `--full` |

Until November 2026 the corpus also drew on PubMed, the ACL Anthology and arXiv, merged with
the books into one model per decade. The grid covers their fields with one taxonomy and one
sampling rule, so those loaders were removed; an ACL layer with a language filter remains an
option in the roadmap.

**Books (retired 2026-11-11).** 358 curated titles in philosophy, cognitive science, AI, design
and cultural theory, 1820s–2020s, were a third layer that only the aligned grid used: no usage
store, round or documented change since the grid read them. Their recent decades held 21 and 5
documents, single authors made up 30–45% of several early decades, and canonical books set
against research abstracts mix a change of genre with a change of meaning. The source, the
ingest and the PDF tooling were removed. Two traces remain on disk: the grid models
(`models/twec_grid`, `models/ppmi_grid` and their variants) were trained against a compass
that included the books, so a retrain without them would give different models; and the drift
tables in `data/drift/` still hold books rows, which the web export skips. The extracted texts
and the manifest are archived locally in `data/retired/books_source.tar.zst`, and the original
files they came from in `data/retired/books_originals.tar` (139 files for 135 books), whose
`PROVENANCE.csv` maps each file to its book, with how the match was checked by content and its
score. No original was found for Gibson's *The Ecological Approach to Visual Perception*. Part
of both archives is in copyright, so they are never committed.

Processed text lives in `data/processed/<source>/texts/<source>_<decade>.txt`, one document
per line.

### Preprocessing

`preprocess_text(text)` applies, in order: NFKC normalisation, lowercasing, removal of the
`¬` soft-hyphen artifact, NFKD transliteration to ASCII, removal of LaTeX math spans,
replacement of non-alphanumeric characters (except in-word hyphens) with spaces, and
whitespace collapsing. There is no stop-word removal: skip-gram's frequent-word subsampling
does that job without changing the context windows.

### Lemmatisation

Since November 2026 the processed cells pass through a type-level lemma map before phrase
detection (`corpus lemmas apply`, `data/lemmatised/`). The map collapses regular *-s*
inflections (plural nouns onto their singular, third-person verbs onto their base form; the
token must be `lemma + s/es`, `y → ies` or the *-is/-es* alternation of it) and British
spellings onto American ones (rules for *-ise/-isation*, *-yse*, *-our*, *-re*, doubled *l* and
a short word list, gated on a word list, an exception list and the American form's own frequency
in the corpus). Other verb forms, participles and *-ing* nominalisations are deliberately left
alone: a general lemmatiser turns *learning* into *learn* and *meaning* into *mean*, which would
destroy the concepts under study. Target words, hyphenated and phrase tokens are never mapped.
Irregular plurals and words kept on purpose (*data*, *media*, *means*, *odds*) are in the tracked
`data/lemma_overrides.tsv`.

Two maps exist. The grid was lemmatised with the first, learned from simplemma's lemmas of bare
word types (`data/processed/lemmas.tsv`, about 37,000 types, 8.3% of tokens). The grid is frozen,
so that map stays as it is, and the scripts that match the grid's vocabulary read it. Since
2026-11-11 the map is learned from spaCy (`corpus lemmas votes`, then `corpus lemmas learn`,
written to `data/processed/lemmas_voted.tsv`). One pass of spaCy's tagger over the raw abstracts,
each abstract tagged once, counts how every word type is read in its sentences; a type merges
only when at least 80% of at least five readings are plural-noun or third-person-verb readings of
one lemma. spaCy's word list gates the spellings. Lemmatising bare words with spaCy, with no
sentence to read, merged *loss* into *los* and *aesthetics* into *aesthetic*
(`docs/spacy.md`).

The first spaCy map (2026-11-12, votes from 1,274,029 abstracts, each tagged once) maps 21,251
types, 8.7% of the tokens: plurals 6.9%, third-person verbs 1.5%, spellings 0.1%, overrides 0.2%.
It agrees with the simplemma map on 15,801 types (8.6% of the tokens) and differs on 0.2%: it
merges possessives that lost their apostrophe (*kants* → *kant*), corrects simplemma's
singulars (*fixes* → *fix*, not *fixe*; *skies* → *sky*, not *ski*), and keeps the plurals the
override file lists (*graphics*, *odds*, *commons*). Known remaining errors, for the override
file: *cosmos* → *cosmo*, *thanks* → *thank*, *mises* → *mis*.

### OpenAlex grid

```bash
uv run interpretant corpus openalex download   # needs OPENALEX_API_KEY in .env
uv run interpretant corpus openalex ingest     # data/processed/openalex/texts/<field>_<decade>.txt
```

Each cell is a seeded random sample (seed 42) of up to 10,000 English works of research
prose with abstracts (articles, reviews, book chapters, books, dissertations and conference
papers; one genre rule for every field),
filtered by the work's primary topic: philosophy, history and philosophy of science,
language and linguistics, psychology, experimental and cognitive psychology, neuroscience,
computer science, AI, HCI, sociology and political science, and communication. The raw
cells keep each work's id, year and author ids, so an author cap can be applied later.
A full download costs about 3,800 credits of the key's 10,000 daily budget; finished cells
are skipped, so a run stopped by the budget resumes the next day.

OpenAlex also indexes archive catalogue records as works. The first, unfiltered download
(kept in `data/raw/openalex_unfiltered/` as a reference) found catalogue records of letters in
a third of 1950s history and philosophy of science (Albert Sabin's alone were 15%), and identical news-script records from a
Fort Worth TV station in half of 1960s–70s communication. Book reviews, editorials,
letters, preprints (often the same paper as a published article) and datasets are
therefore excluded at download, each work's `type` is stored, and `corpus openalex ingest`
drops exact duplicate abstracts and catalogue patterns ("Letter from X to Y dated ...",
"Script from ..."), reporting how many it removed per cell. Findings must also hold in an
articles-only re-check.

`--full` downloads every matching work instead of a sample, for words too rare to measure
in 10,000 abstracts (HCI 2000s–2020s, for *affordance*); those cells are kept in a separate
`--raw-dir` because they break the equal-size grid.

Two caveats from the November 2026 probe. Psychology contains experimental and cognitive
psychology, and computer science contains AI and HCI, so those samples overlap. And topic
labels are noisy in sparse early cells: the 23 "HCI" works of the 1950s are studies of eye
movements.

### Target words

`data/words_of_interest.txt` was fixed in November 2026, before retraining, so the list
cannot be tuned to results. It has three tiers: a core measurable across most of the grid
(adding *cognition*, *computation*, *emergence*, *meaning*, *sign*, *system* and
*control*), late arrivals whose series start where they first reach the threshold
(*affordance*, *embodiment*, *intentionality*, *interface*, *user*, *agent*,
*simulation*), and context-only words read in the communication cells (*image*,
*medium*). *Understanding*, *form*, *pattern* and *signal* were dropped: everyday senses
outweigh conceptual change. Phrase targets (*neural_network*, *social_network*, ...) wait
for phrase detection.

Preprocessing maps plurals and British spellings of the targets onto the lemma
(`TARGET_VARIANTS`: *representations* → *representation*, *behaviour* → *behavior*), so a
target gets one vector instead of several. Forms that are mostly verbs (*controls*) or
carry another sense (*media*) are left alone.

The noise floor comes from the whole vocabulary: for each pair of cells, drift is fitted
against log frequency over every word above the threshold in both cells, and a target
counts as drifting only if it is in the top 5% at its frequency. `data/control_words.txt`
holds 38 everyday words (*people*, *day*, *water*, *church*, ...) spanning the targets'
frequency range, chosen by `scripts/select_control_words.py`; they should sit at or below
the curve. They cannot carry the noise floor themselves: everyday words are rare in
abstracts, and only *people* reaches 150 occurrences in most cells.

### Phrase detection

```bash
uv run interpretant corpus phrases learn   # data/lemmatised/phrases.model + phrases.tsv for review
uv run interpretant corpus phrases apply   # data/phrased/<source>/texts/<source>_<decade>.txt
```

One gensim `Phrases` model (normalised PMI ≥ 0.5, min count 20) is learned from a
balanced sample, 300,000 tokens from every processed file, so the large modern cells do not
decide the phrases of the early decades, and the same model is applied everywhere. Learned phrases
never absorb a single-word target; the phrase targets are always merged. The learning step
also repairs words broken across lines in OCR'd books: a pair whose halves join into a word
more frequent than the pair (*lan guage*) is written as that word. A November 2026 trial on
the books, ACL and arXiv learned about 1,260 phrases and 92 repairs; it also surfaced French
text in ACL (*sur_les*, *c_est*); ACL has since left the corpus.

## Auditing the corpus

```bash
uv run interpretant corpus audit                       # tables in the terminal
uv run interpretant corpus audit --output data/processed/audit.json
uv run interpretant corpus audit --min-occurrences 300 # stricter reliability threshold
```

The audit reports two things:

1. **Tokens per cell**: one cell per `(source, decade)`, plus the merged decade.
2. **Target coverage**: for each word in `data/words_of_interest.txt`, the decades where it
   reaches `--min-occurrences` (default 150) and the number of reliable transitions. A
   transition is reliable only when the word reaches the threshold in both decades. A
   vector trained on a handful of occurrences stays close to its initialisation (the
   compass, under TWEC), so its drift is near zero and its neighbours are noise.

Until the books were retired it also reported each decade's largest book author and the
manifest's dating risks.

## Findings of the November 2026 audit

These findings and fixes concern the merged corpus and the books, both since retired. They stay
as the record of why the grid replaced the merged corpus.

- **Target words are too rare in most decades.** *network* occurs 1–29 times per decade
  from the 1820s to the 1950s; *affordance* never reaches 150 occurrences in any decade;
  *embodiment* and *intentionality* have one and two reliable transitions out of 20. The
  near-zero drift of *network* before 1960, and its 1960 neighbours, sit below any
  reasonable threshold.
- **A decade is a handful of authors.** William James is 45% of the 1890s book words,
  Kierkegaard 30% of the 1840s, Weber 30% of the 1910s, Chalmers 31% of the 2020s. Drift
  between two decades is largely a change of authors.
- **Translations carry the translator's English.** 88 entries have a dating risk. The
  1820s *representation* (526 occurrences) comes mostly from Payne's 1958 translation of
  Schopenhauer; the 1883 Haldane–Kemp translation rendered *Vorstellung* as "Idea". Kant
  (Kemp Smith, 1929) and Descartes are filed in the 1890s.
- **Corpus composition changes with time.** The 1820s–1950s are books only; PubMed enters
  in the 1960s with 13M tokens against 3M of books. Books tagged `ai_cs` appear from the
  1970s and are 67% of the 2010s. A jump at 1960 cannot be separated from the change of
  sources.
- **The design merged fields.** One model per decade over every source, so field-specific
  trajectories, borrowing and divergence were never measured.
- **Wrong files.** 19 book files held a different text from the one in the manifest (for
  example, Nietzsche's *Genealogy* was Dante's *Purgatorio*; Galton was *The Coral
  Island*), mostly from wrong Gutenberg IDs. Four more were a different translation or
  work than the manifest said (Hegel's *Phenomenology* is Inwood 2018, Kant is Meiklejohn
  1855, Kafka's *Trial* is Wyllie, Descartes was the *Discourse*).
- **Books were mostly untrained.** gensim's compiled training loop truncates a sentence
  after 10,000 tokens, and books are stored one per line (about 120,000 tokens on
  average). Every training run before November 2026 saw only the first 10,000 tokens of
  each book, mostly front matter. Training now splits each line into sentences of at most
  10,000 tokens (`interpretant.embedding.base.training_sentences`).
- **Data loss.** PubMed processed files for the 1970s–2020s, raw PubMed and arXiv
  downloads, `arxiv_2010.txt`, the compass vectors and the 1980s–2010s decade vectors were
  deleted in June 2026. `drift.parquet` and `field_lexicons.json` survive.

### Fixes applied (November 2026)

`scripts/redate_books.py` records every edition decision and applies it to the manifest:

- Each translation has `translator`, `original_language` and `original_year`; `year` is the
  English edition in the file, and `decade` follows from `year`. Files moved to their new
  decade folders (121 moves).
- 15 wrong-book files were replaced with the right public-domain edition (Gutenberg or the
  Internet Archive); `source_url` records where each one came from.
- `excluded/` holds 2 duplicates whose text is already in the corpus (Poincaré's *Value
  of Science* is inside *The Foundations of Science*; the Benjamin essay is inside
  *Illuminations*). *Fear and Trembling* was removed from the manifest: its file was a
  different book, and it adds little target vocabulary next to *The Concept of Dread*.
  `canon/` holds Coleridge (1817) and *Frankenstein* (1818), which precede the first
  decade. The loader reads only decade folders, so neither folder enters training.
  Entries there carry an `exclusion` reason.
- Four books were supplied by hand: Morris's *Signs, Language and Behavior* (OCR of the
  1946 edition, replacing a file with a broken font), Lowrie's *Concept of Dread* (1944),
  Austin's Frege (1953, OCR) and Zohn's complete *Illuminations* (1968; the old file held
  only the Work of Art essay).
- The duplicate Lukács entry was removed (362 entries after both removals).

After re-ingest the books are 42.3M tokens and the audit reports no dating risks. Reliable
transitions barely moved (*reasoning* 7 → 12, *behavior* 7 → 10); the rare-word problem is
a volume problem that re-dating cannot fix. The largest author still exceeds 25% in the
1820s, 1840s, 1860s, 1890s, 1930s and 2020s.

## Quality rules, version 2 (2026-11-11)

A corpus-wide scan of the grid's embedding models for words worth measuring surfaced records that
are not English research abstracts. Measured on the cleaned cells that every round reads (677,008
abstracts), `interpretant.corpus.quality.clean_cells` removes them:

| Rule | Removed |
|---|---|
| Not English: 20 or more words, under 15% English function words (*the*, *of*, *and*...) | 2,288 abstracts (0.34%) |
| Template records: half or more of the text in sentences shared by 20 or more grid records (digitised student newspapers, agency newsletters, archive descriptions) | 1,132 (0.17%) |
| Exam-dump advertising (a short pattern list) | 83 (69 in computer science's 2020s) |
| Shared notice sentences stripped, the abstract kept (Springer's "these keywords were added by machine", Altmetric's score text) | from 8,422 abstracts |

English abstracts centre around 43% function words, and 95% of all records score above 27%; the
non-English ones score near zero, so the 15% cut sits in the gap. The hardest-hit cells are
linguistics in the 1960s to 1980s and sociology in the 1950s and 1960s (1 to 2% each). The 269
shared sentences are computed once from the raw grid and cached under `data/processed/`.

The rules are a separate step, applied to a round's cells before sampling, so every round run
before 2026-11-11 reproduces exactly; each round from then on states whether it applies them. The
grid's embedding models were trained before this step and keep these records.

Known residue, not filtered:

- **Non-prose metadata.** Some records are reference lists, journal front matter or conference
  headers with enough English function words to pass (18 to 22%). They are rare and carry few of
  the words measured.
- **Changes of source.** In the 2000s, philosophy and cognitive psychology take in many abstracts
  from Chinese journals on Marxist philosophy and education reform. They are research in English,
  so they stay, but they move words such as *construction*, *development* and *education* by
  source rather than by meaning, and a reading of those words says so.

## Target design: a field × decade grid

Every stated goal compares something specific: one field across decades, or several fields
within a decade. The corpus is therefore a grid of cells, one per field and decade.

| Goal | Corpus requirement |
|---|---|
| Field-specific trajectories | One embedding slice per field and decade, trained on a shared TWEC compass; one field taxonomy across all sources |
| Legible inflection points | A field's series comes from the same sources across its range; a general-English baseline separates field change from language change |
| Asymmetric borrowing | Source and borrowing fields both have volume in the same decades |
| Convergence and divergence | Cells downsampled to equal size, with repeated samples |

Rules for every cell:

- A word counts in a cell only above the occurrence threshold; below it the cell is
  missing for that word, not computed.
- Each author contributes at most a fixed share of a cell.
- The sampling protocol is fixed before results are inspected.

### Layers

Layers 3 to 5 were the plan for books. The books layer was retired on 2026-11-11, and none of
the three is planned.

1. **General-English baseline (1800–2000).** HistWords (Hamilton, Leskovec and Jurafsky,
   2016) publishes SGNS, SVD and PPMI embeddings per decade from Google Books and COHA.
   Use it to check whether a shift is field-specific, and its frequency tables to test
   cell coverage. The Google Books 5-grams allow retraining with phrase detection.
2. **Scientific fields (1960 onward).** OpenAlex classifies works into 4 domains, 26
   fields, 252 subfields and about 4,500 topics with one model, so philosophy,
   psychology, neuroscience, computer science and the social sciences share labels.
   Abstracts arrive as an inverted index and cover about 45% of pre-2000 works.
   The grid covers the fields PubMed, ACL and arXiv served. Sample systematically per
   field and decade at equal size.
3. **Public-domain books (before 1930).** Sample by Library of Congress class (B, BF, Q, H)
   instead of curating, and date each volume by the scanned edition. HathiTrust's
   public-domain dataset needs research approval and an institutional agreement; the
   Internet Archive has edition dates without an agreement.
4. **The 1930–1960 gap.** Most text is in copyright. HathiTrust's Data Capsule served
   in-copyright analysis to member institutions, but HathiTrust announced in October 2024
   that it would stop funding the HTRC. Unless an alternative appears, these decades rely
   on the baseline and the canon layer, and their claims are weaker.
5. **Canon layer.** The curated books stay, sliced by author and tradition instead of by
   decade: how Kant's translators, James, Husserl and Searle use *representation*. Some
   of these files cannot be redistributed, so decade-level claims rest on layers 1–3.

Words such as *affordance*, *interface*, *user* and *embodiment* start their series in the
first decade where they reach the threshold.
