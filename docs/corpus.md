# Corpus

The corpus decides what the drift metrics can mean. This page describes the current
sources, how to audit them, what the November 2026 audit found, and the corpus design the
project's goals require.

## Current sources

| Source | Layer | Coverage | Loader |
|---|---|---|---|
| Books | Curated philosophy, cognitive science, AI, design and cultural theory | 1820s–2020s, 364 manifest entries | `BooksSource` (`data/external/books/`, `manifest.json`) |
| ACL Anthology | NLP abstracts | 1965–present | `ACLAnthologySource` |
| arXiv | cs.CL, cs.LG, cs.AI, physics.hist-ph, q-bio, nlin abstracts | 1991–present | `ArxivSource` |
| PubMed | Biomedical abstracts | 1960s–2020s | `PubMedSource` |
| OpenAlex | Abstracts in 11 fields (the field × decade grid) | 1950s–2020s | `OpenAlexSource` |
| S2ORC | Stub | — | `S2ORCSource` |

Processed text lives in `data/processed/<source>/texts/<source>_<decade>.txt`, one document
per line.

### Preprocessing

`preprocess_text(text)` applies, in order: NFKC normalisation, lowercasing, removal of the
`¬` soft-hyphen artifact, NFKD transliteration to ASCII, removal of LaTeX math spans,
replacement of non-alphanumeric characters (except in-word hyphens) with spaces, and
whitespace collapsing. There is no stop-word removal: skip-gram's frequent-word subsampling
does that job without changing the context windows.

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
*simulation*), and context-only words read in the books and communication cells (*image*,
*medium*). *Understanding*, *form*, *pattern* and *signal* were dropped: everyday senses
outweigh conceptual change. Phrase targets (*neural_network*, *social_network*, ...) wait
for phrase detection.

Preprocessing maps plurals and British spellings of the targets onto the lemma
(`TARGET_VARIANTS`: *representations* → *representation*, *behaviour* → *behavior*), so a
target gets one vector instead of several. Forms that are mostly verbs (*controls*) or
carry another sense (*media*) are left alone.

Control words, frequency-matched to the targets and expected to stay stable, will go in
`data/control_words.txt`; a target counts as drifting only if it moves more than the
controls in the same cell.

## Auditing the corpus

```bash
uv run interpretant corpus audit                       # tables in the terminal
uv run interpretant corpus audit --output data/processed/audit.json
uv run interpretant corpus audit --min-occurrences 300 # stricter reliability threshold
```

The audit reports four things:

1. **Tokens per cell**: one cell per `(source, decade)`, plus the merged decade that
   `embed train` uses.
2. **Target coverage**: for each word in `data/words_of_interest.txt`, the decades where it
   reaches `--min-occurrences` (default 150) and the number of reliable transitions. A
   transition is reliable only when the word reaches the threshold in both decades. A
   vector trained on a handful of occurrences stays close to its initialisation (the
   compass, under TWEC), so its drift is near zero and its neighbours are noise.
3. **Author concentration**: the largest author's share of each decade's book words.
4. **Dating risks**: books filed outside their publication decade, and translations dated
   by the original instead of the English edition. Translation detection reads the
   manifest `notes`, so translations without a note are missed.

## Findings of the November 2026 audit

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
- **The design merges fields.** `embed train` merges all sources into one model per decade,
  so field-specific trajectories, borrowing and divergence are never measured.
- **Wrong files.** 19 book files held a different text from the one in the manifest (for
  example, Nietzsche's *Genealogy* was Dante's *Purgatorio*; Galton was *The Coral
  Island*), mostly from wrong Gutenberg IDs. Four more were a different translation or
  work than the manifest said (Hegel's *Phenomenology* is Inwood 2018, Kant is Meiklejohn
  1855, Kafka's *Trial* is Wyllie, Descartes was the *Discourse*).
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
- Each text is dated by the English edition it comes from; the translator and the
  original language are recorded.
- The sampling protocol is fixed before results are inspected.

### Layers

1. **General-English baseline (1800–2000).** HistWords (Hamilton, Leskovec and Jurafsky,
   2016) publishes SGNS, SVD and PPMI embeddings per decade from Google Books and COHA.
   Use it to check whether a shift is field-specific, and its frequency tables to test
   cell coverage. The Google Books 5-grams allow retraining with phrase detection.
2. **Scientific fields (1960 onward).** OpenAlex classifies works into 4 domains, 26
   fields, 252 subfields and about 4,500 topics with one model, so philosophy,
   psychology, neuroscience, computer science and the social sciences share labels.
   Abstracts arrive as an inverted index and cover about 45% of pre-2000 works.
   PubMed, ACL and arXiv stay as specialised sources mapped onto the same taxonomy.
   Sample systematically per field and decade at equal size.
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
