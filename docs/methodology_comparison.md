# Methodology comparison: which method fits which job

*Recorded 2026-11-11, at a fresh start on the methodology. The rounds so far (`docs/predictions*.md`,
`docs/diagnostics.md`, `docs/known_answers.md`) are treated here as evidence about methods, not as
verdicts that close lines of work. This page states the problems, lists the methods that exist for
each, and records what this project has seen of them. It chooses nothing; the choices come next,
against the past results.*

## The jobs

1. **Detecting change.** Did a word's use in a field change between two periods (or differ between
   two fields), and by how much?
2. **Finding the senses.** Which meanings does the word have?
3. **Counting the senses.** Which meaning does each usage have, and so what share does each meaning
   have in each period and field?
4. **Checking the answers.** How do we know any of the above is right?

Locating *when* a change happened, and in which field first, reuses the first three jobs on finer
time windows. Where spaCy's features fit these jobs is in `docs/spacy.md`.

## The traps every method must survive

- **Words with many uses.** A word used in many ways looks changed between any two samples, even
  when nothing changed.
- **Topic changes.** A field's topics, phrasing or sources shift, and that looks like a change of
  meaning. This is the trap that has caught this project most often.
- **Small samples.** With few usages, noise lands near the decision line.
- **Sense boundaries.** Where one meaning ends and the next begins (are telephone networks and
  neural networks one sense of *network* or two?).
- **Coarse time windows.** A change inside a decade is blurred when whole decades are compared.

## 1. Detecting change

| Method | How it works | Good at | Struggles with | What we've seen |
|---|---|---|---|---|
| Average distance (APD) | How far apart the two periods' usages are, on average | Graded change | Words with many uses; topic changes | Found 1 of 4 known changes (round X); on the human benchmark it kept most of its score with the periods shuffled (round U) |
| Across minus within (J) | Distance across periods minus distance within each period | Words with many uses | Topic changes; small samples | Found 4 of 4 known changes and the weak case, with one narrow false alarm (0.0019 against a bar of 0.0015) on the smallest cells (round X); detected a planted change of source (check 1b) |
| J with same-document controls | Compare the target with other words from the same abstracts, so a field's topic change moves both | Topic changes (the aim) | Small samples | Not tried |
| Resampled intervals | Resample the usages many times; call it a change only if the whole range clears the line | Small samples | — | Not tried |
| Whole-distribution tests | Compare the full spread of usages (optimal transport, two-sample classifier tests), not only averages | Small new meanings | Topic changes | Not tried |
| Aligned word embeddings (the grid) | Train one model per period, line the models up, compare a word's positions | Scanning the whole vocabulary cheaply | Alignment noise; topic changes | Found 1 of 4 known changes (round 5); its silences carry no information |
| Shared neighbours | Count how many of a word's closest words stay the same between periods, with no alignment (Gonen et al. 2020) | Stable, readable | Topic changes | Not tried |
| Substitutes | Which words a masked language model would put in the target's place, compared between periods (Card 2023) | Partly resists topic changes; readable | Can split by topic | Not tried for detection; substitute groups split by topic and by singular and plural when used to build lists (round Q) |
| Collocations | Which words strongly co-occur with the target, compared between periods | Simplest, no neural model, readable | Topic changes, completely | Not tried |
| Compare sense shares | Detect change from the counts that job 3 produces | Readable | Inherits job 3's errors | Found 4 of 4 known changes against shuffles, 2 of 4 against control words (round A) |

## 2. Finding the senses

| Method | How it works | Good at | Struggles with | What we've seen |
|---|---|---|---|---|
| Hand-written lists | A person writes each word's senses | Sense boundaries, decided once and explicitly | The author's blind spots | Best by far: 0.69–0.80 on the human benchmark (rounds Y, Q); missed generic *transformer* and network *cloud* in round A′'s lists |
| Hand-written lists with a dictionary check | Compare the written list with Wiktionary's senses and fix gaps | Catching blind spots | — | Found those two gaps in four words (2026-11-11) |
| Dictionary as the list | Take WordNet's or Wiktionary's senses directly | Independence from the author | Technical senses; sense boundaries | WordNet about 0.18 lower (round Z); lists built with dictionary evidence lower still (round Q) |
| Lists built from the texts | Group substitutes and neighbours into senses, by rule or by a language model | Coverage | Topic changes; sense boundaries | 0.39–0.59 against 0.69–0.80 for hand-written lists on the same usages (round Q) |
| Clustering usages | Group usage vectors from a general encoder automatically (k-means) | Needs no list | Topic changes; turning groups into named meanings | Failed three times (rounds C, C′, C″) |
| Clustering same-meaning judgements | A model judges pairs of usages same or different; the pairs are grouped (correlation clustering, block models), as the human benchmark's senses were made | Picks the number of senses itself | Sense boundaries left to the algorithm | Not tried; can be checked against the human benchmark's own groups |
| Generated definitions | A small fine-tuned model writes a short definition for each usage; the definitions act as senses (Fedorova, Kutuzov and Scherrer 2024) | Readable; finds senses nobody listed | Many paraphrases of one sense to merge | Not tried; published results match or beat earlier unsupervised sense methods |

## 3. Counting the senses

| Method | How it works | Good at | Struggles with | What we've seen |
|---|---|---|---|---|
| Cue words and a classifier (current) | Cue words label the easy usages; a classifier learns from them and labels the rest | Cheap, no hand labelling | Topic changes, because cue words exist only in some eras; old senses with few examples | 84–88% against Claude's development labels; labelled 24% of AI's 2000s *prompt* usages with the language-model sense, which did not exist yet (Sonnet: 3%) |
| Language-model labellers | Sonnet or JEV reads each usage with the list | Accuracy | Cost; no view of why | Closest to the author's judgements; used as a comparison |
| Definition-reading model | A model trained to pick the right definition for a usage, with no cue words (GlossReader and similar) | Topic changes, at the root | Needs good definitions | Not tried; the hand-written lists already have definitions it could read |
| Nearest example from every era | Compare each usage with examples of each sense drawn from all periods, such as dated dictionary quotations | Old senses with few examples; topic changes | Finding enough examples | Not tried |
| A few hand labels across eras | About 30 of the author's labels per word, spread across decades | Topic changes; thin senses | The author's time | Not tried |
| Correcting the counts | Adjust the totals for known error rates (adjusted classify and count, or re-estimating each period's proportions) | Errors that stay the same over time | Errors that change over time | Adjusted counts in use since round A; their error rates come from the easy, cue-labelled usages |

## Methods that help every job

| Method | How it works | Good at | Struggles with | What we've seen |
|---|---|---|---|---|
| Removing the era signal | Find the direction in the vectors that encodes which decade a sentence sounds like, learned from words whose meaning did not change, and remove it (concept erasure) | Topic changes, for every vector method | May remove some real meaning | Not tried |
| Finer time windows | Half-decades or single years, and changepoint search on yearly series | Coarse windows; finding when a change happened | Small samples | Round A′'s cells use half-decades |

## 4. Checking the answers

| Check | What it catches |
|---|---|
| Human benchmark (DWUG EN) | Whether a method agrees with people, sense boundaries included |
| Known documented changes | Whether real changes are found |
| Quiet periods | False alarms |
| Planted changes, with no cue words in the planted usages | How small a change can be detected, without cue words doing the work |
| Placebos that keep the meaning fixed (sociology's *years* in psychology) | Topic changes mistaken for meaning |
| Anachronisms (a meaning's share before the meaning existed) | Topic changes mistaken for meaning, in the counting |
| Control words and shuffles | Drift that every word shows; words with many uses |
| Resampling | Small samples |
| The author's judgements; a language model as comparison | Labelling accuracy |

## What the tables show

- **Topic changes are the open problem.** Almost every method tried so far handles the other traps
  but not this one. The untried methods aimed at it are same-document controls and substitutes for
  detection, a definition-reading model and examples from every era for counting, and removing the
  era signal for everything.
- **For finding the senses, the best method is already known:** hand-written lists, now checked
  against a dictionary.
- **The most promising combination** keeps the hand-written lists with their definitions and counts
  the senses with a definition-reading model instead of cue words: it keeps what has worked and
  removes the cause of the *prompt* error.

## Sources

Marked as in `docs/related_work.md`: **summary** (search results and abstracts only) or **memory**
(cited from prior knowledge, not checked for this page). Neither should be relied on for numbers.

- Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff and Biderman (2023). LEACE: Perfect linear
  concept erasure in closed form. NeurIPS. Memory.
- Card (2023). Substitution-based Semantic Change Detection using Contextual Embeddings. ACL.
  <https://aclanthology.org/2023.acl-short.52>. Summary.
- Fedorova, Kutuzov and Scherrer (2024). Definition generation for lexical semantic change
  detection. Findings of ACL. <https://aclanthology.org/2024.findings-acl.339>. Summary.
- Gonen, Jawahar, Seddah and Goldberg (2020). Simple, Interpretable and Stable Method for Detecting
  Words with Usage Change across Corpora. ACL. <https://aclanthology.org/2020.acl-main.51>.
  Summary.
- Periti and Tahmasebi (2024). A Systematic Comparison of Contextualized Word Embeddings for
  Lexical Semantic Change. <https://arxiv.org/abs/2402.12011>. Summary.
- Rachinskiy and Arefyev (2022). GlossReader at LSCDiscovery: Train to Select a Proper Gloss in
  English – Discover Lexical Semantic Change in Spanish. LChange workshop.
  <https://aclanthology.org/2022.lchange-1.22>. Summary.
- Ravfogel, Elazar, Gonen, Twiton and Goldberg (2020). Null It Out: Guarding Protected Attributes
  by Iterative Nullspace Projection. ACL. Memory.
- Saerens, Latinne and Decaestecker (2002). Adjusting the Outputs of a Classifier to New a Priori
  Probabilities: A Simple Procedure. Neural Computation. Memory.
- Schlechtweg et al. (2021). DWUG: A large Resource of Diachronic Word Usage Graphs in Four
  Languages. EMNLP. Memory.
- Zamora-Reina et al. (2025). A comparison of large language models with specialised models for
  lexical semantic change detection (title not checked). ECAI.
  <https://doi.org/10.3233/FAIA251313>. Summary.
