# interpretant — Roadmap

## State (November 2026)

The corpus was rebuilt as a field × decade grid, lemmatised (plurals and British spellings),
trained twice, checked with a second method and with cross-perplexity, and scored against
predictions committed before the evidence (four rounds on the embeddings, four on
cross-perplexity). The unlemmatised artefacts rounds 1–3 were scored on stay under
`*_unlemmatised` paths. The
history of the redesign is in [devlog 04](devlogs/04.md) and [`docs/corpus.md`](docs/corpus.md).

### Corpus

| Layer | Cells | Size | Use |
|---|---|---|---|
| `openalex` | 11 fields × 1950s–2020s (88 cells) | up to 10,000 abstracts per cell, about 677,000 in all | within-field and cross-field drift, cross-perplexity |
| `openalex_full` | philosophy 1980s–2020s, HCI 2000s–2020s (8 cells) | every matching abstract | within-field trajectories of rare targets |
| `books` | 1820s–2020s (21 cells) | 358 titles, 42.3M tokens, 250,000 tokens per author per decade | within-field trajectories; the pre-1950 decades |

Target words: `data/words_of_interest.txt` (fixed before retraining). Controls:
`data/control_words.txt`, `data/control_words_cross_field.txt`.

### Models and tables

| Artifact | Where | Settings |
|---|---|---|
| TWEC grid, seed 42 and 43 | `models/twec_grid`, `models/twec_grid_seed43` | 100 dimensions, window 5, 10 epochs, negative 5, balanced compass (200,000 tokens per cell) |
| PPMI-SVD grid | `models/ppmi_grid` | window 5, context-distribution smoothing 0.75, no shift, eigenvalue weight 0.5 |
| Drift tables | `data/drift/within_field.parquet`, `cross_field.parquet` | measure ≥ 150, pool ≥ 50, k = 25, top 5% on both measures within frequency bins |
| Cross-perplexity | `data/drift/cross_perplexity.parquet` (first run), `cross_perplexity_corrected.parquet` (5 seeds) | Kneser-Ney bigrams, 200,000 training tokens, shared abstracts removed, 1,000 held-out abstracts |

### Findings

Scored in [`docs/predictions.md`](docs/predictions.md) (rounds 1–3) and
[`docs/predictions_cross_perplexity.md`](docs/predictions_cross_perplexity.md) (C, D, F, H, L8–L10).

- Robust: *embodiment*'s two senses (HCI, philosophy); the subfield → parent asymmetry.
- No change detected: *intentionality* in philosophy, listed as stable until round 5 showed
  the grid misses most documented changes.
- Depends on the instrument: HCI's change between the 1990s and the 2020s. Both instruments
  see its GUI-engineering vocabulary recede; each decade's own language models read the change
  as a move towards the behavioural and social sciences (J, K′), the classifier held at the
  1990s, net of every field's drift, as a move towards computing (N). In one frame (W) both
  instruments read it as a move towards the behavioural and social sciences, exploratory only,
  as W's predictions were untestable. No direction is claimed.
  Fields converging between the 1980s and the 2010s is the language models' finding only.
- Robust in part: *representation* in AI towards learned representations; HCI's
  experiential *affordance* in the 2010s; philosophy's 2020s *consciousness* (a candidate
  under TWEC in both corpora, described differently by each method).
- Not robust: *information* converging between psychology and computer science.
- Not achieved: senses by field and decade (item 8); its readings are exploratory only. The
  sense-label instrument (item 8d) passes its placebo and planted-change tests but stopped at
  rung 3 (2 of 4 documented changes), so it has no reading on open questions yet.
- Open: whether round N's computing reading is its measure's artefact (counting assignments
  misreads a field that changes more than the rest), as round W suggests without testing it. No
  further direction round is planned. Label scope over time is a known hazard: a
  venue-stable check should accompany any whole-cell claim that spans decades.

---

## Next

Each item that produces results gets its predictions committed first.

1. ~~Lemmatise the corpus~~ Done 2026-11-09 as round 4 (`docs/predictions.md`).
2. ~~Library-first orchestration and a `predictions` package~~ Done 2026-11-09: the corpus,
   grid, drift, perplexity and figure loops moved from the CLI into the library with tests,
   the scorers share `interpretant.predictions`, mypy is at zero and enforced in CI, and
   `tests/test_pipeline.py` runs the whole pipeline on a synthetic grid.
3. **Deploy the web explorer** (`web/`, built November 2026) and link it from the README, the
   GitHub "About" field and doval.dev (whose project schema needs an `href` field). Retire the
   Streamlit dashboard once the explorer covers its two panels.
4. ~~Explain the one-sided philosophy → HCI effect~~ Rounds G, J and K′ (2026-11-09): HCI's
   language moved out of computing's register into the behavioural and social sciences',
   inside its own venues; the OpenAlex HCI label also widened (915 → 3,601 sources), which
   inflated the whole-cell size. Venue lookups: `corpus openalex sources`.
5. ~~Field separability over time~~ Round S (2026-11-09), 1 of 6 passed: a TF-IDF classifier
   confirms HCI's prose moved away from its 1990s self, partly towards the behavioural and
   social sciences, but the move is small, not one-sided (psychology's and sociology's 2020s
   abstracts drift towards 1990s HCI too) and HCI leaks towards computing as much; fields did
   not become harder to tell apart (S5). Round V (2 of 4): removing 29 technology terms leaves
   both the drift towards HCI and HCI's own move in place; HCI and psychology converged from
   both sides, HCI travelling further.
6. ~~Held-out check per cell~~ Round M (2026-11-09), 2 of 3: no cell a robust finding uses is
   flagged; 11 small cells are (main grid 1950s–1960s, HCI before 1980, books 2000s and
   2020s). Next: mark the flagged cells in the explorer's neighbour lists.
6b. **Direction net of drift** (round N, 2026-11-10, 1 of 5): psychology did not move towards
   HCI; since the 1990s HCI moved towards the behavioural and social sciences no more than every
   field did, and towards computing more than any field. The README was restated on
   2026-11-10: HCI's change, and no direction.
6c. **Round W, the closing round on HCI's direction** (2026-11-10, 1 passed, 4 untestable):
   both instruments in one frame (the same 1990s abstracts, measure and baseline), on whole
   cells and on the venue-stable cell. Both failed the planted check guarding the direction
   predictions, which was set against the wrong baseline; exploratory, both point towards the
   behavioural and social sciences. No direction rounds follow it.
7. ~~Devlog 05~~ Written 2026-11-09 (`devlogs/05.md`).
8. ~~Sense-level drift~~ **Stopped 2026-11-10, did not pass calibration.** Contextual
   embeddings of every occurrence, clustered into senses shared by every cell (Giulianelli et
   al. 2020; `drift senses`), calibrated on known changes before any open question. Round C
   (SciBERT and ModernBERT, which agreed): the silhouette chose two senses and joined the
   attention mechanism with cognitive attention. Round C′ (eight senses): *attention* separated,
   but the language-model sense of *hallucination* took a mixed sense in two seeds. Round C″ (a
   repaired sense-set rule): failed on *prompt* in both directions. The encoder placed every
   known change correctly; no rule for turning clusters into a meaning's share held on unseen
   data. By the rule committed with round C″, round D was not run (`docs/predictions_senses.md`,
   devlog 06). Word-targeted OpenAlex cells and a label-noise measurement remain on disk.
8a. ~~Calibrate the grid on documented changes~~ Round 5 (2026-11-10), 2 of 6: the grid
   flagged *alignment* in AI's 2020s and stayed quiet on undocumented transitions, but missed
   *attention*'s mechanism sense (a near miss), *platform* in communication and *network* at
   AI's connectionist turn. Its candidates stand; its nulls do not show stability. No survey
   of field-specific meanings is built on it, and no round repairs it.
8b. ~~A methods write-up~~ Drafted 2026-11-10 as `docs/known_answers.md`: what each instrument
   did against known answers (the direction measures, the sense instrument, the grid) and what
   that taught about measuring movement between fields and meaning within them.
8c. **Meaning by usage pairs** (planned 2026-11-10; literature and design in
   [`docs/related_work.md`](docs/related_work.md)). The sense rounds used the configuration the
   literature ranks weakest, a general encoder and clustering, with no external benchmark. The
   redesign measures how much more distant a word's usages are across two cells than within
   each, with no clustering, judged by XL-LEXEME (an encoder trained on word-in-context
   judgements), by a language model on a subsample, and by the author on a small gold set. A
   calibration ladder, each rung pre-registered with its own stopping rule:
   1. ~~reproduce XL-LEXEME's published result on DWUG EN~~ Round U (2026-11-10), 3 of 4: APD
      0.864 against the published 0.886, so the pipeline is validated; but with the periods
      shuffled APD keeps 0.842, so on this benchmark it mostly measures a word's variety of
      uses, and the across-minus-within measure (0.539, noise when shuffled) carries the
      change signal (`docs/predictions_pairs.md`);
   2. ~~this project's known answers over time~~ Round X (2026-11-10), 3 of 5: APD found 1 of
      4 documented changes; J found all 4 and *alignment* but also flagged the quiet *attention*
      in AI's 1990s → 2000s. Neither measure's pair passed, so **the design stopped here**; rungs
      3 and 4 were not run. The diagnostics that followed (`docs/diagnostics.md`) put the
      failure on the measure, not the data or the encoder: the changes are in the cells (check
      1a), and J follows a word's contexts within a sense (check 2b, round X's X5; check 1b's
      placebo also shifted the sense mix, see the correction in the diagnostics);
   3. known answers between fields: *embodiment* between HCI and philosophy diverges, random
      controls do not, and Wiktionary's field-labelled senses as a larger set;
   4. only then open questions: a survey of field-specific meanings, and one-sided changes
      such as the language-model sense of *hallucination* reaching psychology.

   Not part of it: clustering usages into senses, repairing the grid, the direction of HCI's
   change.
8d. **Meaning by sense labels** (started 2026-11-10;
   [`docs/plan_sense_labels.md`](docs/plan_sense_labels.md),
   [`docs/predictions_labels.md`](docs/predictions_labels.md),
   [`docs/labellers.md`](docs/labellers.md)). Each usage gets one sense from a fixed list per
   word, and a cell's meaning is its distribution over the list. On the benchmark two language
   models labelled, Claude Sonnet 5.5 (Batch API) and JEV (TypeSafe AI); from round E a
   cue-seeded classifier labels this corpus, with the two as comparisons. Stopped at rung 3.
   - Round Y (DWUG EN, lists written by Claude): both rank the benchmark's changes well (Sonnet
     0.767, JEV 0.821; XL-LEXEME 0.864) and consistently, and both stopped on a variety check
     that DWUG's own clusters fail (recorded as a correction).
   - Round Z (lists from WordNet): both dropped by about 0.18; the result depends on the lists.
   - Round Q: lists induced from the texts (substitutes, neighbours, dictionaries; tidied by Opus
     5.5 or built by rule) rank the changes far worse than hand-written lists (0.39–0.59 against
     0.69–0.80 on the same half), whoever labels; Opus's tidying adds little over rules.
   - Round E: a logistic regression on XL-LEXEME, trained from cue words and the lists' example
     sentences, with no generative model and no labelling by a person, passes rung 1 (0.669, within
     0.10 of Sonnet's 0.693; JEV 0.800).
   - Lists for this corpus: drafted by Claude, reviewed and approved by the author on review
     pages (cue matches, leave-one-cue-out agreement, out-of-fold doubt, the classifier on usages
     no cue names), then frozen: `corpus.toml`, `corpus_v2.toml`, `corpus_controls.toml`.
   - Round T (rung 2, against the author's check 2b judgements): falsified narrowly (0.449
     against 0.456; 0.397 against 0.414).
   - The head-to-head round (a one-off second attempt on fresh sentences; headed "Round U" in the
     predictions file): the revised classifier (square-root class weights, cue-first labelling)
     against Sonnet, the author judging 48 disagreements blind; passed narrowly (43.9% against
     40%), losing 7 of 8 on *prompt*.
   - Round A (rung 3, round X's transitions): 2 of 4 documented changes found against the
     control-word bar (A1 falsified; all 4 against shuffles alone), neither quiet transition, a
     planted change from 20%, and the *years* placebo in none of 5 draws. The miss on *prompt* is
     a labelling error written down as a risk before the run; the miss on *network* is a change
     inside the 1980s that decade cells blur (exploratory).
   Not run: rung 4 (open questions). Next, with the author: improve the classifier on the spent
   rounds as development data, then one new pre-registered set of known changes at half-decade
   windows (*hallucination*, *transformer*, *cloud*, *embedding*), encoded once for the whole grid.
8e. **Retrain the grid on the cleaned corpus** (added 2026-11-11). The grid's embedding models
   (TWEC, both seeds, and PPMI-SVD) were trained before quality rules version 2
   (`docs/corpus.md`) removed non-English abstracts, template records and spam. Retrain them on
   the cleaned cells, rerun the candidate scan on the new models, and check in a pre-registered
   robustness round that the README's robust findings survive the cleaning.
9. **Changepoint detection** on within-field trajectories once lemmatised models exist.
10. Optional layers: an ACL Anthology layer with a language filter; a full
    cognitive-psychology download on the free OpenAlex budget; public-domain books before
    1930 sampled by Library of Congress class.
