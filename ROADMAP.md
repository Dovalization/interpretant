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
- Not achieved: senses by field and decade (item 8); its readings are exploratory only.
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
   2. this project's known answers over time (*attention, hallucination, prompt, alignment,
      platform, network*, and quiet transitions), configuration frozen from rung 1;
   3. known answers between fields: *embodiment* between HCI and philosophy diverges, random
      controls do not, and Wiktionary's field-labelled senses as a larger set;
   4. only then open questions: a survey of field-specific meanings, and one-sided changes
      such as the language-model sense of *hallucination* reaching psychology.

   Not part of it: clustering usages into senses, repairing the grid, the direction of HCI's
   change.
9. **Changepoint detection** on within-field trajectories once lemmatised models exist.
10. Optional layers: an ACL Anthology layer with a language filter; a full
    cognitive-psychology download on the free OpenAlex budget; public-domain books before
    1930 sampled by Library of Congress class.
