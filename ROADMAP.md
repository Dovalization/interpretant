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

- Robust: *embodiment*'s two senses (HCI, philosophy); *intentionality* stable in
  philosophy; HCI's language moving out of computing's register into the behavioural and
  social sciences' under the language models (every non-computing model gains on HCI,
  one-sided, a word-mix effect; a classifier, round S, sees the move away from 1990s HCI but
  not its one-sidedness);
  fields converging between the 1980s and the 2010s; the subfield → parent asymmetry.
- Robust in part: *representation* in AI towards learned representations; HCI's
  experiential *affordance* in the 2010s; philosophy's 2020s *consciousness* (a candidate
  under TWEC in both corpora, described differently by each method).
- Not robust: *information* converging between psychology and computer science.
- Open: the cause of HCI's move (CHI's turn to user studies is the obvious candidate; the
  corpus can date it decade by decade). Label scope over time is now a known hazard: a
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
6. **Held-out check per cell**: hold out 10% of each cell's text and measure how well its
   model predicts it, flagging cells whose model memorised a small corpus.
7. ~~Devlog 05~~ Written 2026-11-09 (`devlogs/05.md`).
8. **Sense-level drift** with contextual embeddings for *consciousness* and *information*:
   encode occurrences, cluster into senses, track each sense's share by field and decade
   (Giulianelli et al. 2020). Runs on the homelab GPU.
9. **Changepoint detection** on within-field trajectories once lemmatised models exist.
10. Optional layers: an ACL Anthology layer with a language filter; a full
    cognitive-psychology download on the free OpenAlex budget; public-domain books before
    1930 sampled by Library of Congress class.
