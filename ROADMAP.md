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
  philosophy; philosophy's model predicting HCI's abstracts better over time, one-sided;
  fields converging between the 1980s and the 2010s; the subfield → parent asymmetry.
- Robust in part: *representation* in AI towards learned representations; HCI's
  experiential *affordance* in the 2010s; philosophy's 2020s *consciousness* (a candidate
  under TWEC in both corpora, described differently by each method).
- Not robust: *information* converging between psychology and computer science.
- Open: the cause of the philosophy → HCI effect.

---

## Next

Each item that produces results gets its predictions committed first.

1. ~~Lemmatise the corpus~~ Done 2026-11-09 as round 4 (`docs/predictions.md`).
2. **Library-first orchestration and a `predictions` package.** Move the grid, perplexity and
   TWEC loops out of `__main__.py` into the library with tests on synthetic cells; replace the
   seven `score_*.py` scripts' shared helpers with one module. GitHub Actions running ruff,
   mypy and pytest.
3. **Deploy the web explorer** (`web/`, built November 2026) and link it from the README, the
   GitHub "About" field and doval.dev (whose project schema needs an `href` field). Retire the
   Streamlit dashboard once the explorer covers its two panels.
4. **Explain the one-sided philosophy → HCI effect.** Score HCI's held-out abstracts
   individually under philosophy's model and test whether the gain concentrates in a readable
   subset (topics) or spreads evenly (shared academic register).
5. **Field separability over time.** A TF-IDF and logistic-regression classifier that guesses
   the field from an abstract, per decade; falling accuracy between two fields measures
   convergence, and the misclassified abstracts are the borrowing cases to read.
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
