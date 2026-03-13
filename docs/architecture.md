# Architecture

## Pipeline overview

```
Corpus (S2ORC / arXiv)
        │
        ▼
Preprocessing  ──────────────────────────────┐
(preprocess_text, filter_empty)              │
        │                                    │
        ▼                                    │
Decade slices (1960–2010)                    │
        │                                    │
        ▼                                    │
Embedding training                           │
(Word2Vec / FastText per decade)             │
        │                                    │
        ▼                                    │
Temporal alignment                           │
(TWEC / Procrustes)                          │
        │                                    │
        ▼                                    │
Drift metrics                                │
(cosine distance, neighbourhood shift)       │
        │                                    │
        ▼                                    │
Streamlit dashboard ◄────────────────────────┘
```

## Layer responsibilities

| Layer | Module | Role |
|---|---|---|
| Corpus | `src/interpretant/corpus/` | Load, preprocess, and slice raw text |
| Embedding | `src/interpretant/embedding/` | Train decade-specific word vectors |
| Alignment | `src/interpretant/alignment/` | Map vectors to a shared space |
| Drift | `src/interpretant/drift/` | Quantify semantic change |
| Visualization | `src/interpretant/visualization/` | Build Plotly figures |
| App | `src/interpretant/app/` | Streamlit interactive dashboard |

## Configuration

All runtime parameters are managed by [Hydra](https://hydra.cc) via the `conf/` directory tree. Run with `hydra.verbose=true` for full config dumps.

## Pipeline orchestration

[DVC](https://dvc.org) orchestrates the four-stage pipeline defined in `dvc.yaml`. Run `dvc repro` (or `make pipeline`) to execute.
