# Methods

## Embedding models

### Word2Vec (skip-gram)

Default method. Trains one model per decade using gensim's `Word2Vec`.

### FastText

Adds subword information; useful for morphologically rich domains and handling rare words.

## Alignment

### Procrustes (default)

Rotates each decade's embedding matrix onto a reference decade (default: 2010) using orthogonal Procrustes (`scipy.linalg.orthogonal_procrustes`). No extra dependencies required.

### TWEC

Temporal Word Embeddings with a Compass. Requires a custom gensim fork.

**Installation:**

```bash
pip install git+https://github.com/valedica/gensim.git
```

> **Note:** The standard `gensim` package will NOT work for TWEC. Use Procrustes if you cannot install the custom fork.

## Drift metrics

### Cosine distance

`1 − cosine_similarity(v_t1, v_t2)`. Range: [0, 2].

### Neighbourhood shift

Jaccard distance between the top-k nearest-neighbour sets at t1 and t2. Range: [0, 1].

### Frequency-corrected drift

Scales raw drift by a function of the word's minimum frequency across the two periods, reducing noise from rare words.

### Average pairwise distance

Mean cosine distance across all decade pairs in a word's trajectory.
