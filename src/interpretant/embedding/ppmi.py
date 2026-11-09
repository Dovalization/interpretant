"""Count-based embeddings per cell: PPMI factorised by SVD, a robustness check on TWEC.

Word2vec approximates a factorisation of PMI statistics (Levy & Goldberg 2014), and count
models are more stable on small corpora (Hamilton et al. 2016), so a drift finding should hold
under both. The settings follow Levy, Goldberg & Dagan (2015)'s recommendations for SVD:
context-distribution smoothing (0.75), no PMI shift (k=1, since shifting hurts SVD), eigenvalue
weighting (0.5), and the same 5-word window as TWEC.

Each cell is factorised on its own, so its vectors are not aligned with other cells: compare
PPMI cells by neighbourhood shift, not by cosine drift. Models are saved as KeyedVectors with
word counts, the format `drift grid` reads.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors
from scipy import sparse
from scipy.sparse.linalg import svds

WINDOW = 5
DIMENSIONS = 100
MIN_COUNT = 20
CONTEXT_SMOOTHING = 0.75
EIGENVALUE_WEIGHT = 0.5
LINES_PER_BATCH = 20_000


def vocabulary(lines: Iterable[str], min_count: int = MIN_COUNT) -> Counter[str]:
    """Count tokens and keep those occurring at least ``min_count`` times."""
    counts: Counter[str] = Counter()
    for line in lines:
        counts.update(line.split())
    return Counter({word: count for word, count in counts.items() if count >= min_count})


def cooccurrence(
    lines: Iterable[str], index: dict[str, int], window: int = WINDOW
) -> sparse.csr_matrix:
    """Symmetric counts of in-vocabulary word pairs within ``window`` tokens of each other.

    Out-of-vocabulary tokens are dropped before windowing, as word2vec drops words below its
    minimum count, so the two methods see the same contexts.
    """
    size = len(index)
    total = sparse.csr_matrix((size, size), dtype=np.float64)
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []

    def flush() -> sparse.csr_matrix:
        if not rows:
            return sparse.csr_matrix((size, size), dtype=np.float64)
        row, column = np.concatenate(rows), np.concatenate(columns)
        rows.clear()
        columns.clear()
        values = np.ones(len(row), dtype=np.float64)
        return sparse.coo_matrix((values, (row, column)), shape=(size, size)).tocsr()

    for number, line in enumerate(lines, start=1):
        ids = np.fromiter((index[w] for w in line.split() if w in index), dtype=np.int64)
        for offset in range(1, window + 1):
            if len(ids) > offset:
                rows += [ids[:-offset], ids[offset:]]
                columns += [ids[offset:], ids[:-offset]]
        if number % LINES_PER_BATCH == 0:
            total = total + flush()
    return total + flush()


def ppmi(
    counts: sparse.csr_matrix, smoothing: float = CONTEXT_SMOOTHING, shift: float = 1.0
) -> sparse.csr_matrix:
    """Positive PMI with context-distribution smoothing, optionally shifted by log(shift)."""
    coo = counts.tocoo()
    total = coo.data.sum()
    word_totals = np.asarray(counts.sum(axis=1)).ravel()
    context_totals = np.asarray(counts.sum(axis=0)).ravel() ** smoothing
    context_probability = context_totals / context_totals.sum()
    pmi = np.log(coo.data * total / (word_totals[coo.row] * total * context_probability[coo.col]))
    pmi -= np.log(shift)
    keep = pmi > 0
    return sparse.csr_matrix((pmi[keep], (coo.row[keep], coo.col[keep])), shape=counts.shape)


def svd_vectors(
    matrix: sparse.csr_matrix,
    dimensions: int = DIMENSIONS,
    eigenvalue_weight: float = EIGENVALUE_WEIGHT,
    seed: int = 42,
) -> np.ndarray:
    """Truncated SVD of the PPMI matrix, word vectors weighted by singular values ** p."""
    rank = min(dimensions, min(matrix.shape) - 1)
    left, values, _ = svds(matrix, k=rank, random_state=seed)
    vectors: np.ndarray = left * values**eigenvalue_weight
    return vectors


def train_cell(
    path: Path,
    window: int = WINDOW,
    dimensions: int = DIMENSIONS,
    min_count: int = MIN_COUNT,
    seed: int = 42,
) -> KeyedVectors:
    """Build PPMI-SVD vectors for one cell file, with each word's count stored."""
    from interpretant.embedding.base import training_sentences

    def lines() -> Iterable[str]:
        with path.open(encoding="utf-8") as text:
            for line in text:
                for sentence in training_sentences(line):
                    yield " ".join(sentence)

    counts = vocabulary(lines(), min_count)
    words = sorted(counts, key=lambda word: -counts[word])
    index = {word: position for position, word in enumerate(words)}
    vectors = svd_vectors(ppmi(cooccurrence(lines(), index, window)), dimensions, seed=seed)
    keyed = KeyedVectors(vector_size=vectors.shape[1])
    keyed.add_vectors(words, vectors.astype(np.float32))
    for word in words:
        keyed.set_vecattr(word, "count", counts[word])
    return keyed
