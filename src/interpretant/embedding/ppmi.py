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
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from gensim.models import KeyedVectors
from scipy import sparse
from scipy.sparse.linalg import svds

WINDOW = 5
DIMENSIONS = 100
MIN_COUNT = 20
CONTEXT_SMOOTHING = 0.75
EIGENVALUE_WEIGHT = 0.5
SvdBackend = Literal["arpack", "cuda"]


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

    The pairs of every line are counted at once, each pair a key into a size-by-size table
    (``np.bincount``); the matrix is then put in SciPy's canonical form (sorted indices, no
    duplicates), the form adding batch matrices gave before, so the SVD sums in the same order.
    """
    size = len(index)
    per_line = [
        np.fromiter((index[word] for word in line.split() if word in index), dtype=np.int64)
        for line in lines
    ]
    if not per_line or size == 0:
        return sparse.csr_matrix((size, size), dtype=np.float64)
    ids = np.concatenate(per_line)
    line_of = np.repeat(np.arange(len(per_line)), [len(line_ids) for line_ids in per_line])
    keys: list[np.ndarray] = []
    for offset in range(1, window + 1):
        same_line = line_of[:-offset] == line_of[offset:]
        left, right = ids[:-offset][same_line], ids[offset:][same_line]
        keys += [left * size + right, right * size + left]
    counts = np.bincount(np.concatenate(keys), minlength=size * size)
    nonzero = np.flatnonzero(counts)
    matrix = sparse.csr_matrix(
        (counts[nonzero].astype(np.float64), (nonzero // size, nonzero % size)),
        shape=(size, size),
    )
    matrix.sum_duplicates()
    return matrix


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
    backend: SvdBackend = "arpack",
) -> np.ndarray:
    """Truncated SVD of the PPMI matrix, word vectors weighted by singular values ** p.

    ``arpack`` is SciPy's iterative ``svds`` in float64 on the CPU. ``cuda`` is a full SVD of the
    dense matrix in float32 on the GPU (``torch.linalg.svd`` with cuSOLVER's QR-based ``gesvd``,
    which PyTorch's documentation advises when precision matters), keeping the top components:
    much faster on the run machine, but a different numerical route, so it is used only where an
    equivalence check against ``arpack`` has been recorded (docs/predictions_cleaning.md).
    """
    rank = min(dimensions, min(matrix.shape) - 1)
    if backend == "cuda":
        return _cuda_svd_vectors(matrix, rank, eigenvalue_weight)
    left, values, _ = svds(matrix, k=rank, random_state=seed)
    vectors: np.ndarray = left * values**eigenvalue_weight
    return vectors


def _cuda_svd_vectors(matrix: sparse.csr_matrix, rank: int, eigenvalue_weight: float) -> np.ndarray:
    import torch

    dense = torch.from_numpy(matrix.toarray().astype(np.float32)).to("cuda")
    left, values, _ = torch.linalg.svd(dense, full_matrices=False, driver="gesvd")
    vectors = left[:, :rank] * values[:rank] ** eigenvalue_weight
    result: np.ndarray = vectors.cpu().numpy().astype(np.float64)
    return result


def train_cell(
    path: Path,
    window: int = WINDOW,
    dimensions: int = DIMENSIONS,
    min_count: int = MIN_COUNT,
    seed: int = 42,
) -> KeyedVectors:
    """Build PPMI-SVD vectors for one cell file, with each word's count stored."""
    from interpretant.embedding.base import training_sentences

    with path.open(encoding="utf-8") as text:
        lines = [" ".join(sentence) for line in text for sentence in training_sentences(line)]
    return train_lines(lines, window, dimensions, min_count, seed)


def train_lines(
    lines: Sequence[str],
    window: int = WINDOW,
    dimensions: int = DIMENSIONS,
    min_count: int = MIN_COUNT,
    seed: int = 42,
    svd_backend: SvdBackend = "arpack",
) -> KeyedVectors:
    """Build PPMI-SVD vectors for one cell's lines of space-separated tokens, with counts stored.

    No window crosses a line, so a text broken at a removed range is several lines. A cell
    with fewer than two words at the minimum count gets no vectors.
    """
    counts = vocabulary(lines, min_count)
    words = sorted(counts, key=lambda word: -counts[word])
    if len(words) <= 1:
        return KeyedVectors(vector_size=dimensions)
    index = {word: position for position, word in enumerate(words)}
    pairs = cooccurrence(lines, index, window)
    vectors = svd_vectors(ppmi(pairs), dimensions, seed=seed, backend=svd_backend)
    keyed = KeyedVectors(vector_size=vectors.shape[1])
    keyed.add_vectors(words, vectors.astype(np.float32))
    for word in words:
        keyed.set_vecattr(word, "count", counts[word])
    return keyed
