"""Drift metrics between cells that share one embedding space."""

import numpy as np


def cosine_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Compute cosine distance (1 − cosine similarity) between two vectors.

    Args:
        vector_a: First embedding vector.
        vector_b: Second embedding vector.

    Returns:
        Float in [0, 2]; 0 means identical direction, 2 means opposite.
    """
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    similarity = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
    similarity = max(-1.0, min(1.0, similarity))  # guard against float precision
    return 1.0 - similarity


def batch_neighborhood_shift(
    target_words: list[str],
    words_t1: list[str],
    matrix_t1: np.ndarray,
    words_t2: list[str],
    matrix_t2: np.ndarray,
    k: int = 25,
    chunk_size: int = 500,
) -> dict[str, float]:
    """Vectorised neighbourhood shift for many words at once.

    Processes target words in chunks of ``chunk_size`` to bound peak memory.
    Each chunk produces a (chunk × n_vocab) similarity matrix rather than the
    (n_vocab × n_vocab) full pairwise matrix, keeping allocation proportional
    to the number of target words rather than the vocabulary size.

    Args:
        target_words: Words to compute shift for (must be in both word lists).
        words_t1: Ordered vocabulary at time 1.
        matrix_t1: Normalised matrix (len(words_t1) × dim) at time 1.
        words_t2: Ordered vocabulary at time 2.
        matrix_t2: Normalised matrix (len(words_t2) × dim) at time 2.
        k: Number of nearest neighbours to compare.
        chunk_size: Target words processed per batch; controls peak memory.

    Returns:
        Dict mapping each target word to its Jaccard distance in [0, 1].
    """
    idx_t1 = {w: i for i, w in enumerate(words_t1)}
    idx_t2 = {w: i for i, w in enumerate(words_t2)}

    valid_words = [w for w in target_words if w in idx_t1 and w in idx_t2]
    if not valid_words:
        return {}

    results: dict[str, float] = {}
    for start in range(0, len(valid_words), chunk_size):
        chunk = valid_words[start : start + chunk_size]

        chunk_matrix_t1 = matrix_t1[[idx_t1[w] for w in chunk]]  # (chunk × dim)
        chunk_matrix_t2 = matrix_t2[[idx_t2[w] for w in chunk]]

        sim_t1 = chunk_matrix_t1 @ matrix_t1.T  # (chunk × n_vocab)
        sim_t2 = chunk_matrix_t2 @ matrix_t2.T

        for j, word in enumerate(chunk):
            i1 = idx_t1[word]
            i2 = idx_t2[word]

            row1 = sim_t1[j].copy()
            row1[i1] = -2.0  # exclude self
            top1 = set(np.argpartition(row1, -k)[-k:].tolist())

            row2 = sim_t2[j].copy()
            row2[i2] = -2.0
            top2 = set(np.argpartition(row2, -k)[-k:].tolist())

            union = top1 | top2
            results[word] = 1.0 - len(top1 & top2) / len(union) if union else 0.0

    return results
