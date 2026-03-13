"""Drift metrics for measuring semantic change between aligned embedding spaces."""

import itertools

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


def _top_k_neighbors(
    word: str, vectors: dict[str, np.ndarray], k: int
) -> list[str]:
    """Return the k nearest neighbours of word, ranked by cosine similarity."""
    target = vectors[word]
    scores = {
        candidate: float(
            np.dot(target, vector)
            / (np.linalg.norm(target) * np.linalg.norm(vector) + 1e-10)
        )
        for candidate, vector in vectors.items()
        if candidate != word
    }
    return sorted(scores, key=scores.__getitem__, reverse=True)[:k]


def neighborhood_shift(
    word: str,
    vectors_t1: dict[str, np.ndarray],
    vectors_t2: dict[str, np.ndarray],
    k: int = 25,
) -> float:
    """Measure how much a word's nearest-neighbour set changed between two time points.

    Computes the Jaccard distance between the top-k neighbours at t1 and t2.

    Args:
        word: The target word.
        vectors_t1: Mapping of word → vector at time 1.
        vectors_t2: Mapping of word → vector at time 2.
        k: Number of nearest neighbours to compare.

    Returns:
        Jaccard distance in [0, 1]; 0 means identical neighbours, 1 means no overlap.
    """
    if word not in vectors_t1 or word not in vectors_t2:
        raise KeyError(f"Word '{word}' not found in both vector sets.")

    neighbors_t1 = set(_top_k_neighbors(word, vectors_t1, k))
    neighbors_t2 = set(_top_k_neighbors(word, vectors_t2, k))

    union = neighbors_t1 | neighbors_t2
    if not union:
        return 0.0
    return 1.0 - len(neighbors_t1 & neighbors_t2) / len(union)


def frequency_corrected_drift(
    raw_drift: float,
    freq_t1: float,
    freq_t2: float,
    epsilon: float = 1e-6,
) -> float:
    """Adjust a drift score for frequency differences between two time periods.

    Low-frequency words have noisier embeddings and tend to show spurious
    drift. This correction down-weights drift for words that are rare in
    either period.

    Args:
        raw_drift: Uncorrected drift score (e.g. cosine distance).
        freq_t1: Relative frequency of the word at time 1.
        freq_t2: Relative frequency of the word at time 2.
        epsilon: Smoothing term to avoid division by zero.

    Returns:
        Frequency-corrected drift score.
    """
    min_freq = min(freq_t1, freq_t2)
    correction = min_freq / (min_freq + epsilon)
    return raw_drift * correction


def average_pairwise_distance(vectors: list[np.ndarray]) -> float:
    """Compute the mean cosine distance across all pairs of vectors.

    Useful for measuring how dispersed a word's trajectory is across
    multiple time periods.

    Args:
        vectors: List of aligned embedding vectors (one per decade).

    Returns:
        Mean pairwise cosine distance in [0, 2].
    """
    if len(vectors) < 2:
        return 0.0
    distances = [cosine_distance(a, b) for a, b in itertools.combinations(vectors, 2)]
    return float(np.mean(distances))
