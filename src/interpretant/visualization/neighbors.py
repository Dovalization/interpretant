"""Nearest-neighbour comparison table for semantic drift visualisation."""

import numpy as np
import pandas as pd

from interpretant.drift.metrics import _top_k_neighbors


def build_neighbors_table(
    word: str,
    vectors_by_decade: dict[int, dict[str, np.ndarray]],
    k: int = 10,
) -> pd.DataFrame:
    """Build a DataFrame comparing top-k neighbours across decades.

    Each column is a decade; each row is a rank position (1 … k).
    Cell values are the nearest-neighbour word at that rank for that decade.

    Args:
        word: The target word.
        vectors_by_decade: Mapping of decade → {word: vector} for the whole vocab.
        k: Number of nearest neighbours to include per decade.

    Returns:
        DataFrame with columns = decade labels, index = rank (1-indexed).
    """
    decades = sorted(vectors_by_decade.keys())
    columns: dict[str, list[str]] = {}

    for decade in decades:
        vectors = vectors_by_decade[decade]
        if word not in vectors:
            columns[str(decade)] = ["(n/a)"] * k
            continue

        top_words = _top_k_neighbors(word, vectors, k)
        while len(top_words) < k:
            top_words.append("(n/a)")
        columns[str(decade)] = top_words

    return pd.DataFrame(columns, index=range(1, k + 1))
