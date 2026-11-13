"""Is J (across minus within) a known quantity? Check on DWUG EN's stored XL-LEXEME vectors:
J against half the squared distance between the two periods' mean vectors, exactly and with the
small-sample correction, and each measure's agreement with the human graded change."""

import time
from pathlib import Path

import numpy as np
from scipy import stats

from interpretant.drift.pairs import (
    across_minus_within,
    average_pairwise_distance,
    dwug_graded_change,
    prototype_distance,
)

ROOT = Path("data/external/dwug_en_2.0.1/dwug_en")
stored = np.load("data/drift/pairs_dwug_en_2.0.1.npz")
gold = dwug_graded_change(ROOT)
words = sorted({key.split("|")[0] for key in stored.files})


def unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def half_squared_centroid_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Half the squared Euclidean distance between the mean unit vectors (the plug-in form)."""
    return float(0.5 * np.sum((unit(first).mean(0) - unit(second).mean(0)) ** 2))


def j_from_means(first: np.ndarray, second: np.ndarray) -> float:
    """J computed from the two mean vectors only, with the correction that removes each usage's
    pairing with itself: no pairwise similarity matrix is built."""
    a, b = unit(first), unit(second)
    n, m = len(a), len(b)
    within_a = (n * n * np.sum(a.mean(0) ** 2) - n) / (n * (n - 1))
    within_b = (m * m * np.sum(b.mean(0) ** 2) - m) / (m * (m - 1))
    return float((within_a + within_b) / 2 - a.mean(0) @ b.mean(0))


rows = []
for word in words:
    first, second = stored[f"{word}|1"], stored[f"{word}|2"]
    rows.append(
        (
            gold[word],
            average_pairwise_distance(first, second),
            prototype_distance(first, second),
            across_minus_within(first, second),
            j_from_means(first, second),
            half_squared_centroid_distance(first, second),
        )
    )
table = np.array(rows)
names = ["APD", "PRT", "J (pairwise, the repository's)", "J from mean vectors", "half squared centroid distance"]
print(f"{len(words)} words")
print(f"largest |J pairwise - J from means| over all words: {np.abs(table[:, 3] - table[:, 4]).max():.2e}")
print(f"Spearman between J and half squared centroid distance: "
      f"{stats.spearmanr(table[:, 3], table[:, 5]).statistic:.4f}")
print(f"Spearman between J and PRT: {stats.spearmanr(table[:, 3], table[:, 2]).statistic:.4f}")
print("\nagreement with human graded change (Spearman):")
for column, name in enumerate(names, start=1):
    print(f"  {name:<34} {stats.spearmanr(table[:, 0], table[:, column]).statistic:.3f}")

# What does J add to the centroid distance? Only the correction term, which depends on n.
print("\nJ - half squared centroid distance, per word: "
      f"min {np.min(table[:, 3] - table[:, 5]):.5f}, max {np.max(table[:, 3] - table[:, 5]):.5f}")

# Cost: 100,000 shuffles of one word with 500 usages per period, from mean vectors.
rng = np.random.default_rng(0)
vectors = unit(rng.normal(size=(1000, 1024)).astype(np.float32))
start = time.perf_counter()
similarity = vectors @ vectors.T
np.fill_diagonal(similarity, 0.0)
labels = np.zeros((2000, 1000), dtype=np.float32)
for row in labels:
    row[rng.choice(1000, 500, replace=False)] = 1.0
quad = np.einsum("si,ij,sj->s", labels, similarity, labels, optimize=True)
elapsed_matrix = time.perf_counter() - start
start = time.perf_counter()
sums = labels @ vectors
total = vectors.sum(0)
first_sq = np.einsum("sd,sd->s", sums, sums)
second = total - sums
second_sq = np.einsum("sd,sd->s", second, second)
cross = np.einsum("sd,sd->s", sums, second)
j = ((first_sq - 500) / (500 * 499) + (second_sq - 500) / (500 * 499)) / 2 - cross / (500 * 500)
elapsed_means = time.perf_counter() - start
print(f"\n2,000 shuffles of 1,000 usages: similarity-matrix route {elapsed_matrix:.2f} s, "
      f"mean-vector route {elapsed_means:.2f} s "
      f"(100,000 shuffles: about {elapsed_matrix * 50:.0f} s against {elapsed_means * 50:.0f} s)")
