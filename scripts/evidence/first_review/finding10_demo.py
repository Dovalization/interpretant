"""Two checks for finding 10, on simulated data.

1. With 1,000 shuffles, can Benjamini-Hochberg ever pass a word when 100 words are screened?
2. With no change at all, does J's spread depend on the number of usages per period?
"""

import numpy as np
from scipy import stats

rng = np.random.default_rng(0)


def j_statistic(vectors: np.ndarray, period: np.ndarray, axis: int = -1) -> np.ndarray:
    """Mean within-period similarity minus mean across-period similarity (unit vectors)."""
    labels = np.moveaxis(period, axis, -1).astype(float)  # (..., n)
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, 0.0)
    first, second = labels, 1.0 - labels
    within = (
        np.einsum("...i,ij,...j->...", first, similarity, first)
        + np.einsum("...i,ij,...j->...", second, similarity, second)
    ) / (first.sum(-1) * (first.sum(-1) - 1) + second.sum(-1) * (second.sum(-1) - 1))
    across = np.einsum("...i,ij,...j->...", first, similarity, second) / (
        first.sum(-1) * second.sum(-1)
    )
    return within - across


def unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def word(n_per_period: int, shift: float, dimension: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Usages from two senses; `shift` moves the second period towards the second sense."""
    centres = unit(rng.normal(size=(2, dimension)))
    share = np.r_[np.full(n_per_period, 0.3), np.full(n_per_period, 0.3 + shift)]
    sense = rng.random(2 * n_per_period) < share
    vectors = unit(centres[sense.astype(int)] * 3 + rng.normal(size=(2 * n_per_period, dimension)))
    period = np.r_[np.zeros(n_per_period), np.ones(n_per_period)]
    return vectors, period


print("1. A word with a very large change, 1,000 shuffles")
vectors, period = word(200, shift=0.6)
result = stats.permutation_test(
    (period,),
    lambda labels, axis: j_statistic(vectors, labels, axis),
    permutation_type="pairings",
    n_resamples=1000,
    vectorized=True,
    alternative="greater",
    rng=rng,
)
print(f"   p = {result.pvalue:.5f}  (smallest possible: {1 / 1001:.5f})")
for screened in (20, 50, 100, 300):
    print(f"   screening {screened:>3} words: BH needs the best word at p <= {0.05 / screened:.5f}"
          f" -> {'can pass' if result.pvalue <= 0.05 / screened else 'can never pass'}")

print("\n2. No change at all: spread of J over 300 simulated words, by usages per period")
for n_per_period in (100, 200, 500):
    values = [j_statistic(*word(n_per_period, shift=0.0)) for _ in range(300)]
    print(f"   n = {n_per_period}: 95th percentile of J = {np.percentile(values, 95):.4f}")
