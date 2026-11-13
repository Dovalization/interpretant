"""Real XL-LEXEME vectors, a case where the truth is known: one word in ONE cell (one field, one
decade), its 150 usages split at random into two halves. The true J is zero. How often does the 95%
bootstrap interval contain zero, and how often does its lower end sit above zero?"""

import warnings

import numpy as np
from scipy import stats

rng = np.random.default_rng(5)
stored = np.load("data/drift/pairs_known.npz")


def unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=-1, keepdims=True)


def j_from_counts(counts_a: np.ndarray, a: np.ndarray, counts_b: np.ndarray, b: np.ndarray) -> np.ndarray:
    sum_a, sum_b = counts_a @ a, counts_b @ b
    size_a, size_b = counts_a.sum(-1), counts_b.sum(-1)
    self_a, self_b = (counts_a**2).sum(-1), (counts_b**2).sum(-1)
    within_a = ((sum_a**2).sum(-1) - self_a) / (size_a**2 - self_a)
    within_b = ((sum_b**2).sum(-1) - self_b) / (size_b**2 - self_b)
    return (within_a + within_b) / 2 - (sum_a * sum_b).sum(-1) / (size_a * size_b)


def counts_of(indices: np.ndarray, size: int) -> np.ndarray:
    flat = indices.reshape(-1, indices.shape[-1]).astype(int)
    counts = np.zeros((len(flat), size))
    np.add.at(counts, (np.arange(len(flat))[:, None], flat), 1.0)
    return counts.reshape(*indices.shape[:-1], size)


def interval(a: np.ndarray, b: np.ndarray, method: str) -> tuple[float, float]:
    def statistic(index_a: np.ndarray, index_b: np.ndarray, axis: int = -1) -> np.ndarray:
        return j_from_counts(counts_of(index_a, len(a)), a, counts_of(index_b, len(b)), b)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = stats.bootstrap(
            (np.arange(len(a)), np.arange(len(b))), statistic, n_resamples=2000,
            vectorized=True, method=method, rng=rng,
        )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


keys = [key for key in stored.files if key.startswith("openalex|") and len(stored[key]) >= 150]
chosen = rng.choice(len(keys), size=250, replace=False)
outcomes = {"BCa": [], "percentile": []}
points = []
for index in chosen:
    vectors = unit(stored[keys[index]].astype(np.float64))
    order = rng.permutation(len(vectors))
    a, b = vectors[order[:75]], vectors[order[75:150]]
    points.append(float(j_from_counts(np.ones(75), a, np.ones(75), b)))
    for method in outcomes:
        outcomes[method].append(interval(a, b, method))

points = np.array(points)
print(f"{len(points)} word-cells, each split into two random halves of 75 usages (true J = 0)")
print(f"  J itself: mean {points.mean():+.5f}, share above zero {np.mean(points > 0):.0%}")
for method, intervals in outcomes.items():
    low, high = np.array(intervals).T
    print(f"  {method:<10} interval contains zero {np.mean((low <= 0) & (high >= 0)):.1%}; "
          f"lower end above zero {np.mean(low > 0):.1%}; upper end below zero {np.mean(high < 0):.1%}")
