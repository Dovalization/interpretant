"""Two checks of the detection bar (steps 5, 6.3 and A10), on simulated words.

1. Does the bootstrap interval for J cover J's true value 95% of the time, with no change and
   with change? (Abstracts resampled; a usage is never paired with its own copy.)
2. With target and controls all stable, how often does the rule "the target's 95% interval clears
   the controls' 95th percentile" flag the target, with 30 and with 59 controls? Compared with a
   rank rule: the target's J above all but r of K controls, with (r + 1) / (K + 1) <= 0.05.
"""

import warnings

import numpy as np
from scipy import stats

rng = np.random.default_rng(7)
DIMENSION = 64
N = 200


def unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=-1, keepdims=True)


class Word:
    """Two senses; the second period's share of sense two is `share + shift`."""

    def __init__(self, shift: float) -> None:
        self.centres = unit(rng.normal(size=(2, DIMENSION)))
        self.scale = rng.uniform(2.0, 4.0)
        self.share = rng.uniform(0.15, 0.6)
        self.shift = shift

    def draw(self, count: int, late: bool) -> np.ndarray:
        share = self.share + (self.shift if late else 0.0)
        sense = (rng.random(count) < share).astype(int)
        return unit(self.centres[sense] * self.scale + rng.normal(size=(count, DIMENSION)))

    def true_j(self) -> float:
        early = self.draw(200_000, late=False).mean(0)
        late = self.draw(200_000, late=True).mean(0)
        return float(0.5 * np.sum((early - late) ** 2))


def j_from_counts(counts_a: np.ndarray, a: np.ndarray, counts_b: np.ndarray, b: np.ndarray) -> np.ndarray:
    """J for resamples given as counts per original usage; pairs of copies of one usage are left
    out of the within-period means."""
    sum_a, sum_b = counts_a @ a, counts_b @ b
    size_a, size_b = counts_a.sum(-1), counts_b.sum(-1)
    self_a, self_b = (counts_a**2).sum(-1), (counts_b**2).sum(-1)
    within_a = ((sum_a**2).sum(-1) - self_a) / (size_a**2 - self_a)
    within_b = ((sum_b**2).sum(-1) - self_b) / (size_b**2 - self_b)
    across = (sum_a * sum_b).sum(-1) / (size_a * size_b)
    return (within_a + within_b) / 2 - across


def counts_of(indices: np.ndarray, size: int) -> np.ndarray:
    flat = indices.reshape(-1, indices.shape[-1]).astype(int)
    counts = np.zeros((len(flat), size))
    np.add.at(counts, (np.arange(len(flat))[:, None], flat), 1.0)
    return counts.reshape(*indices.shape[:-1], size)


def interval(a: np.ndarray, b: np.ndarray, method: str) -> tuple[float, float, float]:
    def statistic(index_a: np.ndarray, index_b: np.ndarray, axis: int = -1) -> np.ndarray:
        return j_from_counts(counts_of(index_a, len(a)), a, counts_of(index_b, len(b)), b)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = stats.bootstrap(
            (np.arange(len(a)), np.arange(len(b))),
            statistic,
            n_resamples=2000,
            vectorized=True,
            method=method,
            rng=rng,
        )
    point = float(j_from_counts(np.ones(len(a)), a, np.ones(len(b)), b))
    return point, float(result.confidence_interval.low), float(result.confidence_interval.high)


print("1. Coverage of the 95% bootstrap interval for J (300 simulated words per row, n = 200)")
print(f"   {'shift':>6}{'true J':>10}{'BCa covers':>12}{'percentile covers':>19}"
      f"{'BCa lower > true':>18}")
for shift in (0.0, 0.05, 0.15, 0.3):
    covered = {"BCa": 0, "percentile": 0}
    lower_above = 0
    truths = []
    for _ in range(300):
        word = Word(shift)
        truth = word.true_j() if shift else 0.0
        truths.append(truth)
        a, b = word.draw(N, late=False), word.draw(N, late=True)
        for method in covered:
            _, low, high = interval(a, b, method)
            covered[method] += low <= truth <= high
            if method == "BCa":
                lower_above += low > truth
    print(f"   {shift:>6.2f}{np.mean(truths):>10.5f}{covered['BCa'] / 300:>12.1%}"
          f"{covered['percentile'] / 300:>19.1%}{lower_above / 300:>18.1%}")


def background_shift() -> float:
    """Every word drifts a little with its field: the same distribution for targets and controls."""
    return float(rng.normal(0.0, 0.05))


def trial(controls: int, target_shift: float | None) -> dict[str, bool]:
    scores = []
    for _ in range(controls):
        word = Word(background_shift())
        a, b = word.draw(N, late=False), word.draw(N, late=True)
        scores.append(float(j_from_counts(np.ones(N), a, np.ones(N), b)))
    scores = np.sort(scores)
    word = Word(background_shift() if target_shift is None else target_shift)
    a, b = word.draw(N, late=False), word.draw(N, late=True)
    point, low, _ = interval(a, b, "BCa")
    allowed = int(np.floor(0.05 * (controls + 1))) - 1  # controls the target may fail to beat
    return {
        "point > 95th percentile": point > np.percentile(scores, 95),
        "interval > 95th percentile": low > np.percentile(scores, 95),
        "rank rule on the point": allowed >= 0 and point > scores[-(allowed + 1)],
        "rank rule on the interval": allowed >= 0 and low > scores[-(allowed + 1)],
    }


print("\n2. How often a rule flags the target (1,000 trials per row)")
for controls in (30, 59):
    for label, target_shift in (("stable target", None), ("shift 0.15", 0.15), ("shift 0.30", 0.30)):
        outcomes = [trial(controls, target_shift) for _ in range(1000)]
        shown = "  ".join(
            f"{name}: {np.mean([outcome[name] for outcome in outcomes]):.1%}" for name in outcomes[0]
        )
        print(f"   {controls} controls, {label:<14} {shown}")
