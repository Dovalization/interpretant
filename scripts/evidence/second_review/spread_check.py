"""Step 5's spread band, on round X's stored vectors (150 usages per cell).

1. At equal n, does a control's J (and APD) rise with its within-period spread?
2. How many of round X's controls fall within ±25% and ±50% of each documented change's spread,
   matching on the early period's spread or on the mean of both periods?
"""

from pathlib import Path

import numpy as np
from scipy import stats

stored = np.load("data/drift/pairs_known.npz")
TARGETS = {"attention", "network", "platform", "alignment"}
controls = {
    line.split("#")[0].strip()
    for line in Path("data/control_words_senses.txt").read_text(encoding="utf-8").splitlines()
    if line.split("#")[0].strip()
} - TARGETS


def unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def spread(vectors: np.ndarray) -> float:
    """Mean cosine distance over pairs of distinct usages in one period."""
    similarity = vectors @ vectors.T
    count = len(vectors)
    return float(1 - (similarity.sum() - np.trace(similarity)) / (count * (count - 1)))


by_transition: dict[tuple[str, str, str], dict[str, dict[str, np.ndarray]]] = {}
for key in stored.files:
    source, field, early, late, word, decade = key.split("|")
    if source == "openalex":
        by_transition.setdefault((field, early, late), {}).setdefault(word, {})[decade] = unit(
            stored[key].astype(float)
        )

rows = []
for (_, early, late), words in by_transition.items():
    for word, cells in words.items():
        if word not in controls or early not in cells or late not in cells:
            continue
        n = min(len(cells[early]), len(cells[late]))
        if n < 100:
            continue
        a, b = cells[early][:n], cells[late][:n]
        within = (spread(a) + spread(b)) / 2
        apd = float(1 - (a @ b.T).mean())
        rows.append((apd - within, apd, within))
table = np.array(rows)
for name, column in (("J", 0), ("APD", 1)):
    result = stats.spearmanr(table[:, column], table[:, 2])
    print(f"{len(table)} control-and-transition pairs: Spearman of {name} with spread "
          f"{result.statistic:+.2f} (p = {result.pvalue:.3f})")

print("\ncontrols within a spread band around each documented change (early | mean of periods)")
for (field, early, late), words in sorted(by_transition.items()):
    for target in sorted(TARGETS & set(words)):
        cells = words[target]
        if early not in cells or late not in cells:
            continue
        n = min(len(cells[early]), len(cells[late]))
        target_early, target_late = spread(cells[early][:n]), spread(cells[late][:n])
        target_mean = (target_early + target_late) / 2
        early_spreads, mean_spreads = [], []
        for word, other in words.items():
            if word in controls and early in other and late in other:
                m = min(len(other[early]), len(other[late]))
                if m >= 100:
                    first, second = spread(other[early][:m]), spread(other[late][:m])
                    early_spreads.append(first)
                    mean_spreads.append((first + second) / 2)
        early_spreads, mean_spreads = np.array(early_spreads), np.array(mean_spreads)

        def inside(values: np.ndarray, reference: float, width: float) -> int:
            return int((np.abs(values - reference) <= width * reference).sum())

        print(f"  {target} {field} {early}->{late}: spread {target_early:.3f} / {target_late:.3f}; "
              f"±25%: {inside(early_spreads, target_early, 0.25)} | "
              f"{inside(mean_spreads, target_mean, 0.25)}; "
              f"±50%: {inside(early_spreads, target_early, 0.5)} | "
              f"{inside(mean_spreads, target_mean, 0.5)} of {len(early_spreads)}")
