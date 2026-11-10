"""Score round U (U1-U4): does the usage-pair pipeline reproduce a published result on DWUG EN?

Committed before it was run. Encodes every usage of DWUG EN 2.0.1 once with XL-LEXEME
(interpretant.drift.pairs.XLLexeme), caches the vectors under data/drift/, computes APD, PRT
and J per word, correlates each with the human graded change score, and runs the shuffled-period
control, as docs/predictions_pairs.md defines.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from interpretant.drift.pairs import (
    XLLexeme,
    across_minus_within,
    average_pairwise_distance,
    dwug_graded_change,
    dwug_usages,
    prototype_distance,
)
from interpretant.predictions import report

ROOT = Path("data/external/dwug_en_2.0.1/dwug_en")
CACHE = Path("data/drift/pairs_dwug_en_2.0.1.npz")
SHUFFLES = 20
SEED = 2026


def vectors() -> dict[str, dict[int, np.ndarray]]:
    """Every usage's vector by word and period, encoded once and cached."""
    usages = dwug_usages(ROOT)
    keys = [(word, period) for word in sorted(usages) for period in sorted(usages[word])]
    if not CACHE.exists():
        encoder = XLLexeme()
        encoded = {
            f"{word}|{period}": encoder.encode(usages[word][period]) for word, period in keys
        }
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE, **encoded)  # type: ignore[arg-type]
    stored = np.load(CACHE)
    result: dict[str, dict[int, np.ndarray]] = {}
    for word, period in keys:
        result.setdefault(word, {})[period] = stored[f"{word}|{period}"]
    return result


def agreement(scores: dict[str, float], gold: dict[str, float]) -> float:
    words = sorted(gold)
    rho = spearmanr([scores[word] for word in words], [gold[word] for word in words]).statistic
    return float(rho)


def main() -> None:
    gold = dwug_graded_change(ROOT)
    encoded = vectors()
    measures = {
        "APD": average_pairwise_distance,
        "PRT": prototype_distance,
        "J": across_minus_within,
    }
    scores = {
        name: {word: measure(encoded[word][1], encoded[word][2]) for word in gold}
        for name, measure in measures.items()
    }
    rho = {name: agreement(values, gold) for name, values in scores.items()}
    print(f"{len(gold)} words, {sum(len(v) for g in encoded.values() for v in g.values())} usages")
    print("Agreement with change_graded: " + ", ".join(f"{n} {r:.3f}" for n, r in rho.items()))

    rng = np.random.default_rng(SEED)
    shuffled: dict[str, list[float]] = {"APD": [], "J": []}
    for _ in range(SHUFFLES):
        values: dict[str, dict[str, float]] = {"APD": {}, "J": {}}
        for word in gold:
            pooled = np.vstack([encoded[word][1], encoded[word][2]])
            order = rng.permutation(len(pooled))
            first = pooled[order[: len(encoded[word][1])]]
            second = pooled[order[len(encoded[word][1]) :]]
            values["APD"][word] = average_pairwise_distance(first, second)
            values["J"][word] = across_minus_within(first, second)
        for name in shuffled:
            shuffled[name].append(agreement(values[name], gold))
    mean_shuffled = {name: statistics.mean(values) for name, values in shuffled.items()}

    report(
        "U1 the pipeline reproduces the published result",
        rho["APD"] >= 0.85,
        f"APD agreement {rho['APD']:.3f} (published 0.886, needed 0.85)",
    )
    report(
        "U2 APD beats PRT, as published",
        rho["APD"] > rho["PRT"],
        f"APD {rho['APD']:.3f}, PRT {rho['PRT']:.3f} (published 0.886 and 0.676)",
    )
    report(
        "U3 the redesign's measure tracks change too",
        rho["J"] >= 0.75,
        f"J agreement {rho['J']:.3f}",
    )
    report(
        "U4 J responds to change, APD also to variety",
        abs(mean_shuffled["J"]) < 0.2 and mean_shuffled["APD"] >= 0.4,
        f"periods shuffled ({SHUFFLES} times): mean agreement J {mean_shuffled['J']:+.3f} "
        f"(range {min(shuffled['J']):+.3f} to {max(shuffled['J']):+.3f}), "
        f"APD {mean_shuffled['APD']:+.3f} (range {min(shuffled['APD']):+.3f} to "
        f"{max(shuffled['APD']):+.3f})",
    )
    carried = "J" if rho["J"] >= rho["APD"] - 0.05 and abs(mean_shuffled["J"]) < 0.2 else "APD"
    print(f"\nBy the reading rule, rung 2's primary measure would be {carried}.")


if __name__ == "__main__":
    main()
