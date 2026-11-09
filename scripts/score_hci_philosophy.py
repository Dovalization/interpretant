"""Score the HCI and philosophy follow-up (F1-F3) in docs/predictions_cross_perplexity.md.

F1 and F2 read data/drift/cross_perplexity_corrected.parquet. F3 rebuilds the corrected run's
1990s and 2020s models with build_decade (same seeds, budget and shared-abstract removal) and
attributes the excess word by word on HCI's held-out text.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from interpretant.drift.perplexity import END, UNKNOWN, build_decade

TABLE = pd.read_parquet("data/drift/cross_perplexity_corrected.parquet")
SEEDS = sorted(TABLE.seed.unique())
TEXTS = Path("data/phrased/openalex/texts")
PHILOSOPHICAL = {
    "embodiment",
    "embodied",
    "affordance",
    "experience",
    "experiential",
    "phenomenological",
    "phenomenology",
    "body",
    "bodily",
    "meaning",
    "practice",
    "perception",
}


def excess(seed: int, decade: int, model_field: str, text_field: str) -> float:
    rows = TABLE[
        (TABLE.seed == seed)
        & (TABLE.decade == decade)
        & (TABLE.model_field == model_field)
        & (TABLE.text_field == text_field)
    ]
    return float(rows.iloc[0].excess)


def report(name: str, passed: bool, detail: str) -> None:
    print(f"\n{name}: {'PASSED' if passed else 'FALSIFIED'}\n  {detail}")


def f1() -> None:
    means = {
        d: sum(excess(s, d, "philosophy", "hci") for s in SEEDS) / len(SEEDS)
        for d in (1990, 2000, 2010, 2020)
    }
    early, late = means[1990] - means[2000], means[2000] - means[2020]
    report(
        "F1 move concentrated after 2000",
        late > early,
        f"mean E(philosophy -> hci) {{1990: {means[1990]:.3f}, 2000: {means[2000]:.3f}, "
        f"2010: {means[2010]:.3f}, 2020: {means[2020]:.3f}}}; fall 1990s->2000s {early:.3f}, "
        f"2000s->2020s {late:.3f}",
    )


def f2() -> None:
    changes = {
        s: (
            round(excess(s, 2020, "philosophy", "hci") - excess(s, 1990, "philosophy", "hci"), 3),
            round(excess(s, 2020, "hci", "philosophy") - excess(s, 1990, "hci", "philosophy"), 3),
        )
        for s in SEEDS
    }
    passed = all(abs(towards) < abs(phil) for phil, towards in changes.values())
    report(
        "F2 movement is one-sided",
        passed,
        f"per seed (change in E(philosophy->hci), change in E(hci->philosophy)): {changes}",
    )


def word_excess(decade: int) -> tuple[Counter[str], Counter[str]]:
    """Pooled bits saved by HCI's own model over philosophy's, and occurrences, per word."""
    saved: Counter[str] = Counter()
    occurrences: Counter[str] = Counter()
    cells = sorted(TEXTS.glob(f"*_{decade}.txt"))
    for seed in SEEDS:
        models = build_decade(cells, seed=int(seed), fixed_budget=200_000, drop_shared=True)
        held_out = models.held_out["hci"]
        philosophy_bits, counts = models.models["philosophy"].word_bits(held_out)
        own_bits, _ = models.models["hci"].word_bits(held_out)
        for word, count in counts.items():
            saved[word] += philosophy_bits[word] - own_bits[word]
            occurrences[word] += count
    return saved, occurrences


def f3() -> None:
    saved_early, count_early = word_excess(1990)
    saved_late, count_late = word_excess(2020)
    shared = (set(count_early) & set(count_late)) - {UNKNOWN, END}
    score = {
        word: (saved_early[word] / count_early[word] - saved_late[word] / count_late[word])
        * count_late[word]
        for word in shared
    }
    top = sorted(score, key=score.__getitem__, reverse=True)[:50]
    hits = [word for word in top if word in PHILOSOPHICAL]
    report("F3 philosophy's words carry it", len(hits) >= 2, f"hits {hits}\n  top 50: {top}")


if __name__ == "__main__":
    f1()
    f2()
    f3()
