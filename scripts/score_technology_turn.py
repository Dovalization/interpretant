"""Score the technology-turn predictions (H1-H4) in docs/predictions_cross_perplexity.md."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from interpretant.drift.perplexity import END, UNKNOWN, build_decade

# Paths moved on 2026-11-10 when the lemmatised grid took the default paths; the evidence is unchanged.
TABLE = pd.read_parquet("data/drift/unlemmatised/cross_perplexity_corrected.parquet")
SEEDS = sorted(TABLE.seed.unique())
TEXTS = Path("data/phrased_unlemmatised/openalex/texts")
FULL = Path("data/phrased_unlemmatised/openalex_full/texts")
TECHNOLOGY = {
    "internet",
    "online",
    "software",
    "computer",
    "computers",
    "computational",
    "robot",
    "robots",
    "artificial_intelligence",
    "algorithmic",
    "data",
    "machine",
    "machines",
    "web",
    "cyber",
    "automation",
    "platform",
    "platforms",
}
ETHICS = {
    "ethics",
    "ethical",
    "social",
    "society",
    "human",
    "humans",
    "privacy",
    "trust",
    "moral",
    "responsibility",
    "values",
    "fairness",
}


def excess(seed: int, decade: int, model_field: str, text_field: str) -> float:
    rows = TABLE[
        (TABLE.seed == seed)
        & (TABLE.decade == decade)
        & (TABLE.model_field == model_field)
        & (TABLE.text_field == text_field)
    ]
    return float(rows.iloc[0].excess)


def change(seed: int, model_field: str, text_field: str) -> float:
    return excess(seed, 2020, model_field, text_field) - excess(seed, 1990, model_field, text_field)


def report(name: str, passed: bool, detail: str) -> None:
    print(f"\n{name}: {'PASSED' if passed else 'FALSIFIED'}\n  {detail}")


def h1() -> None:
    changes = {
        s: (round(change(s, "philosophy", "compsci"), 3), round(change(s, "philosophy", "ai"), 3))
        for s in SEEDS
    }
    passed = all(cs < 0 and ai < 0 for cs, ai in changes.values())
    report("H1 philosophy predicts computing better", passed, f"(compsci, ai) per seed: {changes}")


def h2() -> None:
    changes = {
        s: (
            round(change(s, "philosophy", "compsci"), 3),
            round(change(s, "compsci", "philosophy"), 3),
        )
        for s in SEEDS
    }
    passed = all(abs(back) < abs(forward) for forward, back in changes.values())
    report(
        "H2 one-sided", passed, f"(philosophy->compsci, compsci->philosophy) per seed: {changes}"
    )


def h3() -> None:
    shares = {}
    for decade in (1980, 1990, 2000, 2010, 2020):
        lines = (FULL / f"philosophy_{decade}.txt").read_text(encoding="utf-8").splitlines()
        hits = sum(1 for line in lines if TECHNOLOGY & set(line.split()))
        shares[decade] = round(hits / len(lines), 4)
    values = list(shares.values())
    passed = all(later > earlier for earlier, later in zip(values, values[1:], strict=False))
    report("H3 technology vocabulary rises", passed, f"share of abstracts: {shares}")


def word_excess(decade: int) -> tuple[Counter[str], Counter[str]]:
    saved: Counter[str] = Counter()
    occurrences: Counter[str] = Counter()
    cells = sorted(TEXTS.glob(f"*_{decade}.txt"))
    for seed in SEEDS:
        models = build_decade(cells, seed=int(seed), fixed_budget=200_000, drop_shared=True)
        held_out = models.held_out["compsci"]
        philosophy_bits, counts = models.models["philosophy"].word_bits(held_out)
        own_bits, _ = models.models["compsci"].word_bits(held_out)
        for word, count in counts.items():
            saved[word] += philosophy_bits[word] - own_bits[word]
            occurrences[word] += count
    return saved, occurrences


def h4() -> None:
    saved_early, count_early = word_excess(1990)
    saved_late, count_late = word_excess(2020)
    eligible = {
        w
        for w in set(count_early) & set(count_late)
        if count_early[w] >= 100 and count_late[w] >= 100
    } - {UNKNOWN, END}
    fall = {w: saved_early[w] / count_early[w] - saved_late[w] / count_late[w] for w in eligible}
    top = sorted(fall, key=fall.__getitem__, reverse=True)[:50]
    hits = [w for w in top if w in ETHICS]
    report(
        "H4 ethics and society words lead",
        len(hits) >= 3,
        f"{len(eligible)} words; hits {hits}\n  top 50: {top}",
    )


if __name__ == "__main__":
    h1()
    h2()
    h3()
    h4()
