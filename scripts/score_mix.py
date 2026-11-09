"""Score round J (J1-J4): is the philosophy -> HCI effect HCI's own word mix changing?

Committed before it was run. J1, J2 and J4 read the corrected cross-perplexity table (whose
E(X -> HCI) cells for X other than philosophy had not been inspected); J3 rebuilds the 1990s
and 2020s models per seed, as rounds G and L8 did, and splits each model's fall on HCI's text
into a word-mix effect and a per-word-rate effect.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import pandas as pd

from interpretant.drift.perplexity import DecadeModels, KneserNeyBigram, build_decade

TABLE = pd.read_parquet("data/drift/cross_perplexity_corrected.parquet")
TEXTS = Path("data/phrased/openalex/texts")
SEEDS = (1, 2, 3, 4, 5)
EARLY, LATE = 1990, 2020
OTHER_MODELS = (
    "ai",
    "cogpsych",
    "communication",
    "compsci",
    "histphilsci",
    "linguistics",
    "neuroscience",
    "psychology",
    "sociology",
)
MIX_MODELS = ("psychology", "sociology")


def excess(seed: int | None, decade: int, model: str, text: str) -> float:
    rows = TABLE[
        (TABLE.decade == decade) & (TABLE.model_field == model) & (TABLE.text_field == text)
    ]
    if seed is not None:
        rows = rows[rows.seed == seed]
    return float(rows.excess.mean())


def fall(seed: int | None, model: str, text: str) -> float:
    return excess(seed, EARLY, model, text) - excess(seed, LATE, model, text)


def report(name: str, passed: bool, detail: str) -> None:
    print(f"\n{name}: {'PASSED' if passed else 'FALSIFIED'}\n  {detail}")


def j1() -> None:
    falls = {model: round(fall(None, model, "hci"), 3) for model in OTHER_MODELS}
    hits = sum(1 for value in falls.values() if value > 0)
    report("J1 every model gains on HCI", hits >= 8, f"{hits} of 9 fall; falls {falls}")


def j2() -> None:
    per_seed = {
        model: [round(fall(seed, model, "hci"), 3) for seed in SEEDS] for model in OTHER_MODELS
    }
    hits = [model for model, values in per_seed.items() if all(value > 0 for value in values)]
    report(
        "J2 in every seed",
        len(hits) >= 6,
        f"{len(hits)} of 9 fall in all 5 seeds: {hits}; per seed {per_seed}",
    )


def j4() -> None:
    texts = sorted(
        set(TABLE[TABLE.decade == EARLY].text_field) & set(TABLE[TABLE.decade == LATE].text_field)
    )
    lines, hits = [], 0
    for model in OTHER_MODELS:
        others = [fall(None, model, text) for text in texts if text not in (model, "hci")]
        median = statistics.median(others)
        on_hci = fall(None, model, "hci")
        hits += on_hci > median
        lines.append(f"{model}: HCI {on_hci:.3f} vs median of others {median:.3f}")
    report(
        "J4 HCI's fall stands out", hits >= 7, f"{hits} of 9 exceed the median; " + "; ".join(lines)
    )


def rates_and_counts(
    model: KneserNeyBigram, own: KneserNeyBigram, held_out: list[str]
) -> tuple[dict[str, float], dict[str, int]]:
    other_bits, counts = model.word_bits(held_out)
    own_bits, _ = own.word_bits(held_out)
    return {w: (other_bits[w] - own_bits[w]) / counts[w] for w in counts}, dict(counts)


def mix_and_rate(
    early: DecadeModels, late: DecadeModels, model_field: str
) -> tuple[float, float, float]:
    rates_early, counts_early = rates_and_counts(
        early.models[model_field], early.models["hci"], early.held_out["hci"]
    )
    rates_late, counts_late = rates_and_counts(
        late.models[model_field], late.models["hci"], late.held_out["hci"]
    )
    n_early, n_late = sum(counts_early.values()), sum(counts_late.values())
    mean_early = sum(rates_early[w] * c for w, c in counts_early.items()) / n_early
    mean_late = sum(rates_late[w] * c for w, c in counts_late.items()) / n_late
    counterfactual = (
        sum(rates_early.get(w, rates_late[w]) * c for w, c in counts_late.items()) / n_late
    )
    return mean_early - mean_late, mean_early - counterfactual, counterfactual - mean_late


def j3() -> None:
    lines, passed = [], True
    decades = {
        (seed, decade): build_decade(sorted(TEXTS.glob(f"*_{decade}.txt")), seed, 200_000, True)
        for seed in SEEDS
        for decade in (EARLY, LATE)
    }
    for model_field in MIX_MODELS:
        results = []
        for seed in SEEDS:
            early, late = decades[(seed, EARLY)], decades[(seed, LATE)]
            if early is None or late is None:
                raise SystemExit("a decade did not qualify")
            results.append(mix_and_rate(early, late, model_field))
        hits = sum(1 for _, mix, rate in results if mix > rate)
        passed &= hits >= 4
        lines.append(
            f"{model_field}: mix > rate in {hits} of 5; (fall, mix, rate) per seed "
            + str([tuple(round(v, 3) for v in r) for r in results])
        )
    report("J3 mix effect dominates for other models", passed, "\n  ".join(lines))


if __name__ == "__main__":
    j1()
    j2()
    j4()
    j3()
