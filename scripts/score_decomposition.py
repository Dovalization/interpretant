"""Score round G (G1-G5): where the philosophy -> HCI gain sits, as the predictions define.

Committed before it was run. Rebuilds philosophy's and HCI's bigram models for the 1990s and
the 2020s with the corrected design (lemmatised cells, shared abstracts removed, 200,000
training tokens, 1,000 held-out abstracts, seeds 1-5), exactly as L8 was scored, and measures
each held-out abstract on its own.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from interpretant.drift.perplexity import END, UNKNOWN, DecadeModels, build_decade

TEXTS = Path("data/phrased/openalex/texts")
SEEDS = (1, 2, 3, 4, 5)
DECADES = (1990, 2020)
BUDGET = 200_000
THEMES = {
    "ethics",
    "ethical",
    "moral",
    "phenomenology",
    "phenomenological",
    "embodiment",
    "embodied",
    "ontology",
    "ontological",
    "epistemic",
    "epistemological",
    "hermeneutic",
    "agency",
    "autonomy",
    "meaning",
    "experience",
    "value",
    "values",
    "critical",
    "theory",
    "theoretical",
    "reflection",
    "reflexive",
    "normative",
}


@dataclass
class Scored:
    """Per-abstract excess of one field's held-out text under another field's model."""

    abstracts: list[str]
    excess: list[float]

    @property
    def mean(self) -> float:
        return statistics.fmean(self.excess)

    @property
    def median(self) -> float:
        return statistics.median(self.excess)

    def quantile(self, share: float) -> float:
        ordered = sorted(self.excess)
        return ordered[min(len(ordered) - 1, int(share * len(ordered)))]

    def partition(self, flavoured: bool) -> list[float]:
        return [
            value
            for abstract, value in zip(self.abstracts, self.excess, strict=True)
            if (bool(set(abstract.split()) & THEMES)) == flavoured
        ]


def score(models: DecadeModels, model_field: str, text_field: str) -> Scored:
    held_out = models.held_out[text_field]
    other, own = models.models[model_field], models.models[text_field]
    return Scored(
        held_out,
        [other.document_cross_entropy(a) - own.document_cross_entropy(a) for a in held_out],
    )


def build(seed: int, decade: int) -> DecadeModels:
    models = build_decade(sorted(TEXTS.glob(f"*_{decade}.txt")), seed, BUDGET, True)
    if models is None:
        raise SystemExit(f"no qualifying fields for {decade}s, seed {seed}")
    return models


def report(name: str, passed: bool, detail: str) -> None:
    print(f"\n{name}: {'PASSED' if passed else 'FALSIFIED'}\n  {detail}")


def main() -> None:
    runs: dict[tuple[int, int], DecadeModels] = {
        (seed, decade): build(seed, decade) for seed in SEEDS for decade in DECADES
    }
    hci = {key: score(models, "philosophy", "hci") for key, models in runs.items()}
    philosophy = {key: score(models, "hci", "philosophy") for key, models in runs.items()}

    print("Per-seed statistics of HCI's per-abstract excess under philosophy's model")
    print("seed  decade   mean  median    q25    q75   flavoured  n_flav")
    for (seed, decade), scored in sorted(hci.items()):
        flavoured = scored.partition(True)
        print(
            f"{seed:>4}  {decade}  {scored.mean:6.3f}  {scored.median:6.3f}  "
            f"{scored.quantile(0.25):6.3f}  {scored.quantile(0.75):6.3f}  "
            f"{statistics.fmean(flavoured):9.3f}  {len(flavoured):6d}"
        )

    def fall(seed: int, statistic: str) -> float:
        early, late = hci[(seed, 1990)], hci[(seed, 2020)]
        return float(getattr(early, statistic) - getattr(late, statistic))

    # G1
    ratios = {seed: fall(seed, "median") / fall(seed, "mean") for seed in SEEDS}
    report(
        "G1 fall spread across abstracts",
        all(ratio >= 0.7 for ratio in ratios.values()),
        f"median fall / mean fall per seed: { {s: round(r, 2) for s, r in ratios.items()} }",
    )
    # G2
    tails = {
        seed: (hci[(seed, 1990)].quantile(0.25) - hci[(seed, 2020)].quantile(0.25))
        / (hci[(seed, 1990)].quantile(0.75) - hci[(seed, 2020)].quantile(0.75))
        for seed in SEEDS
    }
    report(
        "G2 lower tail does not carry it",
        all(ratio <= 1.5 for ratio in tails.values()),
        f"q25 fall / q75 fall per seed: { {s: round(r, 2) for s, r in tails.items()} }",
    )
    # G3
    shares = {}
    for seed in SEEDS:
        early, late = hci[(seed, 1990)], hci[(seed, 2020)]
        plain = statistics.fmean(early.partition(False)) - statistics.fmean(late.partition(False))
        themed = statistics.fmean(early.partition(True)) - statistics.fmean(late.partition(True))
        shares[seed] = (
            round(plain, 3),
            round(themed, 3),
            round(plain / themed, 2) if themed else None,
        )
    hits = sum(1 for plain, themed, _ in shares.values() if themed and plain >= 0.6 * themed)
    report(
        "G3 gain reaches abstracts without philosophy's themes",
        hits >= 4,
        f"(fall on plain abstracts, fall on flavoured abstracts, ratio) per seed: {shares}; "
        f"{hits} of 5 seeds at or above 0.6",
    )
    # G4
    saved: dict[int, Counter[str]] = {1990: Counter(), 2020: Counter()}
    occurrences: dict[int, Counter[str]] = {1990: Counter(), 2020: Counter()}
    for (_seed, decade), models in runs.items():
        held_out = models.held_out["hci"]
        philosophy_bits, counts = models.models["philosophy"].word_bits(held_out)
        own_bits, _ = models.models["hci"].word_bits(held_out)
        for word, count in counts.items():
            saved[decade][word] += philosophy_bits[word] - own_bits[word]
            occurrences[decade][word] += count
    shared = (set(occurrences[1990]) & set(occurrences[2020])) - {UNKNOWN, END}
    word_score = {
        word: (
            saved[1990][word] / occurrences[1990][word]
            - saved[2020][word] / occurrences[2020][word]
        )
        * occurrences[2020][word]
        for word in shared
    }
    positive = sum(value for value in word_score.values() if value > 0)
    top = sorted(word_score, key=word_score.__getitem__, reverse=True)[:50]
    concentration = sum(word_score[word] for word in top) / positive
    report(
        "G4 a small vocabulary does it",
        concentration >= 0.4,
        f"top 50 words hold {concentration:.1%} of the positive score; top 20: {top[:20]}",
    )
    # G5
    sides = {
        seed: (
            round(fall(seed, "median"), 3),
            round(philosophy[(seed, 2020)].median - philosophy[(seed, 1990)].median, 3),
        )
        for seed in SEEDS
    }
    report(
        "G5 one-sided at the document level",
        all(abs(back) < 0.5 * forward for forward, back in sides.values()),
        "(HCI median fall under philosophy, change of philosophy's median under HCI) "
        f"per seed: {sides}",
    )

    print("\nExploratory (unscored): HCI 2020s held-out abstracts, seed 1")
    late = hci[(1, 2020)]
    order = sorted(range(len(late.excess)), key=late.excess.__getitem__)
    for label, indices in (("lowest excess", order[:10]), ("highest excess", order[-10:])):
        print(f"\n  {label}:")
        for index in indices:
            words = " ".join(late.abstracts[index].split()[:25])
            print(f"    {late.excess[index]:6.2f}  {words}")


if __name__ == "__main__":
    main()
