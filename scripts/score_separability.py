"""Score round S (field separability by classifier) as committed in
docs/predictions_cross_perplexity.md, from the tables `interpretant drift separability` writes.
"""

from __future__ import annotations

import statistics

import pandas as pd

from interpretant.predictions import report

PAIRS = pd.read_parquet("data/drift/separability_pairs.parquet")
ASSIGNMENTS = pd.read_parquet("data/drift/separability_assignments.parquet")
SEEDS = sorted(PAIRS.seed.unique())
BEHAVIOURAL_SOCIAL = ("psychology", "cogpsych", "neuroscience", "sociology", "communication")
COMPUTING = ("compsci", "ai")
PARENT_SUBFIELD = {("ai", "compsci"), ("compsci", "hci"), ("cogpsych", "psychology")}


def accuracies(decade: int, field_a: str, field_b: str) -> list[float]:
    """One accuracy per seed for the pair, in seed order (pairs are stored alphabetically)."""
    first, second = sorted((field_a, field_b))
    rows = PAIRS[(PAIRS.decade == decade) & (PAIRS.field_a == first) & (PAIRS.field_b == second)]
    return list(rows.sort_values("seed").accuracy)


def mean_by_pair(decade: int) -> pd.Series:
    rows = PAIRS[PAIRS.decade == decade]
    return rows.groupby(["field_a", "field_b"]).accuracy.mean().sort_values()


def share(seed: int, train: int, test: int, text: str, assigned: tuple[str, ...]) -> float:
    rows = ASSIGNMENTS[
        (ASSIGNMENTS.seed == seed)
        & (ASSIGNMENTS.train_decade == train)
        & (ASSIGNMENTS.test_decade == test)
        & (ASSIGNMENTS.text_field == text)
        & (ASSIGNMENTS.assigned_field.isin(assigned))
    ]
    return float(rows.share.sum())


def s0() -> None:
    parts, ok = [], True
    for decade in (1990, 2020):
        means = mean_by_pair(decade)
        bottom = list(means.index[:5])
        lowest = means.iloc[0]
        missing = [pair for pair in PARENT_SUBFIELD if pair not in bottom]
        ok = ok and lowest >= 0.75 and not missing
        parts.append(
            f"{decade}s lowest {means.index[0]} {lowest:.3f}; bottom five "
            + ", ".join(f"{a}–{b} {means[(a, b)]:.3f}" for a, b in bottom)
            + (f"; parent–subfield outside: {missing}" if missing else "")
        )
    report("S0 the classifier sees the fields", ok, "\n  ".join(parts))


def s1() -> None:
    lines, hits = [], 0
    for field in BEHAVIOURAL_SOCIAL:
        early, late = accuracies(1990, "hci", field), accuracies(2020, "hci", field)
        hit = max(late) < min(early)
        hits += hit
        lines.append(
            f"HCI–{field}: 1990s {min(early):.3f}–{max(early):.3f}, "
            f"2020s {min(late):.3f}–{max(late):.3f}{' ✓' if hit else ''}"
        )
    detail = f"{hits}/5\n  " + "\n  ".join(lines)
    report("S1 HCI harder to tell from behavioural/social", hits >= 4, detail)


def s2() -> None:
    lines, ok = [], True
    for field in COMPUTING:
        early = statistics.mean(accuracies(1990, "hci", field))
        late = statistics.mean(accuracies(2020, "hci", field))
        ok = ok and late >= early - 0.01
        lines.append(f"HCI–{field}: {early:.3f} → {late:.3f}")
    report("S2 but not from computing", ok, "; ".join(lines))


def s3() -> None:
    social = [
        share(seed, 2020, 2020, "hci", BEHAVIOURAL_SOCIAL)
        > share(seed, 1990, 1990, "hci", BEHAVIOURAL_SOCIAL)
        for seed in SEEDS
    ]
    computing = [
        share(seed, 2020, 2020, "hci", COMPUTING) < share(seed, 1990, 1990, "hci", COMPUTING)
        for seed in SEEDS
    ]
    def change(seed: int, group: tuple[str, ...]) -> str:
        early, late = share(seed, 1990, 1990, "hci", group), share(seed, 2020, 2020, "hci", group)
        return f"{early:.3f} → {late:.3f}"

    detail = "; ".join(
        f"seed {seed}: social {change(seed, BEHAVIOURAL_SOCIAL)}, "
        f"computing {change(seed, COMPUTING)}"
        for seed in SEEDS
    )
    report(
        "S3 each decade's classifier sends HCI elsewhere",
        sum(social) >= 4 and sum(computing) >= 4,
        f"social rises in {sum(social)}/5, computing falls in {sum(computing)}/5\n  {detail}",
    )


def s4() -> None:
    hci = [
        share(seed, 1990, 2020, "hci", BEHAVIOURAL_SOCIAL)
        > share(seed, 1990, 1990, "hci", BEHAVIOURAL_SOCIAL)
        for seed in SEEDS
    ]
    towards = {
        field: [
            share(seed, 1990, 2020, field, ("hci",)) <= share(seed, 1990, 1990, field, ("hci",))
            for seed in SEEDS
        ]
        for field in ("psychology", "sociology")
    }
    ok = sum(hci) >= 4 and all(sum(values) >= 4 for values in towards.values())
    detail = [
        "HCI to behavioural/social, 1990s classifier: "
        + ", ".join(
            f"{share(seed, 1990, 1990, 'hci', BEHAVIOURAL_SOCIAL):.3f}→"
            f"{share(seed, 1990, 2020, 'hci', BEHAVIOURAL_SOCIAL):.3f}"
            for seed in SEEDS
        )
        + f" (rises in {sum(hci)}/5)"
    ] + [
        f"{field} to HCI: "
        + ", ".join(
            f"{share(seed, 1990, 1990, field, ('hci',)):.3f}→"
            f"{share(seed, 1990, 2020, field, ('hci',)):.3f}"
            for seed in SEEDS
        )
        + f" (no rise in {sum(values)}/5)"
        for field, values in towards.items()
    ]
    report("S4 HCI moved; nobody moved towards HCI", ok, "\n  ".join(detail))


def s5() -> None:
    means = {
        decade: [
            PAIRS[(PAIRS.decade == decade) & (PAIRS.seed == seed)].accuracy.mean() for seed in SEEDS
        ]
        for decade in (1980, 2010)
    }
    ok = all(late < early for early, late in zip(means[1980], means[2010], strict=True))
    detail = ", ".join(
        f"{early:.4f}→{late:.4f}" for early, late in zip(means[1980], means[2010], strict=True)
    )
    report("S5 fields converged", ok, f"mean pairwise accuracy 1980s→2010s by seed: {detail}")


if __name__ == "__main__":
    s0()
    s1()
    s2()
    s3()
    s4()
    s5()
