"""Score round V (round S's classifier without 29 technology terms) as committed in
docs/predictions_cross_perplexity.md, from the tables written by

    interpretant drift separability --decade 1990 --decade 2020 \\
        --remove-terms data/technology_terms.txt --output-dir data/drift/without_technology
"""

from __future__ import annotations

import statistics

import pandas as pd

from interpretant.predictions import report

ROOT = "data/drift/without_technology"
PAIRS = pd.read_parquet(f"{ROOT}/separability_pairs.parquet")
ASSIGNMENTS = pd.read_parquet(f"{ROOT}/separability_assignments.parquet")
SEEDS = sorted(PAIRS.seed.unique())
FIELDS = sorted(ASSIGNMENTS.text_field.unique())
BEHAVIOURAL_SOCIAL = ("psychology", "cogpsych", "neuroscience", "sociology", "communication")


def share(seed: int, train: int, test: int, text: str, assigned: tuple[str, ...]) -> float:
    rows = ASSIGNMENTS[
        (ASSIGNMENTS.seed == seed)
        & (ASSIGNMENTS.train_decade == train)
        & (ASSIGNMENTS.test_decade == test)
        & (ASSIGNMENTS.text_field == text)
        & (ASSIGNMENTS.assigned_field.isin(assigned))
    ]
    return float(rows.share.sum())


def drift_towards_hci(seed: int, field: str) -> float:
    """Share of the field's 2020s abstracts the 1990s classifier sends to HCI, minus its 1990s."""
    return share(seed, 1990, 2020, field, ("hci",)) - share(seed, 1990, 1990, field, ("hci",))


def v0() -> None:
    parts, ok = [], True
    for decade in (1990, 2020):
        means = PAIRS[PAIRS.decade == decade].groupby(["field_a", "field_b"]).accuracy.mean()
        lowest = means.idxmin()
        ok = ok and means.min() >= 0.75
        parts.append(f"{decade}s lowest {lowest[0]}–{lowest[1]} {means.min():.3f}")
    report("V0 still a working classifier", ok, "; ".join(parts))


def v1() -> None:
    lines, ok = [], True
    for field in ("psychology", "sociology"):
        drifts = [drift_towards_hci(seed, field) for seed in SEEDS]
        held = sum(drift <= 0 for drift in drifts)
        ok = ok and held >= 4
        lines.append(
            f"{field}: drift by seed "
            + ", ".join(f"{d:+.3f}" for d in drifts)
            + f" (≤ 0 in {held}/5)"
        )
    report("V1 psychology and sociology stop drifting towards HCI", ok, "\n  ".join(lines))


def v2() -> None:
    others = [field for field in FIELDS if field != "hci"]
    means = {
        field: statistics.mean(drift_towards_hci(seed, field) for seed in SEEDS) for field in others
    }
    positive = [field for field, drift in means.items() if drift > 0]
    detail = ", ".join(
        f"{field} {drift:+.3f}" for field, drift in sorted(means.items(), key=lambda item: -item[1])
    )
    report(
        "V2 the drift towards HCI shrinks across the board",
        len(positive) <= 5,
        f"{len(positive)} of {len(others)} positive\n  {detail}",
    )


def v3() -> None:
    changes = [
        (
            share(seed, 1990, 1990, "hci", BEHAVIOURAL_SOCIAL),
            share(seed, 1990, 2020, "hci", BEHAVIOURAL_SOCIAL),
        )
        for seed in SEEDS
    ]
    rises = sum(late > early for early, late in changes)
    detail = ", ".join(f"{early:.3f}→{late:.3f}" for early, late in changes)
    report(
        "V3 HCI's move survives without the technology words",
        rises >= 4,
        f"rises in {rises}/5: {detail}",
    )


if __name__ == "__main__":
    v0()
    v1()
    v2()
    v3()
