"""Score round N (direction of movement net of every field's drift) as committed in
docs/predictions_cross_perplexity.md, from the table `interpretant drift direction` writes.
"""

from __future__ import annotations

import statistics

import pandas as pd

from interpretant.drift.direction import present, relative_gain
from interpretant.predictions import report

TABLE = pd.read_parquet("data/drift/direction_assignments.parquet")
SEEDS = sorted(TABLE.seed.unique())
BEHAVIOURAL_SOCIAL = ("psychology", "cogpsych", "neuroscience", "sociology", "communication")
COMPUTING = ("compsci", "ai")


def gains(train: int, test: int, text: str, targets: tuple[str, ...]) -> list[float]:
    return [relative_gain(TABLE, seed, train, test, text, targets) for seed in SEEDS]


def describe(values: list[float]) -> str:
    return (
        f"mean {statistics.mean(values):+.4f}, sd {statistics.stdev(values):.4f}, "
        f"range {min(values):+.4f} to {max(values):+.4f}"
        f" ({'present' if present(values) else 'absent'})"
    )


def n1() -> None:
    results = {train: gains(train, 2020, "hci", BEHAVIOURAL_SOCIAL) for train in (1980, 1990)}
    detail = "\n  ".join(f"{train}s classifier: {describe(v)}" for train, v in results.items())
    report(
        "N1 HCI towards behavioural/social, net of drift",
        all(present(v) for v in results.values()),
        detail,
    )


def n2() -> None:
    values = gains(1990, 2000, "hci", BEHAVIOURAL_SOCIAL)
    report(
        "N2 under way by the 2000s", present(values), f"1990s classifier, 2000s: {describe(values)}"
    )


def n3() -> None:
    results = {train: gains(train, 2020, "hci", COMPUTING) for train in (1980, 1990)}
    detail = "\n  ".join(f"{train}s classifier: {describe(v)}" for train, v in results.items())
    report(
        "N3 not towards computing",
        not any(present(v) for v in results.values()),
        detail,
    )


def n4() -> None:
    values = gains(1980, 2020, "psychology", ("hci",))
    report(
        "N4 psychology's half of the meeting",
        present(values),
        f"1980s classifier: {describe(values)}",
    )


def n5() -> None:
    values = gains(1990, 2000, "psychology", ("hci",))
    report(
        "N5 psychology moved later",
        not present(values),
        f"1990s classifier, 2000s: {describe(values)}",
    )


if __name__ == "__main__":
    n1()
    n2()
    n3()
    n4()
    n5()
