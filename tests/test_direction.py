"""Calibration of the relative gain on synthetic fields with planted effects.

Six fields share most of their words, so a classifier sends about a fifth of each field's
held-out abstracts elsewhere, as in the corpus. By the 2020s every field has replaced 30% of its
own words with new ones (general drift) and borrowed 20% from HCI (a target everyone moves
towards); HCI alone has also borrowed 20% from psychology (the planted move). The relative gain
must find the planted move and nothing else; the raw gain is fooled by the shared target.
"""

from __future__ import annotations

import random

import pandas as pd
import pytest

from interpretant.drift.direction import assignment_runs, gain, present, relative_gain

FIELDS = ("hci", "psychology", "sociology", "compsci", "philosophy", "linguistics")
OWN = {field: [f"{field}{n}" for n in range(40)] for field in FIELDS}
SHARED = [f"shared{n}" for n in range(200)]
NEW = [f"new{n}" for n in range(100)]
SEEDS = (1, 2, 3, 4, 5)


def word(rng: random.Random, field: str, decade: int) -> str:
    if rng.random() < 0.95:
        return rng.choice(SHARED)
    draw = rng.random()
    if decade == 2020 and draw < 0.3:
        return rng.choice(NEW)
    if decade == 2020 and field != "hci" and draw < 0.5:
        return rng.choice(OWN["hci"])
    if decade == 2020 and field == "hci" and draw < 0.5:
        return rng.choice(OWN["psychology"])
    return rng.choice(OWN[field])


@pytest.fixture(scope="module")
def table(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    root = tmp_path_factory.mktemp("texts")
    for decade in (1990, 2020):
        for field in FIELDS:
            rng = random.Random(f"{field}{decade}")
            lines = [
                " ".join(word(rng, field, decade) for _ in range(30)) + f" id{field}{decade}n{n}"
                for n in range(1_250)
            ]
            (root / f"{field}_{decade}.txt").write_text("\n".join(lines), encoding="utf-8")
    return assignment_runs(root, [1990], [2020], SEEDS, per_field=1_000)


def values(table: pd.DataFrame, measure: object, text: str, target: str) -> list[float]:
    return [measure(table, seed, 1990, 2020, text, (target,)) for seed in SEEDS]  # type: ignore[operator]


def test_the_planted_move_is_present(table: pd.DataFrame) -> None:
    assert present(values(table, relative_gain, "hci", "psychology"))


def test_a_target_everyone_drifts_towards_fools_the_raw_gain_only(table: pd.DataFrame) -> None:
    assert present(values(table, gain, "psychology", "hci"))
    assert not present(values(table, relative_gain, "psychology", "hci"))


def test_general_drift_moves_no_field_towards_another(table: pd.DataFrame) -> None:
    for field in ("sociology", "philosophy", "linguistics"):
        assert not present(values(table, relative_gain, field, "psychology"))


def test_presence_needs_every_seed_positive_and_two_standard_deviations() -> None:
    assert present([0.10, 0.11, 0.09, 0.10, 0.12])
    assert not present([0.10, 0.11, -0.01, 0.10, 0.12])
    assert not present([0.01, 0.20, 0.02, 0.15, 0.01])
