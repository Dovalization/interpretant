"""Calibration of the direction measures on synthetic fields with planted effects.

Six fields share most of their words, so a classifier sends about a fifth of each field's
held-out abstracts elsewhere, as in the corpus. By the 2020s every field has replaced 30% of its
own words with new ones (general drift) and borrowed 20% from HCI (a target everyone moves
towards); HCI alone has also borrowed 20% from psychology (the planted move). The relative gain
and the contrast must find the planted move and nothing else; the raw gain is fooled by the
shared target. A second corpus has no planted move, but HCI replaces 60% of its words: the
relative gain reports a move there, and the contrast does not.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest

from interpretant.drift.direction import (
    ClassifierCosts,
    FieldCosts,
    LanguageModelCosts,
    assignment_runs,
    contrast,
    cost_matrix,
    frame_runs,
    gain,
    planted_runs,
    present,
    relative_gain,
    relative_movement,
)
from interpretant.drift.separability import decade_samples, split

FIELDS = ("hci", "psychology", "sociology", "compsci", "philosophy", "linguistics")
OWN = {field: [f"{field}{n}" for n in range(40)] for field in FIELDS}
SHARED = [f"shared{n}" for n in range(200)]
NEW = [f"new{n}" for n in range(100)]
SEEDS = (1, 2, 3, 4, 5)
INSTRUMENTS: dict[str, Callable[[dict[str, list[str]]], FieldCosts]] = {
    "classifier": ClassifierCosts,
    "language models": LanguageModelCosts,
}


def word(  # noqa: PLR0911
    rng: random.Random,
    field: str,
    decade: int,
    planted: float = 0.2,
    towards_hci: float = 0.2,
    hci_new: float = 0.3,
) -> str:
    if rng.random() < 0.95:
        return rng.choice(SHARED)
    draw = rng.random()
    new = hci_new if field == "hci" else 0.3
    if decade == 2020 and draw < new:
        return rng.choice(NEW)
    if decade == 2020 and field != "hci" and draw < new + towards_hci:
        return rng.choice(OWN["hci"])
    if decade == 2020 and field == "hci" and draw < new + planted:
        return rng.choice(OWN["psychology"])
    return rng.choice(OWN[field])


def write_fields(root: Path, lines: int, **setting: float) -> Path:
    for decade in (1990, 2020):
        for field in FIELDS:
            rng = random.Random(f"{field}{decade}")
            text = [
                " ".join(word(rng, field, decade, **setting) for _ in range(30))
                + f" id{field}{decade}n{n}"
                for n in range(lines)
            ]
            (root / f"{field}_{decade}.txt").write_text("\n".join(text), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def texts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_fields(tmp_path_factory.mktemp("texts"), 1_250)


@pytest.fixture(scope="module")
def changing(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_fields(
        tmp_path_factory.mktemp("changing"), 1_250, planted=0.0, towards_hci=0.0, hci_new=0.6
    )


@pytest.fixture(scope="module")
def table(texts: Path) -> pl.DataFrame:
    return assignment_runs(texts, [1990], [2020], SEEDS, per_field=1_000)


@pytest.fixture(scope="module")
def costs(texts: Path) -> dict[str, pl.DataFrame]:
    return {
        name: frame_runs(texts, 1990, 2020, SEEDS, fit=fit, per_field=1_000)
        for name, fit in INSTRUMENTS.items()
    }


def values(table: pl.DataFrame, measure: object, text: str, target: str) -> list[float]:
    return [measure(table, seed, 1990, 2020, text, (target,)) for seed in SEEDS]  # type: ignore[operator]


def movements(
    table: pl.DataFrame, text: str, towards: tuple[str, ...], over: tuple[str, ...]
) -> list[float]:
    def matrix(seed: int, decade: int) -> dict[str, dict[str, float]]:
        return cost_matrix(table.filter(pl.col("seed") == seed, pl.col("test_decade") == decade))

    return [
        relative_movement(matrix(seed, 1990), matrix(seed, 2020), text, towards, over)
        for seed in SEEDS
    ]


def test_the_planted_move_is_present(table: pl.DataFrame) -> None:
    assert present(values(table, relative_gain, "hci", "psychology"))


def test_a_target_everyone_drifts_towards_fools_the_raw_gain_only(table: pl.DataFrame) -> None:
    assert present(values(table, gain, "psychology", "hci"))
    assert not present(values(table, relative_gain, "psychology", "hci"))


def test_general_drift_moves_no_field_towards_another(table: pl.DataFrame) -> None:
    for field in ("sociology", "philosophy", "linguistics"):
        assert not present(values(table, relative_gain, field, "psychology"))


def test_presence_needs_every_seed_positive_and_two_standard_deviations() -> None:
    assert present([0.10, 0.11, 0.09, 0.10, 0.12])
    assert not present([0.10, 0.11, -0.01, 0.10, 0.12])
    assert not present([0.01, 0.20, 0.02, 0.15, 0.01])


@pytest.mark.parametrize("instrument", INSTRUMENTS)
def test_the_contrast_finds_the_planted_move(
    costs: dict[str, pl.DataFrame], instrument: str
) -> None:
    assert present(movements(costs[instrument], "hci", ("psychology",), ("compsci",)))


@pytest.mark.parametrize("instrument", INSTRUMENTS)
def test_the_contrast_ignores_a_target_everyone_drifts_towards(
    costs: dict[str, pl.DataFrame], instrument: str
) -> None:
    for towards, over in ((("hci",), ("compsci",)), (("compsci",), ("hci",))):
        assert not present(movements(costs[instrument], "psychology", towards, over))


@pytest.mark.parametrize("instrument", INSTRUMENTS)
def test_the_contrast_sees_no_move_in_general_drift(
    costs: dict[str, pl.DataFrame], instrument: str
) -> None:
    for text, towards, over in (
        ("sociology", ("psychology",), ("compsci",)),
        ("sociology", ("compsci",), ("psychology",)),
        ("linguistics", ("psychology",), ("philosophy",)),
        ("linguistics", ("philosophy",), ("psychology",)),
    ):
        assert not present(movements(costs[instrument], text, towards, over))


def test_a_field_changing_more_fools_the_relative_gain_but_not_the_contrast(
    changing: Path,
) -> None:
    group = ("psychology", "sociology")
    gains = assignment_runs(changing, [1990], [2020], SEEDS, per_field=1_000)
    scored = frame_runs(changing, 1990, 2020, SEEDS, per_field=1_000)

    assert present([relative_gain(gains, seed, 1990, 2020, "hci", group) for seed in SEEDS])
    assert not present(movements(scored, "hci", group, ("compsci",)))
    assert not present(movements(scored, "hci", ("compsci",), group))


def test_the_contrast_leaves_the_text_field_out_of_both_groups() -> None:
    costs = {
        "hci": {"hci": 1.0, "psychology": 3.0, "sociology": 5.0, "compsci": 2.0},
        "psychology": {"hci": 4.0, "psychology": 1.0, "sociology": 2.0, "compsci": 6.0},
    }

    assert contrast(costs, "hci", ("hci", "psychology", "sociology"), ("compsci",)) == 2.0
    assert contrast(costs, "psychology", ("psychology", "sociology"), ("hci", "compsci")) == -3.0


def test_relative_movement_nets_the_median_of_fields_that_keep_both_groups() -> None:
    def costs(hci: float, others: tuple[float, float, float]) -> dict[str, dict[str, float]]:
        rows = {"hci": hci, "philosophy": others[0], "linguistics": others[1], "ai": others[2]}
        matrix = {text: {"psychology": cost, "compsci": 0.0} for text, cost in rows.items()}
        matrix["compsci"] = {"psychology": 9.0, "compsci": 0.0}
        return matrix

    early, late = costs(3.0, (3.0, 3.0, 3.0)), costs(1.0, (2.0, 2.5, 0.0))

    assert relative_movement(early, late, "hci", ("psychology",), ("compsci",)) == 1.0


@pytest.mark.parametrize("instrument", INSTRUMENTS)
def test_a_move_planted_in_one_decade_is_found_and_the_unplanted_halves_show_none(
    texts: Path, instrument: str
) -> None:
    table = planted_runs(
        texts,
        1990,
        SEEDS,
        INSTRUMENTS[instrument],
        ["psychology", "compsci"],
        planted_share=0.4,
        per_field=1_000,
    )

    def found(cell: str, towards: tuple[str, ...], over: tuple[str, ...]) -> bool:
        rows = table.filter(pl.col("cell") == cell)
        return present(
            [
                relative_movement(
                    cost_matrix(rows.filter(pl.col("seed") == seed, pl.col("half") == "early")),
                    cost_matrix(rows.filter(pl.col("seed") == seed, pl.col("half") == "late")),
                    "hci",
                    towards,
                    over,
                )
                for seed in SEEDS
            ]
        )

    assert found("psychology", ("psychology",), ("compsci",))
    assert found("compsci", ("compsci",), ("psychology",))
    assert not found("none", ("psychology",), ("compsci",))
    assert not found("none", ("compsci",), ("psychology",))


def test_frame_runs_replace_only_the_field_text_in_the_stable_cell(tmp_path: Path) -> None:
    write_fields(tmp_path, 150)
    train, _ = split(decade_samples(tmp_path, 1990, 1, 100))
    psychology_line = (tmp_path / "psychology_1990.txt").read_text().splitlines()[0]
    stable = {1990: [*train["hci"], psychology_line], 2020: [psychology_line]}

    table = frame_runs(tmp_path, 1990, 2020, [1], stable=stable, per_field=100)

    whole, kept = table.filter(pl.col("cell") == "whole"), table.filter(pl.col("cell") == "stable")
    columns = ["test_decade", "text_field", "model_field", "cost"]
    others = whole.filter(pl.col("text_field") != "hci").select(columns)
    assert others.equals(kept.filter(pl.col("text_field") != "hci").select(columns))
    hci = kept.filter(pl.col("text_field") == "hci")
    stable_hci = {
        decade: hci.filter(pl.col("test_decade") == decade)["cost"].to_list()
        for decade in (1990, 2020)
    }
    psychology = kept.filter(pl.col("text_field") == "psychology", pl.col("test_decade") == 1990)
    assert stable_hci[1990] == stable_hci[2020]
    assert len(stable_hci[1990]) == len(FIELDS)
    assert psychology["model_field"].to_list() == sorted(FIELDS)


def test_the_language_models_train_on_equal_budgets_and_send_each_field_home() -> None:
    rng = random.Random(1)
    train = {
        field: [" ".join(rng.choices(OWN[field] + SHARED[:20], k=30)) for _ in range(size)]
        for field, size in (("hci", 200), ("psychology", 50))
    }
    instrument = LanguageModelCosts(train)

    held_out = [" ".join(rng.choices(OWN["psychology"] + SHARED[:20], k=30)) for _ in range(20)]
    costs = instrument.mean_costs(held_out)

    assert costs["psychology"] < costs["hci"]
    trained = {field: lm.context_counts.total() for field, lm in instrument.models.items()}
    assert abs(trained["hci"] - trained["psychology"]) <= 32
