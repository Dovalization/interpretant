"""How much a version of a comparison's flagged drift table moved from the base's (round S1)."""

from __future__ import annotations

import math

import polars as pl
import pytest

from interpretant.drift.stability import flag_turnover, movement, rank_disagreement


def table(words: list[str], ranks: list[float], flagged: set[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {"word": words, "shift_rank": ranks, "candidate": [word in flagged for word in words]}
    )


def test_a_table_does_not_move_from_itself() -> None:
    base = table(["a", "b", "c", "d"], [10.0, 40.0, 70.0, 99.0], {"d"})
    moved = movement(base, base)
    assert moved.flag_turnover == 0.0
    assert moved.rank_disagreement == pytest.approx(0.0)
    assert (moved.entering, moved.leaving) == (0, 0)


def test_flag_turnover_is_one_minus_the_jaccard_of_the_flag_sets() -> None:
    base = table(["a", "b", "c"], [10.0, 50.0, 99.0], {"b", "c"})
    version = table(["a", "b", "c"], [10.0, 99.0, 50.0], {"c", "a"})
    assert flag_turnover(base, version) == pytest.approx(1 - 1 / 3)


def test_a_flagged_word_no_longer_measured_counts_as_turnover() -> None:
    base = table(["a", "b"], [10.0, 99.0], {"b"})
    version = table(["a"], [10.0], set())
    assert flag_turnover(base, version) == 1.0
    assert movement(base, version).leaving == 1


def test_no_flags_on_either_side_is_no_turnover() -> None:
    base = table(["a", "b"], [10.0, 20.0], set())
    assert flag_turnover(base, table(["a", "c"], [10.0, 20.0], set())) == 0.0


def test_rank_disagreement_reverses_to_two_and_needs_three_shared_words() -> None:
    base = table(["a", "b", "c"], [10.0, 50.0, 90.0], set())
    reversed_ranks = table(["a", "b", "c"], [90.0, 50.0, 10.0], set())
    assert rank_disagreement(base, reversed_ranks) == pytest.approx(2.0)
    assert math.isnan(rank_disagreement(base, table(["a", "z"], [1.0, 2.0], set())))
