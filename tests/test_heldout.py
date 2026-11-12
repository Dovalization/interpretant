"""Tests for the held-out check on cell models."""

from __future__ import annotations

import random
from pathlib import Path

import polars as pl
import pytest

from interpretant.drift.heldout import cell_gap, heldout_runs

WORDS = [f"w{n}" for n in range(300)]


def documents(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return [" ".join(rng.choices(WORDS, k=30)) for _ in range(count)]


def test_a_small_cell_has_a_larger_gap_than_a_large_one() -> None:
    small = cell_gap(documents(40, 1), seed=1)
    large = cell_gap(documents(4_000, 1), seed=1)

    assert small.held_out_documents == 4
    assert large.held_out_documents == 400
    assert small.gap > large.gap
    assert large.gap < 1.0


def test_the_training_part_respects_the_token_cap() -> None:
    gap = cell_gap(documents(1_000, 2), seed=1, max_training_tokens=3_000)

    assert gap.training_tokens <= 3_000 + 30


def test_a_cell_needs_two_documents() -> None:
    with pytest.raises(ValueError, match="at least two documents"):
        cell_gap(["only one"], seed=1)


def test_runs_score_every_cell_for_every_seed(tmp_path: Path) -> None:
    for stem, count in (("hci_1990", 60), ("hci_2000", 600)):
        path = tmp_path / "openalex" / "texts" / f"{stem}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(documents(count, 3)), encoding="utf-8")

    table = heldout_runs(tmp_path, ("openalex",), seeds=[1, 2], workers=1)

    assert len(table) == 4
    assert set(table["decade"]) == {1990, 2000}
    gaps = dict(table.group_by("decade").agg(pl.col("gap").mean()).iter_rows())
    assert gaps[1990] > gaps[2000]
