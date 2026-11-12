"""Tests for drift on the field × decade grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from gensim.models import KeyedVectors

from interpretant.drift.grid import (
    CellModel,
    add_noise_floor,
    cell_neighbours,
    compare_cells,
    cross_field_pairs,
    measure_grid,
    within_field_pairs,
)

RNG = np.random.default_rng(0)
BASE = {f"w{n}": RNG.normal(size=8) for n in range(60)}


def make_vectors(vectors: dict[str, np.ndarray], counts: dict[str, int]) -> KeyedVectors:
    keyed = KeyedVectors(vector_size=8)
    keyed.add_vectors(list(vectors), np.array(list(vectors.values())))
    for word, count in counts.items():
        keyed.set_vecattr(word, "count", count)
    return keyed


def test_compare_cells_measures_only_frequent_shared_words_and_sees_movement() -> None:
    later = dict(BASE)
    later["w0"] = -BASE["w0"]
    counts = dict.fromkeys(BASE, 500)
    counts_later = {**counts, "w59": 20}
    pair = compare_cells(
        make_vectors(BASE, counts), make_vectors(later, counts_later), neighbours=5
    )

    assert "w59" not in set(pair["word"])
    drift = dict(zip(pair["word"], pair["cosine_drift"], strict=True))
    assert drift["w0"] > 1.5
    assert drift["w1"] < 1e-6


def test_noise_floor_ranks_within_frequency_bins_and_needs_both_measures() -> None:
    words = [f"w{n}" for n in range(400)]
    shift = np.tile(np.linspace(0, 1, 100), 4)
    shift[399] = 0.0
    pair = pl.DataFrame(
        {
            "word": words,
            "count_a": np.repeat([200, 2_000, 20_000, 200_000], 100),
            "count_b": np.repeat([200, 2_000, 20_000, 200_000], 100),
            "cosine_drift": np.tile(np.linspace(0, 1, 100), 4),
            "neighbour_shift": shift,
        }
    )

    ranked = add_noise_floor(pair)

    assert ranked["frequency_bin"].n_unique() == 4
    assert set(ranked.filter(pl.col("candidate"))["word"]) == {
        "w95",
        "w96",
        "w97",
        "w98",
        "w99",
        "w195",
        "w196",
        "w197",
        "w198",
        "w199",
        "w295",
        "w296",
        "w297",
        "w298",
        "w299",
        "w395",
        "w396",
        "w397",
        "w398",
    }


def test_pairs_follow_consecutive_decades_and_same_decade_fields() -> None:
    models = [
        CellModel("openalex", "hci", 1990, Path("a")),
        CellModel("openalex", "hci", 2000, Path("b")),
        CellModel("openalex", "hci", 2020, Path("c")),
        CellModel("openalex", "philosophy", 2000, Path("d")),
    ]

    within = [(a.decade, b.decade) for a, b in within_field_pairs(models)]
    across = [(a.name, b.name, a.decade) for a, b in cross_field_pairs(models)]

    assert within == [(1990, 2000)]
    assert across == [("hci", "philosophy", 2000)]


def test_cell_neighbours_come_from_the_pool_and_need_a_measurable_word() -> None:
    vectors = dict(BASE)
    vectors["near"] = BASE["w0"] + 0.01
    vectors["rare"] = BASE["w0"] + 0.02
    counts = {**dict.fromkeys(vectors, 500), "rare": 10}
    keyed = make_vectors(vectors, counts)

    neighbours = cell_neighbours(keyed, "w0", k=3)

    assert neighbours[0] == "near"
    assert "rare" not in neighbours
    assert "w0" not in neighbours
    assert len(neighbours) == 3
    assert cell_neighbours(keyed, "rare") == []


def test_measure_grid_compares_trajectories_and_fields_and_flags_words(tmp_path: Path) -> None:
    counts = dict.fromkeys(BASE, 500)
    for field in ("ai", "hci"):
        for decade in (1990, 2000):
            path = tmp_path / "openalex" / f"{field}_{decade}.kv"
            path.parent.mkdir(parents=True, exist_ok=True)
            make_vectors(BASE, counts).save(str(path))

    tables = measure_grid(tmp_path, targets=["w0"], controls=["w1", "w2"], neighbours=5)

    within, cross = tables["within_field"], tables["cross_field"]
    assert set(within.select("field_a", "decade_a", "decade_b").iter_rows()) == {
        ("ai", 1990, 2000),
        ("hci", 1990, 2000),
    }
    assert set(cross.select("field_a", "field_b", "decade_a").iter_rows()) == {
        ("ai", "hci", 1990),
        ("ai", "hci", 2000),
    }
    assert len(within) == len(cross) == 2 * len(BASE)
    assert within["is_target"].sum() == 2
    assert within["is_control"].sum() == 4


def test_measure_grid_needs_pairs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no within_field pairs"):
        measure_grid(tmp_path, targets=[], controls=[])
