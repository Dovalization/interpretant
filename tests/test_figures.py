"""Tests for the README figures, on synthetic inputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from interpretant.visualization.figures import (
    NeighbourColumn,
    corpus_grid,
    distinct_neighbours,
    excess_by_decade,
    neighbour_panel,
    same_family,
)


def texts(figure: plt.Figure) -> set[str]:
    return {text.get_text() for axes in figure.axes for text in axes.texts}


@pytest.fixture(autouse=True)
def close_figures() -> None:
    yield
    plt.close("all")


def test_neighbour_panel_lists_words_per_cell_and_marks_the_shared_ones(tmp_path: Path) -> None:
    columns = [
        NeighbourColumn("philosophy", 2010, 1_757, ["corporeality", "body", "temporality"]),
        NeighbourColumn("hci", 2010, 1_018, ["ownership", "body", "presence"]),
        NeighbourColumn("hci", 2020, 2_809, ["illusion", "presence"]),
    ]

    figure = neighbour_panel("embodiment", columns, title="Embodiment", k=3)
    labels = texts(figure)

    assert {"Philosophy", "HCI", "2010s", "2020s", "corporeality", "illusion"} <= labels
    assert "1,018 occurrences" in labels
    shared = [text for axes in figure.axes for text in axes.texts if text.get_text() == "body"]
    unique = [text for axes in figure.axes for text in axes.texts if text.get_text() == "illusion"]
    assert all(text.get_color() != unique[0].get_color() for text in shared)
    figure.savefig(tmp_path / "panel.png")
    assert (tmp_path / "panel.png").stat().st_size > 0


def test_excess_by_decade_draws_one_line_and_one_label_per_pair() -> None:
    rows = [
        {"seed": seed, "decade": decade, "model_field": model, "text_field": text, "excess": value}
        for seed in (1, 2)
        for decade in (1990, 2000)
        for model, text, value in (("philosophy", "hci", 1.7), ("hci", "philosophy", 1.2))
    ]

    figure = excess_by_decade(
        pl.DataFrame(rows), [("philosophy", "hci"), ("hci", "philosophy")], title="Excess"
    )
    axes = figure.axes[0]

    assert len(axes.lines) == 2
    assert {"Philosophy → HCI", "HCI → Philosophy"} <= texts(figure)
    assert [label.get_text() for label in axes.get_xticklabels()] == ["1990s", "2000s"]


def test_corpus_grid_shows_counts_and_marks_empty_cells() -> None:
    cells = pl.DataFrame(
        {
            "source": ["ai", "ai", "hci"],
            "decade": [1990, 2000, 2000],
            "documents": [120, 10_000, 3_000],
        }
    )

    figure = corpus_grid(cells, title="Grid", fields=["hci", "ai"])
    axes = figure.axes[0]

    assert [label.get_text() for label in axes.get_yticklabels()] == ["HCI", "AI"]
    assert {"120", "10,000", "3,000", "–"} <= texts(figure)
    image = axes.images[0].get_array()
    assert np.ma.is_masked(image[0, 0])


def test_variants_of_the_word_and_of_earlier_neighbours_are_collapsed() -> None:
    assert same_family("structure", "structures")
    assert same_family("conceptualisation", "conceptualization")
    assert same_family("learn", "learned")
    assert not same_family("self-location", "self-perception")
    assert not same_family("sense", "sensorial")

    neighbours = ["representing", "structures", "knowledge", "structure", "declarative", "learn"]

    assert distinct_neighbours("representation", neighbours, k=3) == [
        "structures",
        "knowledge",
        "declarative",
    ]
