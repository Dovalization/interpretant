"""Tests for the web export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors

from interpretant.export import NeighbourIndex, parse_predictions, read_tiers

REAL_PREDICTIONS = [Path("docs/predictions.md"), Path("docs/predictions_cross_perplexity.md")]


def test_tiers_follow_the_section_headers(tmp_path: Path) -> None:
    words = tmp_path / "words.txt"
    words.write_text(
        "# --- Core: measurable ---\nmind\n# --- Late arrivals: each series ---\nuser # note\n",
        encoding="utf-8",
    )

    assert read_tiers(words) == {"mind": "core", "user": "late arrivals"}


def test_neighbour_index_excludes_the_word_and_rare_words() -> None:
    rng = np.random.default_rng(1)
    names = [f"w{n}" for n in range(30)] + ["near", "rare"]
    base = rng.normal(size=(30, 6))
    vectors = np.vstack([base, base[0] + 0.01, base[0] + 0.02])
    keyed = KeyedVectors(vector_size=6)
    keyed.add_vectors(names, vectors)
    for name in names:
        keyed.set_vecattr(name, "count", 10 if name == "rare" else 500)

    index = NeighbourIndex.build(keyed)
    neighbours = index.neighbours("w0", k=3)

    assert neighbours[0] == "near"
    assert "rare" not in neighbours and "w0" not in neighbours
    assert index.neighbours("rare") == []


def test_predictions_parser_reads_titles_and_verdicts(tmp_path: Path) -> None:
    text = (
        "## Predictions\n\n"
        "**P1 — Method sanity.** In each table...\n\n"
        "**P3 — *representation* in AI moves** (main grid). Its...\n\n"
        "| P1 | **Falsified** (second criterion) | Control candidates 0.7% |\n"
        "| P3 | **Passed** | 1980s neighbours include *knowledge* |\n"
    )
    path = tmp_path / "predictions.md"
    path.write_text(text, encoding="utf-8")

    parsed = parse_predictions([path])

    assert parsed["rounds"] == [{"id": "P", "title": "Round 1", "subtitle": "the trained models"}]
    assert parsed["predictions"] == [
        {
            "id": "P1",
            "round": "P",
            "title": "Method sanity",
            "verdict": "falsified",
            "note": "second criterion",
            "numbers": "Control candidates 0.7%",
            "file": "predictions.md",
        },
        {
            "id": "P3",
            "round": "P",
            "title": "*representation* in AI moves",
            "verdict": "passed",
            "note": "",
            "numbers": "1980s neighbours include *knowledge*",
            "file": "predictions.md",
        },
    ]


def test_every_scored_prediction_in_the_real_files_has_a_title() -> None:
    parsed = parse_predictions(REAL_PREDICTIONS)
    scored = [prediction for prediction in parsed["predictions"] if "verdict" in prediction]

    assert len(scored) >= 45
    assert all(prediction["title"] for prediction in scored)
    assert {prediction["round"] for prediction in parsed["predictions"]} >= {
        "P",
        "R",
        "T",
        "C",
        "D",
        "F",
        "H",
        "L",
    }
