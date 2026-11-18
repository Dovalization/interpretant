"""End-to-end smoke test: the grid commands on a synthetic two-field, two-decade grid.

The cells are written as the grid commands read them, already lemmatised and without a phrase
step. Each command reads what the previous one wrote, as in a real run, so a broken hand-off
between commands fails here even when every library function passes its own tests.
"""

from __future__ import annotations

import random
from pathlib import Path

import polars as pl
from click.testing import CliRunner
from gensim.models import KeyedVectors

from interpretant.__main__ import cli

SHARED = ["the", "of", "and", "in", "a", "study", "model", "user", "system", "theory"]
FIELD_WORDS = {
    "philosophy": ["mind", "consciousness", "perception", "experience", "world", "body"],
    "hci": ["interface", "display", "interaction", "design", "usability", "screen"],
}
TARGETS = ["consciousness", "interface", "model"]
CONTROLS = ["study", "theory"]
LINES_PER_CELL = 1_300


def write_cell(path: Path, field: str, decade: int) -> None:
    rng = random.Random(f"{field}{decade}")
    vocabulary = SHARED + FIELD_WORDS[field]
    lines = [" ".join(rng.choices(vocabulary, k=12)) for _ in range(LINES_PER_CELL)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(*args: str | Path) -> str:
    result = CliRunner().invoke(cli, [str(arg) for arg in args])
    assert result.exit_code == 0, f"{args[:2]} failed:\n{result.output}\n{result.exception!r}"
    return result.output


def test_grid_commands_run_from_lemmatised_text_to_drift_tables(tmp_path: Path) -> None:
    lemmatised = tmp_path / "lemmatised"
    models, ppmi, drift = (tmp_path / name for name in ("twec", "ppmi", "drift"))
    for field in FIELD_WORDS:
        for decade in (1990, 2000):
            write_cell(lemmatised / "openalex" / "texts" / f"{field}_{decade}.txt", field, decade)
    words, controls = (tmp_path / name for name in ("w.txt", "c.txt"))
    words.write_text("\n".join(TARGETS))
    controls.write_text("\n".join(CONTROLS))

    run("twec", "grid", "--input-dir", lemmatised, "--output-dir", models, "--vector-size", "8",
        "--epochs", "1", "--workers", "1", "--min-count", "1",
        "--compass-tokens-per-cell", "2000")  # fmt: skip
    run("ppmi", "grid", "--input-dir", lemmatised, "--layer", "openalex", "--output-dir", ppmi,
        "--dimensions", "4", "--min-count", "5")  # fmt: skip
    run("drift", "grid", "--model-dir", models, "--output-dir", drift, "--words-file", words,
        "--controls-file", controls, "--measure-min", "20", "--neighbour-min", "10",
        "--neighbours", "5")  # fmt: skip

    for root in (models, ppmi):
        vectors = KeyedVectors.load(str(root / "openalex" / "hci_2000.kv"))
        assert "interface" in vectors.key_to_index
    within = pl.read_parquet(drift / "within_field.parquet")
    cross = pl.read_parquet(drift / "cross_field.parquet")
    assert set(within["field_a"]) == {"hci", "philosophy"}
    assert set(cross["decade_a"]) == {1990, 2000}
    assert within["is_target"].any()
    assert within["is_control"].any()
