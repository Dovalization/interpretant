"""Tests for the field × decade TWEC grid."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner
from gensim.models import KeyedVectors

from interpretant.__main__ import cli
from interpretant.embedding.grid import discover_cells, write_compass_sample

SENTENCE = "the mind forms a representation of the world through perception and memory "


def write_cell(root: Path, layer: str, stem: str, repeats: int) -> Path:
    path = root / layer / "texts" / f"{stem}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([SENTENCE * 5] * repeats) + "\n", encoding="utf-8")
    return path


def test_discover_cells_reads_layer_name_and_decade(tmp_path: Path) -> None:
    write_cell(tmp_path, "openalex", "philosophy_1990", 2)
    write_cell(tmp_path, "books", "books_1890", 2)
    (tmp_path / "books" / "texts" / "notes.txt").write_text("not a cell")

    cells = discover_cells(tmp_path, ("books", "openalex"))

    assert [cell.label for cell in cells] == ["books/books_1890", "openalex/philosophy_1990"]
    assert cells[1].decade == 1990


def test_compass_sample_takes_the_same_budget_from_every_cell(tmp_path: Path) -> None:
    write_cell(tmp_path, "openalex", "compsci_2010", 200)
    write_cell(tmp_path, "books", "books_1850", 40)
    cells = discover_cells(tmp_path, ("books", "openalex"))

    written = write_compass_sample(cells, 2_000, tmp_path / "sample.txt")

    assert 4_000 <= written < 4_200


def test_twec_grid_trains_every_cell_in_one_space_and_resumes(tmp_path: Path) -> None:
    write_cell(tmp_path, "books", "books_1890", 30)
    write_cell(tmp_path, "openalex", "philosophy_1990", 30)
    write_cell(tmp_path, "openalex_full", "hci_2010", 30)
    args = ["twec", "grid", "--input-dir", str(tmp_path), "--output-dir", str(tmp_path / "m")]
    args += ["--vector-size", "8", "--epochs", "1", "--workers", "1", "--min-count", "1"]

    first = CliRunner().invoke(cli, args)
    second = CliRunner().invoke(cli, args)

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert (tmp_path / "m" / "compass.model").exists()
    for label in ["books/books_1890", "openalex/philosophy_1990", "openalex_full/hci_2010"]:
        kv = KeyedVectors.load(str(tmp_path / "m" / f"{label}.kv"))
        assert kv.vector_size == 8
        assert "representation" in kv.key_to_index
