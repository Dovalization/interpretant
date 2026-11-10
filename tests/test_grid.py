"""Tests for the field × decade TWEC grid."""

from __future__ import annotations

from pathlib import Path

from gensim.models import KeyedVectors

from interpretant.embedding.grid import discover_cells, train_twec_grid, write_compass_sample
from interpretant.embedding.twec import TWECTrainer

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
    main_cells = discover_cells(tmp_path, ("books", "openalex"))
    extra_cells = discover_cells(tmp_path, ("openalex_full",))
    output = tmp_path / "m"

    def train() -> list[Path]:
        trainer = TWECTrainer(vector_size=8, epochs=1, workers=1, min_count=1)
        return train_twec_grid(trainer, main_cells, extra_cells, output, 2_000)

    first, second = train(), train()

    assert len(first) == 3
    assert second == []
    assert (output / "compass.model").exists()
    assert not (output / "compass_sample.txt").exists()
    reference = KeyedVectors.load(str(output / "books" / "books_1890.kv"))
    for label in ["books/books_1890", "openalex/philosophy_1990", "openalex_full/hci_2010"]:
        vectors = KeyedVectors.load(str(output / f"{label}.kv"))
        assert vectors.vector_size == 8
        assert "representation" in vectors.key_to_index
        assert set(vectors.key_to_index) == set(reference.key_to_index)
