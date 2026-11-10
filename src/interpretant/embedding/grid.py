"""Field × decade grid for TWEC: find the cells and build the compass's balanced sample.

A cell is one phrased text file, ``<input_dir>/<layer>/texts/<name>_<decade>.txt``: a books
decade (``books/texts/books_1890.txt``), an OpenAlex field and decade
(``openalex/texts/philosophy_1990.txt``) or a full OpenAlex cell
(``openalex_full/texts/hci_2010.txt``). Every cell gets its own model, trained against one
shared compass, so any two cells are comparable.

The compass is trained on the same token budget from every main cell, so the large
2010s–2020s cells do not define the shared space for the 1850s or for philosophy.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from interpretant.corpus.phrases import sample_chunks
from interpretant.embedding.ppmi import train_cell
from interpretant.embedding.twec import EpochProgress, TWECTrainer

_CELL_FILENAME = re.compile(r"^(?P<name>[a-z0-9]+)_(?P<decade>\d{4})$")


@dataclass(frozen=True)
class Cell:
    layer: str
    name: str
    decade: int
    path: Path

    @property
    def label(self) -> str:
        return f"{self.layer}/{self.name}_{self.decade}"


def discover_cells(input_dir: Path, layers: tuple[str, ...]) -> list[Cell]:
    """Return every cell file in the given layers, in layer, name and decade order."""
    cells = []
    for layer in layers:
        for path in sorted((input_dir / layer / "texts").glob("*.txt")):
            match = _CELL_FILENAME.match(path.stem)
            if match:
                cells.append(Cell(layer, match["name"], int(match["decade"]), path))
    return cells


def write_compass_sample(cells: list[Cell], tokens_per_cell: int, output: Path) -> int:
    """Write the same token budget from every cell to ``output``; return tokens written."""
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as sample:
        for cell in cells:
            for chunk in sample_chunks(cell.path, tokens_per_cell):
                sample.write(" ".join(chunk) + "\n")
                written += len(chunk)
    return written


def cell_model_path(model_dir: Path, cell: Cell) -> Path:
    """Where a cell's model lives: ``<model_dir>/<layer>/<name>_<decade>.kv``."""
    return model_dir / cell.layer / f"{cell.name}_{cell.decade}.kv"


def _ignore(message: str) -> None:
    del message


def train_twec_grid(  # noqa: PLR0913
    trainer: TWECTrainer,
    main_cells: list[Cell],
    extra_cells: list[Cell],
    output_dir: Path,
    compass_tokens_per_cell: int,
    log: Callable[[str], None] = _ignore,
) -> list[Path]:
    """Train one TWEC model per cell against a compass shared by every cell.

    The compass is trained on the same token budget from every main cell (or loaded from
    ``output_dir/compass.model`` if it exists); extra cells are trained against it but left
    out of it. Cells whose model already exists are skipped, so an interrupted run resumes.
    Returns the paths of the models written in this run.
    """
    compass_path = output_dir / "compass.model"
    if compass_path.exists():
        trainer.load_compass(compass_path)
        log(f"  compass loaded from {compass_path}")
    else:
        sample_path = output_dir / "compass_sample.txt"
        tokens = write_compass_sample(main_cells, compass_tokens_per_cell, sample_path)
        log(f"  compass sample: {tokens:,} tokens → training compass …")
        with tqdm(total=trainer.epochs, desc="  compass", unit="epoch") as bar:
            trainer.train_compass([sample_path], extra_callbacks=[EpochProgress(bar)])
        trainer.save_compass(output_dir)
        sample_path.unlink()
    log(f"  compass vocab: {trainer.compass_vocab_size:,} words")

    written = []
    for cell in tqdm(main_cells + extra_cells, desc="  cells", unit="cell"):
        target = cell_model_path(output_dir, cell)
        if target.exists():
            continue
        with tqdm(total=trainer.epochs, desc=f"    {cell.label}", unit="epoch", leave=False) as bar:
            vectors = trainer.train_decade(
                [cell.path], cell.decade, extra_callbacks=[EpochProgress(bar)]
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        vectors.save(str(target))
        written.append(target)
    return written


def build_ppmi_grid(
    cells: list[Cell], output_dir: Path, window: int, dimensions: int, min_count: int
) -> list[Path]:
    """Build PPMI-SVD vectors for every cell whose model does not exist yet; return the new
    paths. Each cell is factorised on its own, so compare PPMI cells by neighbourhood shift."""
    written = []
    for cell in tqdm(cells, desc="  cells", unit="cell"):
        target = cell_model_path(output_dir, cell)
        if target.exists():
            continue
        vectors = train_cell(cell.path, window, dimensions, min_count)
        target.parent.mkdir(parents=True, exist_ok=True)
        vectors.save(str(target))
        written.append(target)
    return written
