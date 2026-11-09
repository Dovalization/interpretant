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
from dataclasses import dataclass
from pathlib import Path

from interpretant.corpus.phrases import sample_chunks

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
