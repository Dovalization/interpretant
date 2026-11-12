"""Corpus audit: per-cell volume and target-word frequency.

A cell is one ``(layer, source, decade)`` processed text file: the same field and decade can
exist in two layers (a main-grid sample and its full cell), so the layer is part of the key.
Drift between two decades is reliable only when a word occurs often enough in both: a vector
trained on a handful of occurrences stays close to its initialisation, and its nearest
neighbours are noise.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

FULL_LAYER = "openalex_full"
_CELL_FILENAME = re.compile(r"^(?P<source>[a-z0-9]+)_(?P<decade>\d{4})\.txt$")


@dataclass
class CellStats:
    """Volume and target-word counts for one (layer, source, decade) cell."""

    layer: str
    source: str
    decade: int
    documents: int = 0
    tokens: int = 0
    target_counts: Counter[str] = field(default_factory=Counter)


def load_target_words(path: Path) -> list[str]:
    """Read a words-of-interest file: one word per line, ``#`` starts a comment."""
    words: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        word = line.split("#", 1)[0].strip()
        if word:
            words.append(word)
    return words


def count_cell(path: Path, layer: str, source: str, decade: int, targets: set[str]) -> CellStats:
    """Stream a processed text file (one document per line) and count its contents."""
    stats = CellStats(layer=layer, source=source, decade=decade)
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            tokens = line.split()
            if not tokens:
                continue
            stats.documents += 1
            stats.tokens += len(tokens)
            stats.target_counts.update(token for token in tokens if token in targets)
    return stats


def scan_processed(processed_dir: Path, targets: set[str]) -> list[CellStats]:
    """Count every ``<layer>/texts/<source>_<decade>.txt`` file under ``processed_dir``."""
    cells: list[CellStats] = []
    for path in sorted(processed_dir.glob("*/texts/*.txt")):
        match = _CELL_FILENAME.match(path.name)
        if match is None:
            continue
        layer = path.parent.parent.name
        cells.append(count_cell(path, layer, match["source"], int(match["decade"]), targets))
    return cells


def merge_by_decade(cells: Iterable[CellStats]) -> dict[int, CellStats]:
    """Sum cells across sources, giving one merged cell per decade."""
    merged: dict[int, CellStats] = {}
    for cell in cells:
        total = merged.setdefault(
            cell.decade, CellStats(layer="merged", source="merged", decade=cell.decade)
        )
        total.documents += cell.documents
        total.tokens += cell.tokens
        total.target_counts.update(cell.target_counts)
    return dict(sorted(merged.items()))


def reliable_transitions(
    cells_by_decade: dict[int, CellStats], word: str, min_occurrences: int
) -> list[tuple[int, int]]:
    """Return consecutive decade pairs where ``word`` reaches ``min_occurrences`` in both."""
    decades = sorted(cells_by_decade)
    return [
        (earlier, later)
        for earlier, later in zip(decades, decades[1:], strict=False)
        if cells_by_decade[earlier].target_counts[word] >= min_occurrences
        and cells_by_decade[later].target_counts[word] >= min_occurrences
    ]


def format_decade_ranges(decades: Iterable[int], step: int = 10) -> str:
    """Compress decades into ranges, e.g. ``[1960, 1970, 1980, 2000]`` → ``1960–1980, 2000``."""
    ranges: list[list[int]] = []
    for decade in sorted(decades):
        if ranges and decade - ranges[-1][-1] == step:
            ranges[-1].append(decade)
        else:
            ranges.append([decade])
    return ", ".join(str(run[0]) if len(run) == 1 else f"{run[0]}–{run[-1]}" for run in ranges)


@dataclass
class CorpusAudit:
    """Everything ``corpus audit`` reports, built by :func:`audit_corpus`."""

    min_occurrences: int
    targets: list[str]
    cells: list[CellStats]
    merged: dict[int, CellStats]
    reliable_transitions: dict[str, list[tuple[int, int]]]

    def to_json(self) -> dict[str, object]:
        """The audit as JSON-ready data (the file ``figures`` and ``export web`` read)."""
        return {
            "min_occurrences": self.min_occurrences,
            "cells": [
                {**asdict(cell), "target_counts": dict(cell.target_counts)} for cell in self.cells
            ],
            "reliable_transitions": self.reliable_transitions,
        }


def audit_corpus(processed_dir: Path, targets: list[str], min_occurrences: int) -> CorpusAudit:
    """Count every cell under ``processed_dir``."""
    cells = scan_processed(processed_dir, set(targets))
    # A full cell contains its main-grid sample, so counting both would count it twice.
    merged = merge_by_decade(cell for cell in cells if cell.layer != FULL_LAYER)
    return CorpusAudit(
        min_occurrences=min_occurrences,
        targets=targets,
        cells=cells,
        merged=merged,
        reliable_transitions={
            word: reliable_transitions(merged, word, min_occurrences) for word in targets
        },
    )
