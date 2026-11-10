"""Held-out check: does a cell's model generalise to text from the cell it never saw?

A cell's stored documents are shuffled with a seed, 10% are held out, and a Kneser-Ney bigram
model is fitted on the rest (capped at a token budget). The cell's *gap* is the cross-entropy
of the held-out documents minus that of an equally sized sample of the training documents: a
model that generalises has a small gap, one that memorised a small cell a large one. The method
and predictions are fixed in docs/predictions_cross_perplexity.md (round M).
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from interpretant.drift.perplexity import KneserNeyBigram, read_cell, training_sample
from interpretant.embedding.grid import discover_cells

HELD_OUT_SHARE = 0.1
MAX_TRAINING_TOKENS = 5_000_000
WORKERS = 4


@dataclass(frozen=True)
class CellGap:
    documents: int
    held_out_documents: int
    training_tokens: int
    training_cross_entropy: float
    held_out_cross_entropy: float

    @property
    def gap(self) -> float:
        return self.held_out_cross_entropy - self.training_cross_entropy


def cell_gap(
    documents: list[str],
    seed: int,
    held_out_share: float = HELD_OUT_SHARE,
    max_training_tokens: int = MAX_TRAINING_TOKENS,
) -> CellGap:
    """Fit a bigram model on a cell's training part and compare held-out and training scores.

    Raises ValueError for a cell with fewer than two documents, which cannot be split.
    """
    if len(documents) < 2:
        raise ValueError("a cell needs at least two documents to hold one out")
    shuffled = list(documents)
    random.Random(seed).shuffle(shuffled)
    held = max(1, round(len(shuffled) * held_out_share))
    held_out = shuffled[:held]
    training = training_sample(shuffled[held:], max_training_tokens)
    counts = Counter(word for document in training for word in document.split())
    # Every training word, not only frequent ones: a frequency floor maps most of a small cell's
    # words to <unk>, which makes its held-out text look cheap (round M's method amendment).
    model = KneserNeyBigram(set(counts)).fit(training)
    comparison = random.Random(seed).sample(training, min(len(held_out), len(training)))
    return CellGap(
        documents=len(documents),
        held_out_documents=len(held_out),
        training_tokens=sum(counts.values()),
        training_cross_entropy=model.cross_entropy(comparison),
        held_out_cross_entropy=model.cross_entropy(held_out),
    )


def _score_cell(job: tuple[str, str, int, Path, int]) -> dict[str, object]:
    layer, field, decade, path, seed = job
    _, documents = read_cell(path)
    gap = cell_gap(documents, seed)
    return {
        "seed": seed,
        "layer": layer,
        "field": field,
        "decade": decade,
        **asdict(gap),
        "gap": gap.gap,
    }


def _ignore(message: str) -> None:
    del message


def heldout_runs(
    input_dir: Path,
    layers: tuple[str, ...],
    seeds: Iterable[int],
    workers: int = WORKERS,
    log: Callable[[str], None] = _ignore,
) -> pd.DataFrame:
    """Score every cell of the given layers once per seed; one row per seed and cell."""
    jobs = [
        (cell.layer, cell.name, cell.decade, cell.path, seed)
        for seed in seeds
        for cell in discover_cells(input_dir, layers)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(_score_cell, jobs):
            rows.append(row)
            cell = f"{row['layer']}/{row['field']}_{row['decade']}"
            log(f"  seed {row['seed']} {cell}: gap {row['gap']:.3f}")
    return pd.DataFrame(rows)
