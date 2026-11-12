"""Drift on the field × decade grid, judged against a whole-vocabulary noise floor.

Every cell model shares the TWEC compass, so two cells can be compared word by word: the
same field across consecutive decades (a trajectory) or two fields in the same decade
(divergence). For a pair of cells, a word is measured only if it occurs at least
``MEASURE_MIN`` times in both; its neighbours are drawn from words occurring at least
``NEIGHBOUR_MIN`` times in both, so a change of vocabulary cannot pose as a change of
neighbourhood.

The noise floor comes from every measured word in the pair: words are grouped into
equal-sized frequency bins, and each word gets its percentile rank within its bin on cosine
drift and on neighbourhood shift. A word is a drift candidate only if it is in the top
``1 - FLOOR_QUANTILE`` on both measures; candidates become findings only if they survive the
robustness checks.
"""

from __future__ import annotations

import itertools
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from gensim.models import KeyedVectors
from tqdm import tqdm

from interpretant.drift.metrics import batch_neighborhood_shift

MEASURE_MIN = 150
NEIGHBOUR_MIN = 50
NEIGHBOURS = 25
FLOOR_QUANTILE = 0.95
MIN_BIN_WORDS = 100
MAX_BINS = 10

PAIR_SCHEMA = {
    "word": pl.String,
    "count_a": pl.Int64,
    "count_b": pl.Int64,
    "cosine_drift": pl.Float32,
    "neighbour_shift": pl.Float64,
}

_CELL_MODEL = re.compile(r"^(?P<name>[a-z0-9]+)_(?P<decade>\d{4})$")


@dataclass(frozen=True)
class CellModel:
    layer: str
    name: str
    decade: int
    path: Path


def discover_models(model_dir: Path, layers: tuple[str, ...]) -> list[CellModel]:
    """Return the trained cell models of the given layers, in layer, name and decade order."""
    models = []
    for layer in layers:
        for path in sorted((model_dir / layer).glob("*.kv")):
            match = _CELL_MODEL.match(path.stem)
            if match:
                models.append(CellModel(layer, match["name"], int(match["decade"]), path))
    return models


def within_field_pairs(models: list[CellModel]) -> list[tuple[CellModel, CellModel]]:
    """Pair each field's consecutive decades (a missing decade breaks the series)."""
    by_field: dict[tuple[str, str], list[CellModel]] = defaultdict(list)
    for model in models:
        by_field[(model.layer, model.name)].append(model)
    return [
        (earlier, later)
        for series in by_field.values()
        for earlier, later in itertools.pairwise(sorted(series, key=lambda m: m.decade))
        if later.decade - earlier.decade == 10
    ]


def cross_field_pairs(models: list[CellModel]) -> list[tuple[CellModel, CellModel]]:
    """Pair every two fields of the same layer in the same decade."""
    by_decade: dict[tuple[str, int], list[CellModel]] = defaultdict(list)
    for model in models:
        by_decade[(model.layer, model.decade)].append(model)
    return [
        pair
        for cells in by_decade.values()
        for pair in itertools.combinations(sorted(cells, key=lambda m: m.name), 2)
    ]


def _counts(vectors: KeyedVectors) -> dict[str, int]:
    return {word: int(vectors.get_vecattr(word, "count")) for word in vectors.index_to_key}


def _unit_rows(vectors: KeyedVectors, words: list[str]) -> np.ndarray:
    matrix = np.asarray([vectors[word] for word in words], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


def compare_cells(
    vectors_a: KeyedVectors,
    vectors_b: KeyedVectors,
    measure_min: int = MEASURE_MIN,
    neighbour_min: int = NEIGHBOUR_MIN,
    neighbours: int = NEIGHBOURS,
) -> pl.DataFrame:
    """Measure cosine drift and neighbourhood shift for every word frequent in both cells."""
    counts_a, counts_b = _counts(vectors_a), _counts(vectors_b)
    candidates = [
        word
        for word, count in counts_a.items()
        if count >= neighbour_min and counts_b.get(word, 0) >= neighbour_min
    ]
    measured = [
        word
        for word in candidates
        if counts_a[word] >= measure_min and counts_b[word] >= measure_min
    ]
    if not measured or len(candidates) <= neighbours:
        return pl.DataFrame(schema=PAIR_SCHEMA)
    matrix_a, matrix_b = _unit_rows(vectors_a, candidates), _unit_rows(vectors_b, candidates)
    index = {word: position for position, word in enumerate(candidates)}
    rows = [index[word] for word in measured]
    cosine = 1.0 - np.sum(matrix_a[rows] * matrix_b[rows], axis=1)
    shift = batch_neighborhood_shift(
        measured, candidates, matrix_a, candidates, matrix_b, k=neighbours
    )
    return pl.DataFrame(
        {
            "word": measured,
            "count_a": [counts_a[word] for word in measured],
            "count_b": [counts_b[word] for word in measured],
            "cosine_drift": cosine,
            "neighbour_shift": [shift[word] for word in measured],
        },
        schema=PAIR_SCHEMA,
    )


def cell_neighbours(
    vectors: KeyedVectors,
    word: str,
    k: int = NEIGHBOURS,
    pool_min: int = NEIGHBOUR_MIN,
    measure_min: int = MEASURE_MIN,
) -> list[str]:
    """Return a word's ``k`` nearest neighbours in one cell, nearest first.

    Neighbours are drawn from words with at least ``pool_min`` occurrences, as in
    :func:`compare_cells`; a word under ``measure_min`` occurrences has no reliable
    neighbourhood and gets an empty list.
    """
    counts = _counts(vectors)
    if counts.get(word, 0) < measure_min:
        return []
    pool = [other for other, count in counts.items() if count >= pool_min and other != word]
    if not pool:
        return []
    similarity = _unit_rows(vectors, pool) @ _unit_rows(vectors, [word])[0]
    order = np.argsort(-similarity)[:k]
    return [pool[position] for position in order]


def frequency_bins(frequency: pl.Series, bins: int) -> pl.Series:
    """Equal-sized bins by frequency: ordinal ranks, then linear percentile edges on the ranks.

    Each bin is closed on the right and the first also on the left. The edges are computed here
    rather than with Polars' ``qcut``, which its documentation marks unstable and which does not
    state how it interpolates the edges. They are percentiles (quantile × 100), as pandas'
    ``qcut`` computed them for the stored drift tables: when an edge falls on a whole rank, the
    rounding of ``np.quantile`` moves the boundary word to the other bin in 44 of 532 pairs.
    """
    ranks = frequency.rank("ordinal").cast(pl.Float64).to_numpy()
    edges = np.percentile(ranks, np.linspace(0, 1, bins + 1) * 100)
    codes = np.clip(np.searchsorted(edges, ranks, side="left") - 1, 0, bins - 1)
    return pl.Series("frequency_bin", codes, dtype=pl.Float64)


def add_noise_floor(
    pair: pl.DataFrame,
    quantile: float = FLOOR_QUANTILE,
    min_bin_words: int = MIN_BIN_WORDS,
    max_bins: int = MAX_BINS,
) -> pl.DataFrame:
    """Rank each word's drift within its frequency bin and flag drift candidates.

    Bins hold equal numbers of words by geometric-mean frequency, at least ``min_bin_words``
    each and at most ``max_bins``. Ranks are percentiles (0-100) within the bin, ties sharing
    their average rank; a candidate is in the top ``1 - quantile`` on both cosine drift and
    neighbourhood shift.
    """
    if pair.is_empty():
        return pair.with_columns(
            pl.lit(None, pl.Float64).alias("frequency_bin"),
            pl.lit(None, pl.Float64).alias("cosine_rank"),
            pl.lit(None, pl.Float64).alias("shift_rank"),
            pl.lit(None, pl.Boolean).alias("candidate"),
        )
    frequency = pl.Series(np.log(pair["count_a"].to_numpy() * pair["count_b"].to_numpy()) / 2)
    bins = max(1, min(max_bins, len(pair) // min_bin_words))

    def percentile(column: str) -> pl.Expr:
        # NaN counts as missing, as in pandas: Polars ranks and counts only non-null values.
        value = pl.col(column).fill_nan(None)
        return (
            value.rank("average").over("frequency_bin") / value.count().over("frequency_bin") * 100
        )

    threshold = quantile * 100
    return (
        pair.with_columns(frequency_bins(frequency, bins))
        .with_columns(
            cosine_rank=percentile("cosine_drift"), shift_rank=percentile("neighbour_shift")
        )
        .with_columns(
            candidate=(
                (pl.col("cosine_rank") > threshold) & (pl.col("shift_rank") > threshold)
            ).fill_null(False)
        )
    )


def measure_grid(
    model_dir: Path,
    targets: Iterable[str],
    controls: Iterable[str],
    measure_min: int = MEASURE_MIN,
    neighbour_min: int = NEIGHBOUR_MIN,
    neighbours: int = NEIGHBOURS,
) -> dict[str, pl.DataFrame]:
    """Compare every within-field and cross-field pair of cell models under ``model_dir``.

    Returns two tables keyed ``within_field`` (each field's consecutive decades in the OpenAlex
    and full OpenAlex layers) and ``cross_field`` (every two OpenAlex fields in the
    same decade): one row per word measured in a pair, ranked against the pair's noise
    floor, with ``is_target`` and ``is_control`` flags.
    """
    target_set, control_set = set(targets), set(controls)
    loaded: dict[Path, KeyedVectors] = {}

    def vectors(path: Path) -> KeyedVectors:
        if path not in loaded:
            loaded[path] = KeyedVectors.load(str(path))
        return loaded[path]

    pairs_by_table = {
        "within_field": within_field_pairs(
            discover_models(model_dir, ("openalex", "openalex_full"))
        ),
        "cross_field": cross_field_pairs(discover_models(model_dir, ("openalex",))),
    }
    tables = {}
    for name, pairs in pairs_by_table.items():
        if not pairs:
            raise ValueError(f"no {name} pairs of cell models under {model_dir}")
        frames = []
        for cell_a, cell_b in tqdm(pairs, desc=f"  {name}", unit="pair"):
            pair = add_noise_floor(
                compare_cells(
                    vectors(cell_a.path),
                    vectors(cell_b.path),
                    measure_min,
                    neighbour_min,
                    neighbours,
                )
            )
            labels = pl.DataFrame(
                {
                    "layer": [cell_a.layer],
                    "field_a": [cell_a.name],
                    "decade_a": [cell_a.decade],
                    "field_b": [cell_b.name],
                    "decade_b": [cell_b.decade],
                }
            )
            frames.append(labels.join(pair, how="cross"))
        tables[name] = pl.concat(frames).with_columns(
            is_target=pl.col("word").is_in(list(target_set)),
            is_control=pl.col("word").is_in(list(control_set)),
        )
    return tables
