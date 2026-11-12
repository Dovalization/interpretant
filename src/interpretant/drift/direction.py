"""Where a classifier trained on one decade sends the abstracts of later decades.

A classifier trained on one decade sends later abstracts away from their own field more often
as every field's language changes, and some fields receive more of those departures than
others. :func:`assignment_runs` records, per seed, training decade and test decade, the share of
each field's held-out abstracts sent to every field (round N's data; the method and predictions
are in docs/predictions_cross_perplexity.md).

An effect is *present* when it is positive in every seed and its mean is at least twice its
standard deviation across seeds.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable
from pathlib import Path

import polars as pl

from interpretant.drift.separability import (
    PER_FIELD,
    assignment_shares,
    decade_samples,
    fit_fields,
    split,
)

PRESENCE_RATIO = 2.0


def _ignore(message: str) -> None:
    del message


def assignment_runs(
    texts_dir: Path,
    train_decades: Iterable[int],
    test_decades: Iterable[int],
    seeds: Iterable[int],
    per_field: int = PER_FIELD,
    log: Callable[[str], None] = _ignore,
) -> pl.DataFrame:
    """Train one classifier per seed and training decade, and apply it to the held-out
    abstracts of that decade and of every later test decade.

    One row per seed, training decade, test decade, text field and assigned field.
    """
    train_decades, test_decades = list(train_decades), list(test_decades)
    rows: list[dict[str, object]] = []
    for seed in seeds:
        held_out = {
            decade: split(decade_samples(texts_dir, decade, seed, per_field))[1]
            for decade in sorted(set(train_decades) | set(test_decades))
        }
        for train_decade in train_decades:
            train, _ = split(decade_samples(texts_dir, train_decade, seed, per_field))
            model = fit_fields(train)
            for test_decade in [train_decade, *(d for d in test_decades if d > train_decade)]:
                rows.extend(
                    {"seed": seed, "train_decade": train_decade, "test_decade": test_decade, **row}
                    for row in assignment_shares(model, held_out[test_decade])
                )
            log(f"  seed {seed}: {train_decade}s classifier applied")
    return pl.DataFrame(rows)


def present(values: list[float], ratio: float = PRESENCE_RATIO) -> bool:
    """True when every value is positive and the mean is at least ``ratio`` standard
    deviations: the seed-noise rule for an effect to count."""
    if len(values) < 2 or min(values) <= 0:
        return False
    return statistics.mean(values) >= ratio * statistics.stdev(values)
