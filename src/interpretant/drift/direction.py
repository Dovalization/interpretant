"""Who moved towards whom, with the general drift of every field taken out.

A classifier trained on one decade sends later abstracts away from their own field more often
as every field's language changes, and some fields receive more of those departures than
others. Round S read a field's raw gain towards another as movement towards it; rounds S and V
showed that every field drifts and that some fields act as sinks. The *relative gain* corrects
for both: for a text field A, a target field (or group) B, a training decade and a later test
decade, it is A's gain in the share of its abstracts sent to B minus the median gain towards B
of every other field. A target that receives everyone's drift leaves it unchanged. On synthetic
fields calibrated to the corpus's regime it finds a planted move of 5% of a field's words and
never reports one that was not planted, while the raw gain reports every target that all fields
drift towards (tests/test_direction.py; round N's method).

An effect is *present* when it is positive in every seed and its mean is at least twice its
standard deviation across seeds. The method and predictions are fixed in
docs/predictions_cross_perplexity.md (round N).
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

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


def assignment_runs(  # noqa: PLR0913
    texts_dir: Path,
    train_decades: Iterable[int],
    test_decades: Iterable[int],
    seeds: Iterable[int],
    per_field: int = PER_FIELD,
    log: Callable[[str], None] = _ignore,
) -> pd.DataFrame:
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
    return pd.DataFrame(rows)


def share(  # noqa: PLR0913
    table: pd.DataFrame, seed: int, train: int, test: int, text: str, targets: tuple[str, ...]
) -> float:
    """Share of ``text``'s held-out abstracts in ``test`` that the ``train`` classifier sends to
    any of ``targets``."""
    rows = table[
        (table.seed == seed)
        & (table.train_decade == train)
        & (table.test_decade == test)
        & (table.text_field == text)
        & (table.assigned_field.isin(targets))
    ]
    return float(rows.share.sum())


def gain(  # noqa: PLR0913
    table: pd.DataFrame, seed: int, train: int, test: int, text: str, targets: tuple[str, ...]
) -> float:
    """Change in the share of ``text``'s abstracts sent to ``targets``, training to test decade."""
    return share(table, seed, train, test, text, targets) - share(
        table, seed, train, train, text, targets
    )


def relative_gain(  # noqa: PLR0913
    table: pd.DataFrame, seed: int, train: int, test: int, text: str, targets: tuple[str, ...]
) -> float:
    """``text``'s gain towards ``targets`` minus the median gain towards them of every other
    field (neither ``text`` nor one of ``targets``)."""
    fields = set(table.text_field.unique())
    others = sorted(fields - {text} - set(targets))
    baseline = statistics.median(gain(table, seed, train, test, other, targets) for other in others)
    return gain(table, seed, train, test, text, targets) - baseline


def present(values: list[float], ratio: float = PRESENCE_RATIO) -> bool:
    """True when every value is positive and the mean is at least ``ratio`` standard
    deviations: the seed-noise rule for an effect to count."""
    if len(values) < 2 or min(values) <= 0:
        return False
    return statistics.mean(values) >= ratio * statistics.stdev(values)
