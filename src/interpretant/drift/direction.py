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

Round W holds the frame of reference and the baseline fixed and changes only the instrument.
Each instrument trained on one decade gives every held-out abstract a *cost* under every
field's model: −log₂ p(field | abstract) for the classifier (:class:`ClassifierCosts`), bits per
scored token for that field's bigram model (:class:`LanguageModelCosts`). The *contrast* of a
text field between two target groups is its mean cost under one group's models minus under the
other's, each group without the text field, so neither the field's own model nor a field that
every field drifts towards enters it unless it is a target. A field's *movement* is the fall in
its contrast from the training decade to a later one, and the *relative movement* nets the
median movement of every other field (:func:`relative_movement`). :func:`frame_runs` gives both
instruments the same abstracts, on whole cells and with one field's text restricted to a
venue-stable subset; :func:`planted_runs` checks on real abstracts of one decade that an
instrument finds a move planted in them.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl

from interpretant.drift.perplexity import MIN_VOCABULARY_COUNT, KneserNeyBigram, training_sample
from interpretant.drift.separability import (
    PER_FIELD,
    assignment_shares,
    decade_samples,
    fit_fields,
    split,
)

PRESENCE_RATIO = 2.0
PLANTED_SHARE = 0.1

Costs = dict[str, dict[str, float]]


def _ignore(message: str) -> None:
    del message


def assignment_runs(  # noqa: PLR0913
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


class FieldCosts(Protocol):
    """An instrument trained on every field of one decade, giving texts a cost per field."""

    def mean_costs(self, texts: list[str]) -> dict[str, float]:
        """The mean cost of ``texts`` under each field's model."""
        ...


class ClassifierCosts:
    """Round N's classifier, read through its probabilities: a text's cost under field T is
    −log₂ p(T | text)."""

    def __init__(self, train: dict[str, list[str]]) -> None:
        self.pipeline = fit_fields(train)

    def mean_costs(self, texts: list[str]) -> dict[str, float]:
        bits = -np.log2(self.pipeline.predict_proba(texts))
        return {
            str(field): float(bits[:, column].mean())
            for column, field in enumerate(self.pipeline.classes_)
        }


class LanguageModelCosts:
    """One bigram model per field: a text's cost under field T is its bits per scored token
    under T's model.

    Every field's model trains on the same number of tokens (the smallest field's training
    total; abstracts taken in their drawn order) over one vocabulary, the words with at least
    ``MIN_VOCABULARY_COUNT`` occurrences across those samples. Only vocabulary words and end
    tokens are scored, so every model scores the same tokens of a text.
    """

    def __init__(self, train: dict[str, list[str]]) -> None:
        budget = min(sum(len(text.split()) for text in texts) for texts in train.values())
        samples = {field: training_sample(texts, budget) for field, texts in train.items()}
        counts: Counter[str] = Counter()
        for sample in samples.values():
            for text in sample:
                counts.update(text.split())
        vocabulary = {word for word, count in counts.items() if count >= MIN_VOCABULARY_COUNT}
        self.models = {
            field: KneserNeyBigram(vocabulary).fit(sample)
            for field, sample in sorted(samples.items())
        }

    def mean_costs(self, texts: list[str]) -> dict[str, float]:
        return {
            field: statistics.fmean(model.known_bits_per_token(text) for text in texts)
            for field, model in self.models.items()
        }


def cost_rows(instrument: FieldCosts, test: dict[str, list[str]]) -> list[dict[str, object]]:
    """One row per text field in ``test`` and field model, with the text's mean cost."""
    return [
        {"text_field": text_field, "model_field": model_field, "cost": cost}
        for text_field, texts in test.items()
        for model_field, cost in instrument.mean_costs(texts).items()
    ]


def frame_runs(  # noqa: PLR0913
    texts_dir: Path,
    early: int,
    late: int,
    seeds: Iterable[int],
    fit: Callable[[dict[str, list[str]]], FieldCosts] = ClassifierCosts,
    stable: dict[int, list[str]] | None = None,
    field: str = "hci",
    per_field: int = PER_FIELD,
    log: Callable[[str], None] = _ignore,
) -> pl.DataFrame:
    """Fit one instrument per seed on ``early``'s training abstracts and score the held-out
    abstracts of ``early`` and ``late`` under every field's model: round N's samples, with the
    instrument as a parameter (:class:`ClassifierCosts` or :class:`LanguageModelCosts`).

    One row per seed, cell, test decade, text field and model field. Cell ``whole`` holds every
    field's held-out abstracts. With ``stable`` (a decade's venue-stable abstracts of
    ``field``), cell ``stable`` repeats the other fields' rows and replaces ``field``'s: its
    ``early`` text is every stable abstract outside that seed's training abstracts, its
    ``late`` text every stable abstract.
    """
    rows: list[dict[str, object]] = []
    for seed in seeds:
        train, held_out_early = split(decade_samples(texts_dir, early, seed, per_field))
        _, held_out_late = split(decade_samples(texts_dir, late, seed, per_field))
        instrument = fit(train)
        trained = set(train[field]) if field in train else set()
        for decade, held_out in ((early, held_out_early), (late, held_out_late)):
            scored = cost_rows(instrument, held_out)
            key = {"seed": seed, "train_decade": early, "test_decade": decade}
            rows.extend({**key, "cell": "whole", **row} for row in scored)
            if stable is None:
                continue
            texts = [text for text in stable[decade] if decade != early or text not in trained]
            others = [row for row in scored if row["text_field"] != field]
            rows.extend(
                {**key, "cell": "stable", **row}
                for row in others + cost_rows(instrument, {field: texts})
            )
        log(f"  seed {seed}: {early}s instrument scored the {early}s and {late}s")
    return pl.DataFrame(rows)


def planted_runs(  # noqa: PLR0913
    texts_dir: Path,
    decade: int,
    seeds: Iterable[int],
    fit: Callable[[dict[str, list[str]]], FieldCosts],
    donors: Iterable[str],
    field: str = "hci",
    planted_share: float = PLANTED_SHARE,
    per_field: int = PER_FIELD,
    log: Callable[[str], None] = _ignore,
) -> pl.DataFrame:
    """A move planted in real abstracts, all from one decade, for an instrument to find.

    Each seed's held-out abstracts are halved: the first half stands for the early decade, the
    second for the late one, so no field moves but by resampling. Cell ``none`` scores the
    halves as drawn. Each donor's cell replaces the first ``planted_share`` of ``field``'s late
    abstracts with the donor's late abstracts and repeats every other field's rows: a change in
    which abstracts the field holds, not in their words.

    One row per seed, cell, half (``early`` or ``late``), text field and model field.
    """
    rows: list[dict[str, object]] = []
    for seed in seeds:
        train, held_out = split(decade_samples(texts_dir, decade, seed, per_field))
        instrument = fit(train)
        halves = {
            "early": {name: texts[: len(texts) // 2] for name, texts in held_out.items()},
            "late": {name: texts[len(texts) // 2 :] for name, texts in held_out.items()},
        }
        scored = {half: cost_rows(instrument, test) for half, test in halves.items()}
        late = halves["late"]
        planted = round(len(late[field]) * planted_share)
        others = [row for row in scored["late"] if row["text_field"] != field]
        cells = {"none": scored["late"]}
        for donor in donors:
            texts = late[donor][:planted] + late[field][planted:]
            cells[donor] = others + cost_rows(instrument, {field: texts})
        for cell, late_rows in cells.items():
            rows.extend(
                {"seed": seed, "cell": cell, "half": "early", **row} for row in scored["early"]
            )
            rows.extend({"seed": seed, "cell": cell, "half": "late", **row} for row in late_rows)
        log(f"  seed {seed}: {decade}s instrument scored the planted halves")
    return pl.DataFrame(rows)


def cost_matrix(rows: pl.DataFrame) -> Costs:
    """Text field to model field to cost, from the rows of one seed, cell and test decade."""
    matrix: Costs = {}
    for text, model, cost in rows.select("text_field", "model_field", "cost").iter_rows():
        matrix.setdefault(str(text), {})[str(model)] = float(cost)
    return matrix


def contrast(costs: Costs, text: str, towards: tuple[str, ...], over: tuple[str, ...]) -> float:
    """``text``'s mean cost under ``towards``' models minus under ``over``'s, each group
    without ``text``: lower means ``text`` reads more like ``towards`` than like ``over``."""

    def group_cost(group: tuple[str, ...]) -> float:
        return statistics.fmean(costs[text][member] for member in group if member != text)

    return group_cost(towards) - group_cost(over)


def movement(
    early: Costs, late: Costs, text: str, towards: tuple[str, ...], over: tuple[str, ...]
) -> float:
    """The fall in ``text``'s contrast from ``early`` to ``late``: positive is a move towards
    ``towards`` over ``over``."""
    return contrast(early, text, towards, over) - contrast(late, text, towards, over)


def relative_movement(
    early: Costs, late: Costs, text: str, towards: tuple[str, ...], over: tuple[str, ...]
) -> float:
    """``text``'s movement minus the median movement of every other field (each field for
    which both groups keep a member once it is left out)."""
    others = [
        other for other in early if other != text and set(towards) - {other} and set(over) - {other}
    ]
    baseline = statistics.median(movement(early, late, other, towards, over) for other in others)
    return movement(early, late, text, towards, over) - baseline


def share(  # noqa: PLR0913
    table: pl.DataFrame, seed: int, train: int, test: int, text: str, targets: tuple[str, ...]
) -> float:
    """Share of ``text``'s held-out abstracts in ``test`` that the ``train`` classifier sends to
    any of ``targets``."""
    rows = table.filter(
        pl.col("seed") == seed,
        pl.col("train_decade") == train,
        pl.col("test_decade") == test,
        pl.col("text_field") == text,
        pl.col("assigned_field").is_in(targets),
    )
    return float(rows.select(pl.col("share").sum()).item())


def gain(  # noqa: PLR0913
    table: pl.DataFrame, seed: int, train: int, test: int, text: str, targets: tuple[str, ...]
) -> float:
    """Change in the share of ``text``'s abstracts sent to ``targets``, training to test decade."""
    return share(table, seed, train, test, text, targets) - share(
        table, seed, train, train, text, targets
    )


def relative_gain(  # noqa: PLR0913
    table: pl.DataFrame, seed: int, train: int, test: int, text: str, targets: tuple[str, ...]
) -> float:
    """``text``'s gain towards ``targets`` minus the median gain towards them of every other
    field (neither ``text`` nor one of ``targets``)."""
    fields = set(table["text_field"].unique())
    others = sorted(fields - {text} - set(targets))
    baseline = statistics.median(gain(table, seed, train, test, other, targets) for other in others)
    return gain(table, seed, train, test, text, targets) - baseline


def present(values: list[float], ratio: float = PRESENCE_RATIO) -> bool:
    """True when every value is positive and the mean is at least ``ratio`` standard
    deviations: the seed-noise rule for an effect to count."""
    if len(values) < 2 or min(values) <= 0:
        return False
    return statistics.mean(values) >= ratio * statistics.stdev(values)
