"""The sampling plans' statistics (`review/sampling_plans.py`): the bar, Wilson, the correction."""

from __future__ import annotations

import numpy as np
import pytest

from interpretant.review.sampling_plans import (
    ConfusionMatrix,
    compute_acceptance_number,
    compute_corrected_rate,
    compute_wilson_interval,
)

PERFECT_JUDGES = ConfusionMatrix(
    true_positives=2000, false_negatives=0, false_positives=0, true_negatives=2000
)
SEED = 20261407
DRAWS = 20_000


def test_the_usages_audits_bar_is_51_errors_in_1980_items() -> None:
    """The bar `docs/usages_audit.md` states, and the plan's earlier one for about 800 items."""
    assert compute_acceptance_number(1980, true_rate=0.02, pass_probability=0.96) == 51
    assert compute_acceptance_number(800, true_rate=0.02, pass_probability=0.96) == 23


@pytest.mark.parametrize(
    ("pass_probability", "bar"),
    [(0.25, 0), (0.26, 1), (0.75, 1), (0.76, 2), (1.0, 2)],
)
def test_the_bar_is_the_smallest_count_that_passes_often_enough(
    pass_probability: float, bar: int
) -> None:
    """Two items at a true rate of a half: no error 25% of the time, at most one 75%."""
    assert compute_acceptance_number(2, true_rate=0.5, pass_probability=pass_probability) == bar


@pytest.mark.parametrize(("true_rate", "pass_probability"), [(0.02, 1.2), (-0.1, 0.96)])
def test_a_rate_or_probability_outside_0_to_1_is_refused(
    true_rate: float, pass_probability: float
) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        compute_acceptance_number(100, true_rate=true_rate, pass_probability=pass_probability)


@pytest.mark.parametrize(
    ("positives", "total", "interval"),
    [(16, 200, (0.0498, 0.1260)), (1, 150, (0.0012, 0.0368))],
)
def test_the_wilson_interval_is_the_one_the_reports_state(
    positives: int, total: int, interval: tuple[float, float]
) -> None:
    """16 of 200 is 5–13% (`docs/datasheet.md`); 1 of 150 is 0.1–3.7% (`docs/compounds.md`)."""
    assert compute_wilson_interval(positives, total) == pytest.approx(interval, abs=1e-4)


def test_the_wilson_interval_narrows_at_a_lower_confidence() -> None:
    low, high = compute_wilson_interval(16, 200)
    narrower_low, narrower_high = compute_wilson_interval(16, 200, confidence=0.8)

    assert low < narrower_low < 0.08 < narrower_high < high


def test_the_wilson_interval_of_no_positives_starts_at_zero() -> None:
    low, high = compute_wilson_interval(0, 10)

    assert low == 0.0
    assert 0.0 < high < 0.5


def test_a_share_of_no_items_has_no_wilson_interval() -> None:
    with pytest.raises(ValueError, match="not less than 1"):
        compute_wilson_interval(0, 0)


def test_the_corrected_rate_of_perfect_judges_is_the_observed_rate() -> None:
    corrected = compute_corrected_rate(PERFECT_JUDGES, 250, 1000, seed=SEED, draws=DRAWS)

    assert corrected.median == pytest.approx(0.25, abs=0.005)
    assert 0.2 < corrected.low < 0.25 < corrected.high < 0.3


def test_the_corrected_rate_is_the_same_for_the_same_seed() -> None:
    once = compute_corrected_rate(PERFECT_JUDGES, 250, 1000, seed=SEED, draws=DRAWS)
    again = compute_corrected_rate(PERFECT_JUDGES, 250, 1000, seed=SEED, draws=DRAWS)
    other_seed = compute_corrected_rate(PERFECT_JUDGES, 250, 1000, seed=SEED + 1, draws=DRAWS)

    assert once == again
    assert once != other_seed


def test_the_correction_lowers_a_rate_the_judges_overstate() -> None:
    """Judges calling a third of the true negatives positive: half the positives called are not."""
    overstating = ConfusionMatrix(
        true_positives=1000, false_negatives=0, false_positives=1000, true_negatives=2000
    )

    corrected = compute_corrected_rate(overstating, 500, 1000, seed=SEED, draws=DRAWS)

    assert corrected.median == pytest.approx(0.25, abs=0.03)
    assert corrected.high < 0.5


def test_the_correction_raises_a_rate_the_judges_understate() -> None:
    """Judges finding half the true positives: the true rate is twice the observed one."""
    understating = ConfusionMatrix(
        true_positives=1000, false_negatives=1000, false_positives=0, true_negatives=2000
    )

    corrected = compute_corrected_rate(understating, 100, 1000, seed=SEED, draws=DRAWS)

    assert corrected.median == pytest.approx(0.2, abs=0.03)
    assert corrected.low > 0.1


def test_rates_corrected_from_one_generator_take_successive_draws() -> None:
    generator = np.random.default_rng(SEED)

    first = compute_corrected_rate(PERFECT_JUDGES, 250, 1000, seed=generator, draws=DRAWS)
    second = compute_corrected_rate(PERFECT_JUDGES, 250, 1000, seed=generator, draws=DRAWS)

    assert first == compute_corrected_rate(PERFECT_JUDGES, 250, 1000, seed=SEED, draws=DRAWS)
    assert second != first
    assert second.median == pytest.approx(first.median, abs=0.005)


def test_judges_no_better_than_chance_in_every_draw_are_refused() -> None:
    inverted = ConfusionMatrix(
        true_positives=0, false_negatives=5000, false_positives=5000, true_negatives=0
    )

    with pytest.raises(ValueError, match="sensitivity \\+ specificity above 1"):
        compute_corrected_rate(inverted, 250, 1000, seed=SEED, draws=10)
