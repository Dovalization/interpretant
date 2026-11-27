"""The statistics behind the audits' bars: a sampling plan's bar, and a rate with its uncertainty.

An audit bar is a sampling plan: a number of items judged and the most errors that pass
(``docs/preprocessing_phase.md``, "Audit bars are sampling plans"). A rate is reported with its
Wilson interval and, where the judges were calibrated against a reference, corrected for their
misclassification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom, binomtest

# A corrected rate's posterior summary: its median, between the two ends of its 95% interval
_SUMMARY_PERCENTILES = (2.5, 50, 97.5)


@dataclass(frozen=True)
class ConfusionMatrix:
    """The judges' calls against a reference's on the same items, as the counts of a 2x2 table.

    A positive is the answer whose rate is estimated (an error, a junk record); the reference says
    which items are truly positive.
    """

    true_positives: int
    false_negatives: int
    false_positives: int
    true_negatives: int


@dataclass(frozen=True)
class CorrectedRate:
    """A rate corrected for the judges' misclassification: its posterior median and 95% interval."""

    median: float
    low: float
    high: float


def compute_acceptance_number(judged: int, *, true_rate: float, pass_probability: float) -> int:
    """The smallest number of errors that passes at least ``pass_probability`` of the time.

    A plan allowing this many errors among ``judged`` items passes with at least that probability
    when the true error rate is ``true_rate``: the binomial quantile, which is the acceptance
    number of a single sampling plan (NIST/SEMATECH e-Handbook of Statistical Methods, 6.2.2).

    Raises:
        ValueError: if the rate or the probability is not between 0 and 1.
    """
    if not (0 <= true_rate <= 1 and 0 <= pass_probability <= 1):
        raise ValueError(
            f"a true rate of {true_rate} and a pass probability of {pass_probability} must both "
            "be between 0 and 1"
        )
    return next(
        count
        for count in range(judged + 1)
        if binom.cdf(count, judged, true_rate) >= pass_probability
    )


def compute_wilson_interval(
    positives: int, total: int, *, confidence: float = 0.95
) -> tuple[float, float]:
    """The Wilson score interval of the share ``positives`` of ``total``, low end then high end.

    The interval the project reports rates with (Brown, Cai and DasGupta 2001, "Interval
    Estimation for a Binomial Proportion"), computed by scipy.

    Raises:
        ValueError: if ``total`` is zero, where a share has no interval.
    """
    interval = binomtest(positives, total).proportion_ci(
        confidence_level=confidence, method="wilson"
    )
    return float(interval.low), float(interval.high)


def compute_corrected_rate(
    confusion: ConfusionMatrix,
    positives: int,
    total: int,
    *,
    seed: int | np.random.Generator,
    draws: int,
) -> CorrectedRate:
    """A rate corrected for the judges' misclassification (Rogan and Gladen 1978).

    The judges called ``positives`` of ``total`` items positive. Their sensitivity and specificity
    come from ``confusion``. All three are drawn from their Beta posteriors under uniform priors,
    and each draw's corrected rate is (observed + specificity - 1) / (sensitivity + specificity
    - 1), clipped to [0, 1]. Draws where sensitivity + specificity is at most 1 are left out:
    there the judges do no better than chance and the estimator divides by zero or changes sign.

    Args:
        confusion: The judges' calls against the reference's on the calibration items.
        positives: The items the judges called positive in the sample whose rate is corrected.
        total: The items judged in that sample.
        seed: The draws' seed, or a generator to continue: rates corrected in turn from one
            generator each take the draws that follow the last one's.
        draws: The number of posterior draws.

    Returns:
        The posterior median of the corrected rate and its 95% interval.

    Raises:
        ValueError: if no draw has sensitivity + specificity above 1.
    """
    generator = np.random.default_rng(seed)
    sensitivity = generator.beta(1 + confusion.true_positives, 1 + confusion.false_negatives, draws)
    specificity = generator.beta(1 + confusion.true_negatives, 1 + confusion.false_positives, draws)
    rate = generator.beta(1 + positives, 1 + total - positives, draws)
    is_informative = sensitivity + specificity > 1
    if not is_informative.any():
        raise ValueError("no posterior draw has sensitivity + specificity above 1")
    corrected = np.clip(
        (rate[is_informative] + specificity[is_informative] - 1)
        / (sensitivity[is_informative] + specificity[is_informative] - 1),
        0,
        1,
    )
    low, median, high = np.percentile(corrected, _SUMMARY_PERCENTILES)
    return CorrectedRate(median=float(median), low=float(low), high=float(high))
