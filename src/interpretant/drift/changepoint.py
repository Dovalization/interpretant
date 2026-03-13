"""Changepoint detection for semantic drift time series.

This module provides utilities for identifying the decade in which a word's
semantic drift accelerated most sharply.

Planned implementation: PELT (Pruned Exact Linear Time) changepoint algorithm
via the ``ruptures`` library, applied to the per-decade cosine-distance series.
"""

from __future__ import annotations

import numpy as np


def detect_changepoints(
    drift_series: list[float],
    decades: list[int],
    penalty: float = 3.0,
) -> list[int]:
    """Detect changepoints in a per-decade drift series.

    Args:
        drift_series: Ordered list of drift scores, one per decade.
        decades: Corresponding decade labels (must be same length).
        penalty: PELT penalty parameter; higher values yield fewer changepoints.

    Returns:
        List of decade labels at which changepoints were detected.

    Raises:
        NotImplementedError: Until the ``ruptures`` dependency is added.
    """
    raise NotImplementedError(
        "Changepoint detection is not yet implemented. "
        "Add 'ruptures>=1.1' to dependencies and implement PELT here."
    )


def drift_acceleration(drift_series: list[float]) -> np.ndarray:
    """Compute the second-order difference (acceleration) of a drift series.

    Args:
        drift_series: Ordered list of drift scores.

    Returns:
        Array of second differences (length = len(drift_series) - 2).
    """
    series = np.array(drift_series, dtype=float)
    return np.diff(series, n=2)
