"""Tests for the seed-noise rule of the direction measures."""

from __future__ import annotations

from interpretant.drift.direction import present


def test_presence_needs_every_seed_positive_and_two_standard_deviations() -> None:
    assert present([0.10, 0.11, 0.09, 0.10, 0.12])
    assert not present([0.10, 0.11, -0.01, 0.10, 0.12])
    assert not present([0.01, 0.20, 0.02, 0.15, 0.01])
