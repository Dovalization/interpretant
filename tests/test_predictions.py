"""Tests for the helpers the prediction scorers share."""

from __future__ import annotations

import numpy as np
import pytest

from interpretant.predictions import report


@pytest.mark.parametrize(
    ("verdict", "word"),
    [
        (True, "PASSED"),
        (False, "FALSIFIED"),
        (None, "UNTESTABLE"),
        (np.bool_(True), "PASSED"),
        ("partial", "PARTIAL"),
    ],
)
def test_report_prints_the_verdict_word(
    verdict: bool | str | None, word: str, capsys: pytest.CaptureFixture[str]
) -> None:
    report("P1 sanity", verdict, "detail")

    assert capsys.readouterr().out == f"\nP1 sanity: {word}\n  detail\n"
