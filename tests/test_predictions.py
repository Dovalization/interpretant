"""Tests for the helpers the prediction scorers share."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from gensim.models import KeyedVectors

from interpretant.predictions import load_vectors, overlap, report, word_count


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


def test_load_vectors_reads_a_cell_once_and_counts_words(tmp_path: Path) -> None:
    vectors = KeyedVectors(vector_size=2)
    vectors.add_vectors(["mind", "world"], np.eye(2, dtype=np.float32))
    vectors.set_vecattr("mind", "count", 7)
    vectors.set_vecattr("world", "count", 3)
    (tmp_path / "openalex").mkdir()
    vectors.save(str(tmp_path / "openalex" / "philosophy_1990.kv"))

    loaded = load_vectors(tmp_path, "openalex", "philosophy", 1990)

    assert load_vectors(tmp_path, "openalex", "philosophy", 1990) is loaded
    assert word_count(loaded, "mind") == 7
    assert word_count(loaded, "absent") == 0


def test_overlap_counts_distinct_shared_words() -> None:
    assert overlap(["mind", "mind", "body", "world"], {"mind", "world", "soul"}) == 2
