"""Usage-pair measures on synthetic inputs: XL-LEXEME's input preparation, the distances and the
DWUG readers. The encoder's weights are not loaded here."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from interpretant.drift.pairs import (
    Usage,
    average_pairwise_distance,
    centred,
    dwug_graded_change,
    dwug_usages,
    marked_tokens,
    prototype_distance,
)


def split(text: str) -> list[str]:
    return text.split()


def test_the_word_is_wrapped_in_markers_and_its_span_runs_from_one_marker_to_the_other() -> None:
    usage = Usage("the quick fox jumps", 10, 13)

    tokens, first, last = marked_tokens(usage, split)

    assert tokens == ["the", "quick", "<t>", "fox", "</t>", "jumps"]
    assert tokens[first:last] == ["<t>", "fox", "</t>"]


def test_a_word_at_the_start_or_the_end_has_no_empty_context_tokenised() -> None:
    calls: list[str] = []

    def recording(text: str) -> list[str]:
        calls.append(text)
        return text.split()

    marked_tokens(Usage("fox", 0, 3), recording)

    assert calls == ["<t>", "fox", "</t>"]


def test_centring_keeps_the_target_and_takes_equal_room_on_both_sides() -> None:
    tokens = [f"l{n}" for n in range(10)] + ["<t>", "w", "</t>"] + [f"r{n}" for n in range(10)]

    kept = centred(tokens, 10, 13, 9)

    assert kept == ["l7", "l8", "l9", "<t>", "w", "</t>", "r0", "r1", "r2"]


def test_centring_cuts_only_the_long_side_and_leaves_the_short_sides_room_unused() -> None:
    tokens = ["l0", "<t>", "w", "</t>"] + [f"r{n}" for n in range(10)]

    kept = centred(tokens, 1, 4, 9)

    assert kept == ["l0", "<t>", "w", "</t>", "r0", "r1", "r2"]


def test_pairwise_and_prototype_distances() -> None:
    first = np.array([[1.0, 0.0], [1.0, 0.0]])
    second = np.array([[0.0, 1.0], [1.0, 0.0]])

    assert average_pairwise_distance(first, first) == pytest.approx(0.0)
    assert average_pairwise_distance(first, second) == pytest.approx(0.5)
    assert prototype_distance(first, second) == pytest.approx(1 - np.sqrt(0.5))


def test_dwug_readers_take_usages_by_period_and_the_graded_change(tmp_path: Path) -> None:
    (tmp_path / "data" / "fox_nn").mkdir(parents=True)
    (tmp_path / "stats" / "opt").mkdir(parents=True)
    header = "lemma\tgrouping\tcontext\tindexes_target_token\n"
    rows = 'fox_nn\t1\tthe "quick" fox\t12:15\nfox_nn\t2\ta fox ran\t2:5\n'
    (tmp_path / "data" / "fox_nn" / "uses.csv").write_text(header + rows, encoding="utf-8")
    (tmp_path / "stats" / "opt" / "stats_groupings.csv").write_text(
        "lemma\tgrouping\tchange_graded\nfox_nn\t1_2\t0.25\n", encoding="utf-8"
    )

    usages = dwug_usages(tmp_path)

    assert {
        period: [u.text[u.start : u.end] for u in us] for period, us in usages["fox_nn"].items()
    } == {
        1: ["fox"],
        2: ["fox"],
    }
    assert dwug_graded_change(tmp_path) == {"fox_nn": 0.25}
