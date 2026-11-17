"""A cell's text under a build's removals, and the versions round S1 compares."""

from __future__ import annotations

import numpy as np
import pytest

from interpretant.corpus.cell_versions import (
    Cut,
    bootstrap_sample,
    cell_lines,
    kept_segments,
    placebo_cut,
    seeded,
    token_count,
)

TOKENS = ["a", "b", "c", "d"]
STARTS = [0, 2, 4, 6]


def test_a_removed_range_drops_its_tokens_and_ends_the_line() -> None:
    assert kept_segments(TOKENS, STARTS, [(4, 5)]) == [["a", "b"], ["d"]]


def test_a_range_between_two_kept_tokens_ends_the_line() -> None:
    assert kept_segments(TOKENS, STARTS, [(1, 2)]) == [["a"], ["b", "c", "d"]]


def test_overlapping_ranges_act_as_one() -> None:
    assert kept_segments(TOKENS, STARTS, [(2, 5), (3, 7)]) == [["a"]]


def test_no_range_keeps_one_line() -> None:
    assert kept_segments(TOKENS, STARTS, []) == [TOKENS]


def test_cell_lines_leave_out_dropped_works_and_cut_spans() -> None:
    works = {"w1": [["a", "b", "c", "d", "e"]], "w2": [["x", "y"]], "w3": [["p"], ["q", "r"]]}
    cut = Cut(dropped=frozenset({"w2"}), spans={"w1": (1, 2), "w3": (0, 2)})
    assert cell_lines(works, cut) == ["a", "d e", "r"]


def test_cell_lines_use_a_sample_with_repeats() -> None:
    works = {"w1": [["a"]], "w2": [["b"]]}
    assert cell_lines(works, sample=["w2", "w2", "w1"]) == ["b", "b", "a"]


def lengths_cell() -> dict[str, list[list[str]]]:
    return {f"w{n:03d}": [["t"] * (n % 40 + 1)] for n in range(200)}


def test_placebo_matches_the_treatment_in_works_and_spans_and_avoids_its_works() -> None:
    works = lengths_cell()
    dropped = frozenset({"w001", "w050", "w120", "w199"})
    passages = {"w010": 3, "w020": 7}
    cut = placebo_cut(works, dropped, passages, seeded(20261116, 0, "ai_2010"))
    assert len(cut.dropped) == len(dropped)
    assert not (cut.dropped | set(cut.spans)) & (dropped | set(passages))
    assert sorted(size for _, size in cut.spans.values()) == [3, 7]
    assert all(token_count(works[host]) >= size for host, (_, size) in cut.spans.items())
    treated = sum(token_count(works[w]) for w in dropped)
    placebo = sum(token_count(works[w]) for w in cut.dropped)
    assert placebo == pytest.approx(treated, rel=0.35)


def test_placebo_and_bootstrap_draws_repeat_with_the_same_seed() -> None:
    works = lengths_cell()
    dropped = frozenset({"w001", "w050"})
    first = placebo_cut(works, dropped, {"w010": 3}, seeded(20261116, 3, "ai_2010"))
    again = placebo_cut(works, dropped, {"w010": 3}, seeded(20261116, 3, "ai_2010"))
    other = placebo_cut(works, dropped, {"w010": 3}, seeded(20261116, 4, "ai_2010"))
    assert first == again
    assert first != other
    ids = sorted(works)
    assert bootstrap_sample(ids, seeded(1, 0, "x")) == bootstrap_sample(ids, seeded(1, 0, "x"))


def test_bootstrap_sample_draws_as_many_works_with_replacement() -> None:
    ids = [f"w{n}" for n in range(50)]
    sample = bootstrap_sample(ids, np.random.default_rng(0))
    assert len(sample) == 50
    assert set(sample) <= set(ids)
    assert len(set(sample)) < 50
