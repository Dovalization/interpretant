"""The junk gate's rules: the classifier's screen, the decade rule and the exclusions."""

from __future__ import annotations

import polars as pl
import pytest
from scipy.stats import binom, fisher_exact

from interpretant.review.junk_rules import (
    build_exclusions,
    compute_decade_verdicts,
    is_junk_usage,
    screen_cells,
)


def drawn_usages(cells: dict[tuple[str, int], tuple[int, int]]) -> pl.DataFrame:
    """Drawn usages for each cell: (usages, of which from records called junk)."""
    rows = [
        {"word": word, "decade": decade, "predicted_junk": index < junk}
        for (word, decade), (count, junk) in cells.items()
        for index in range(count)
    ]
    return pl.DataFrame(rows)


def test_the_screen_flags_a_cell_with_an_excess_of_predicted_junk() -> None:
    drawn = drawn_usages(
        {("attention", 1990): (400, 20), ("network", 1990): (100, 25), ("model", 2000): (200, 10)}
    )

    cells = screen_cells(drawn, min_drawn_usages=100, alpha=0.01)

    flagged = cells.filter("flagged").select("word", "decade").rows()
    assert flagged == [("network", 1990)]
    network = cells.filter(pl.col("word") == "network").row(0, named=True)
    assert network["decade_rate"] == pytest.approx(45 / 500)
    assert network["tail_probability"] == pytest.approx(binom.sf(24, 100, 45 / 500))


def test_a_cell_under_the_minimum_is_never_flagged_however_much_junk() -> None:
    drawn = drawn_usages({("attention", 1990): (400, 4), ("network", 1990): (99, 99)})

    cells = screen_cells(drawn, min_drawn_usages=100, alpha=0.01)

    network = cells.filter(pl.col("word") == "network").row(0, named=True)
    assert not network["gated"]
    assert not network["flagged"]
    assert network["tail_probability"] is None
    assert cells.columns == [
        "word",
        "decade",
        "drawn",
        "predicted_junk",
        "decade_rate",
        "tail_probability",
        "flagged",
        "gated",
    ]


def test_a_cell_at_its_decades_rate_is_not_flagged() -> None:
    drawn = drawn_usages({("attention", 1990): (300, 30), ("network", 1990): (100, 10)})

    cells = screen_cells(drawn, min_drawn_usages=100, alpha=0.01)

    assert not cells["flagged"].any()
    assert cells["gated"].all()


def test_each_decade_is_tested_against_the_other_decades_together_one_sided() -> None:
    verdicts = compute_decade_verdicts({1960: (40, 500), 1950: (60, 500), 1970: (20, 500)})

    assert [verdict.decade for verdict in verdicts] == [1950, 1960, 1970]
    expected = fisher_exact([[60, 440], [60, 940]], alternative="greater").pvalue
    assert verdicts[0].fisher_p == pytest.approx(expected)
    assert verdicts[2].fisher_p > 0.99
    assert not verdicts[0].passes_bar
    assert verdicts[1].passes_bar


def test_a_junk_usage_is_a_junk_record_or_another_texts_sentence() -> None:
    assert is_junk_usage("junk", "own")
    assert is_junk_usage("abstract", "other")
    assert not is_junk_usage("mixed", "own")


def test_exclusions_hold_the_words_usages_from_records_called_junk() -> None:
    drawn = pl.DataFrame(
        {
            "work_id": ["W1", "W1", "W2", "W3"],
            "word": ["network", "attention", "network", "network"],
            "predicted_junk": [True, True, False, True],
        }
    )

    assert build_exclusions(drawn, {"network"}).rows() == [("W1", "network"), ("W3", "network")]
    assert build_exclusions(drawn, set()).height == 0
