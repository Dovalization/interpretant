"""Round S1's pipeline: the cells' plans, the finite-verb check and a run on the small corpus."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from interpretant.catalog.config import read_catalog
from interpretant.catalog.repaired_texts import load_repaired_texts
from interpretant.catalog.schemas import REMOVAL
from interpretant.pipelines.annotate_corpus import annotate_corpus
from interpretant.pipelines.build_static_view import build_static_view
from interpretant.pipelines.cleaning_sensitivity import (
    SensitivitySettings,
    average_pairs,
    has_finite_verb_outside,
    measure_cleaning_sensitivity,
    plan_cells,
)
from interpretant.pipelines.params import CorpusSelection
from tests.corpus_catalog import ANNOTATE_SETTINGS

BUILD_SCHEMA = {
    "work_id": pl.String(),
    "fields": pl.List(pl.String()),
    "decade": pl.Int64(),
    "source": pl.String(),
    "text": pl.String(),
    "removals": pl.List(REMOVAL),
    "dropped_reason": pl.String(),
}


BuildRow = tuple[str, list[str], int, str, list[tuple[int, int]], str | None]


def build(rows: list[BuildRow]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            (
                work,
                fields,
                decade,
                "regular",
                text,
                [{"start": s, "end": e, "reason": "url"} for s, e in ranges],
                dropped,
            )
            for work, fields, decade, text, ranges, dropped in rows
        ],
        schema=BUILD_SCHEMA,
        orient="row",
    )


def test_a_finite_verb_inside_a_removed_range_does_not_count() -> None:
    assert has_finite_verb_outside([0, 5], [False, True], [])
    assert not has_finite_verb_outside([0, 5], [False, True], [(4, 9)])


def test_plan_cells_takes_the_later_build_s_drops_and_added_ranges() -> None:
    base = build(
        [
            ("W1", ["ai", "hci"], 2010, "one", [(0, 1)], None),
            ("W2", ["ai"], 2010, "two", [], None),
            ("W3", ["ai"], 2010, "three", [], None),
            ("W4", ["ai"], 2010, "four", [], "letter"),
        ]
    )
    later = build(
        [
            ("W1", ["ai", "hci"], 2010, "one", [(0, 1), (2, 3)], None),
            ("W2", ["ai"], 2010, "two", [], "web_scrape"),
            ("W3", ["ai"], 2010, "three", [], None),
        ]
    )
    plans = plan_cells(base, later, ())
    assert set(plans) == {"ai", "hci"}
    cell = plans["ai"][2010]
    assert cell.work_ids == ["W1", "W2", "W3"]
    assert cell.later_dropped == {"W2"}
    assert cell.later_removals == {"W1": [(0, 1), (2, 3)]}
    assert plan_cells(base, later, ("hci",))["hci"][2010].work_ids == ["W1"]


def test_plan_cells_refuses_builds_whose_texts_differ() -> None:
    base = build([("W1", ["ai"], 2010, "one", [], None)])
    later = build([("W1", ["ai"], 2010, "uno", [], None)])
    with pytest.raises(ValueError, match="another text"):
        plan_cells(base, later, ())


def small_builds(corpus_catalog_file: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """P and the later build over the small corpus's static view; W2 dropped later."""
    catalog = read_catalog(corpus_catalog_file)
    selection = CorpusSelection(fields=("ai", "compsci"), decades=(2010, 2020))
    annotate_corpus(catalog, selection, ANNOTATE_SETTINGS)
    build_static_view(catalog, selection)
    works: dict[str, tuple[list[str], int, str]] = {}
    for field, decade in (("ai", 2010), ("compsci", 2010), ("ai", 2020)):
        for work_id, text in load_repaired_texts(catalog, f"{field}_{decade}").iter_rows():
            fields = works.get(work_id, ([], decade, text))[0]
            works[work_id] = ([*fields, field], decade, text)
    # W3 also counts in HCI, which has no part of its own: its cell is found in AI's part.
    rows: list[BuildRow] = [
        (work, [*fields, "hci"] if work.endswith("W3") else fields, decade, text, [], None)
        for work, (fields, decade, text) in works.items()
    ]
    dropped = next(work for work in works if work.endswith("W2"))
    later_rows: list[BuildRow] = [
        (work, fields, decade, text, ranges, "web_scrape" if work == dropped else reason)
        for work, fields, decade, text, ranges, reason in rows
    ]

    return build(rows), build(later_rows)


def test_measure_cleaning_sensitivity_measures_each_version_of_each_comparison(
    corpus_catalog_file: Path,
) -> None:
    catalog = read_catalog(corpus_catalog_file)
    base, later = small_builds(corpus_catalog_file)

    result = measure_cleaning_sensitivity(
        catalog,
        base,
        later,
        targets=frozenset({"model"}),
        settings=SensitivitySettings(placebos=2, bootstraps=2),
    )

    movements = result.movements
    assert movements.height == 5
    assert set(zip(movements["version"], movements["replicate"], strict=True)) == {
        ("treatment", 0),
        ("placebo", 0),
        ("placebo", 1),
        ("bootstrap", 0),
        ("bootstrap", 1),
    }
    assert set(movements["field"]) == {"ai"}
    assert result.doses.height == 4
    hci = result.doses.filter(pl.col("field") == "hci")
    assert hci["works"].to_list() == [1]
    ai_2010 = result.doses.filter((pl.col("field") == "ai") & (pl.col("decade") == 2010))
    assert ai_2010["dropped_works"].item() >= 1
    assert result.models == 6 * 4
    assert result.target_doses["target"].to_list() == ["model"]


def test_the_averaged_round_measures_averaged_c_and_averaged_p_again(
    corpus_catalog_file: Path,
) -> None:
    catalog = read_catalog(corpus_catalog_file)
    base, later = small_builds(corpus_catalog_file)
    result = measure_cleaning_sensitivity(
        catalog,
        base,
        later,
        targets=frozenset({"model"}),
        settings=SensitivitySettings(bootstraps=2, averaged=True),
    )
    assert set(result.movements["version"]) == {"averaged_treatment", "averaged_noise"}
    assert result.movements.height == 2
    assert result.models == 2 * 3 * 4


def test_average_pairs_keeps_words_measured_in_every_replicate() -> None:
    schema = {
        "word": pl.String,
        "count_a": pl.Int64,
        "count_b": pl.Int64,
        "cosine_drift": pl.Float32,
        "neighbour_shift": pl.Float64,
    }
    first = pl.DataFrame(
        {
            "word": ["a", "b"],
            "count_a": [200, 300],
            "count_b": [200, 300],
            "cosine_drift": [0.1, 0.2],
            "neighbour_shift": [0.2, 0.4],
        },
        schema=schema,
    )
    second = pl.DataFrame(
        {
            "word": ["a"],
            "count_a": [400],
            "count_b": [200],
            "cosine_drift": [0.3],
            "neighbour_shift": [0.6],
        },
        schema=schema,
    )
    averaged = average_pairs([first, second])
    assert averaged["word"].to_list() == ["a"]
    assert averaged["neighbour_shift"].item() == pytest.approx(0.4)
    assert averaged["count_a"].item() == pytest.approx(300)
