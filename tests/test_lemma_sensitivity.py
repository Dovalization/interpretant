"""Round L1: the static view without and with the lemma exceptions, against bootstrap noise."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from interpretant.catalog.config import CatalogConfig, read_catalog
from interpretant.pipelines.build_static_view import build_static_view
from interpretant.pipelines.cleaning_sensitivity import CellPlan
from interpretant.pipelines.lemma_sensitivity import cell_segments, measure_lemma_sensitivity
from interpretant.pipelines.params import CorpusSelection
from tests.test_cleaning_sensitivity import small_builds

SELECTION = CorpusSelection(fields=("ai", "compsci"), decades=(2010, 2020))


def control_catalog(catalog: CatalogConfig, folder: Path) -> CatalogConfig:
    """The catalog with its own static view and no lemma exceptions, built: view A."""
    exceptions_file = folder / "no_exceptions.tsv"
    exceptions_file.write_text("form\tlemma\ttag\tsource\n", encoding="utf-8")
    control = catalog.model_copy(
        update={
            "static_view": catalog.static_view.model_copy(update={"path": folder / "view_a"}),
            "lemma_exceptions": catalog.lemma_exceptions.model_copy(
                update={"path": exceptions_file}
            ),
        }
    )
    build_static_view(control, SELECTION)
    return control


def test_measure_lemma_sensitivity_measures_averaged_b_and_averaged_a_again(
    corpus_catalog_file: Path, tmp_path: Path
) -> None:
    treatment = read_catalog(corpus_catalog_file)
    abstracts, _ = small_builds(corpus_catalog_file)
    control = control_catalog(treatment, tmp_path)

    result = measure_lemma_sensitivity(
        control, treatment, abstracts, targets=frozenset({"sense", "sens"}), bootstraps=2
    )

    assert set(result.movements["version"]) == {"averaged_treatment", "averaged_noise"}
    assert result.movements.height == 2
    assert result.models == 2 * 3 * 4
    assert int(result.doses["changed_tokens"].sum()) > 0
    counts = dict(
        zip(result.target_counts["target"], result.target_counts["tokens_a"], strict=True)
    )
    counts_b = dict(
        zip(result.target_counts["target"], result.target_counts["tokens_b"], strict=True)
    )
    assert counts["sens"] > 0
    assert counts_b["sens"] == 0
    assert counts_b["sense"] == counts["sense"] + counts["sens"]


def test_measure_lemma_sensitivity_refuses_one_view_for_both(corpus_catalog_file: Path) -> None:
    catalog = read_catalog(corpus_catalog_file)
    with pytest.raises(ValueError, match="same static view"):
        measure_lemma_sensitivity(catalog, catalog, pl.DataFrame(), targets=frozenset())


def test_cell_segments_refuse_views_with_other_offsets() -> None:
    plan = CellPlan(work_ids=["W1"], base_removals={"W1": []})
    view_a = {"W1": (["senses", "drift"], [0, 7], [False, True])}
    view_b = {"W1": (["sense", "drift"], [0, 8], [False, True])}
    with pytest.raises(ValueError, match="other offsets"):
        cell_segments(view_a, view_b, plan)


def test_cell_segments_count_the_changed_tokens_outside_the_removals() -> None:
    view_a = {"W1": (["sens", "drift", "sens"], [0, 7, 13], [False, True, False])}
    view_b = {"W1": (["sense", "drift", "sense"], [0, 7, 13], [False, True, False])}
    kept = CellPlan(work_ids=["W1"], base_removals={"W1": []})
    cut = CellPlan(work_ids=["W1"], base_removals={"W1": [(0, 6)]})
    assert cell_segments(view_a, view_b, kept)[2] == (1, 3, 2)
    _, segments_b, dose = cell_segments(view_a, view_b, cut)
    assert dose == (1, 2, 1)
    assert segments_b["W1"] == [["drift", "sense"]]


def test_cell_segments_count_words_of_view_a_and_changed_entries() -> None:
    view_a = {"W1": (["data", "base", "grew"], [0, 5, 10], [False, False, True])}
    view_b = {"W1": (["database", "", "grew"], [0, 5, 10], [False, False, True])}
    plan = CellPlan(work_ids=["W1"], base_removals={"W1": []})
    assert cell_segments(view_a, view_b, plan)[2] == (1, 3, 2)
    assert cell_segments(view_b, view_a, plan)[2] == (1, 2, 2)
