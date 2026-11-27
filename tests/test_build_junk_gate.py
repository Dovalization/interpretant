"""The build-junk-gate stage on a synthetic corpus with a planted junk-heavy cell.

In the 1990s, 60 of 320 records are junk, and *network* (held out) is drawn in all 60 and in 60
prose records: its cell is half junk where the decade's drawn usages are about a quarter. In the
2000s, *network* is drawn in 30 junk records only, a cell under 100 drawn usages.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import yaml

from interpretant.catalog.config import read_catalog
from interpretant.catalog.junk_gate import JunkLabel, JunkLabels, load_junk_cells, save_junk_labels
from interpretant.catalog.junk_gate_tables import SURFACE_FEATURES
from interpretant.catalog.record_vectors import load_record_vectors
from interpretant.pipelines.build_junk_gate import (
    ClassifierSettings,
    OcrSettings,
    ScreenSettings,
    build_junk_gate,
)
from interpretant.review.junk_classifier import FEATURE_NAMES, parse_classifier
from interpretant.review.junk_gate import GATED_CELL_USAGES
from tests.junk_gate_records import (
    SEALED,
    SyntheticRecord,
    build_label,
    work_id,
    write_records,
    write_words,
)

REPOSITORY = Path(__file__).resolve().parents[1]
CLASSIFIER = ClassifierSettings(
    regularisation=1.0,
    class_weight="balanced",
    folds=5,
    specificity_target=0.97,
    max_iterations=1000,
)
SCREEN = ScreenSettings(min_drawn_usages=100, alpha=0.01)
# The corpus is too small for its word counts to mean anything: only the dictionary knows words
OCR = OcrSettings(min_letters=3, min_words=5, min_known_works=1000)


def build_records() -> list[SyntheticRecord]:
    nineties = [
        SyntheticRecord(
            number, 1995, is_junk=number <= 60, has_network=number <= 120, fields=("ai", "compsci")
        )
        if number % 3 == 0
        else SyntheticRecord(number, 1995, is_junk=number <= 60, has_network=number <= 120)
        for number in range(1, 321)
    ]
    noughties = [
        SyntheticRecord(number, 2005, is_junk=number <= 1030, has_network=number <= 1030)
        for number in range(1001, 1151)
    ]
    return nineties + noughties


def build_labels(usages: dict[tuple[int, str], dict[str, object]]) -> JunkLabels:
    """L: 20 junk and 80 prose records in the 1990s, 10 and 40 in the 2000s; E: 30 junk records.

    Records 1 and 61 are labelled on *network* too: record 1's second item disagrees.
    """
    labels: list[JunkLabel] = []
    for number in [*range(1, 21), *range(61, 81), *range(121, 181)]:
        is_junk = number <= 60
        labels.append(
            build_label(
                usages[number, "attention"],
                "labelled",
                decade=1990,
                verdict="junk" if is_junk else "abstract",
                sentence="other" if is_junk else "own",
                damaged="yes" if is_junk else "no",
            )
        )
    for number in [*range(1001, 1011), *range(1031, 1071)]:
        is_junk = number <= 1030
        labels.append(
            build_label(
                usages[number, "attention"],
                "labelled",
                decade=2000,
                verdict="junk" if is_junk else "abstract",
                damaged="yes" if is_junk else "no",
            )
        )
    labels += [
        build_label(usages[1, "network"], "labelled", decade=1990, verdict="mixed", damaged="yes"),
        build_label(usages[61, "network"], "labelled", decade=1990, verdict="abstract"),
    ]
    labels += [
        build_label(usages[number, "attention"], "enrichment", decade=1990, verdict="junk")
        for number in range(21, 51)
    ]
    return JunkLabels(
        codebook_sha256="c" * 64,
        items={f"S{number:04d}": label for number, label in enumerate(labels, start=1)},
    )


@pytest.fixture(scope="module")
def gate(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """The synthetic corpus, its labels, and the stage run once on it."""
    folder = tmp_path_factory.mktemp("junk_gate")
    shutil.copy(REPOSITORY / "catalog.yaml", folder / "catalog.yaml")
    catalog = read_catalog(folder / "catalog.yaml")
    write_words(catalog)
    usages = write_records(catalog, build_records())
    labels = build_labels(usages)
    save_junk_labels(catalog, labels)
    result = build_junk_gate(catalog, CLASSIFIER, SCREEN, OCR, process_count=1)
    return SimpleNamespace(catalog=catalog, labels=labels, result=result)


def read_cells(gate: SimpleNamespace) -> dict[tuple[str, int], dict[str, object]]:
    cells = pl.read_parquet(gate.catalog.junk_cells.path)
    return {(row["word"], row["decade"]): row for row in cells.iter_rows(named=True)}


def test_the_planted_junk_heavy_cell_is_flagged_and_no_other(gate: SimpleNamespace) -> None:
    cells = read_cells(gate)

    assert cells["network", 1990]["drawn"] == 120
    assert cells["network", 1990]["predicted_junk"] == 60
    assert cells["network", 1990]["flagged"]
    assert [cell for cell, row in cells.items() if row["flagged"]] == [("network", 1990)]
    assert gate.result.flagged_count == 1
    flagged = load_junk_cells(gate.catalog).filter("flagged")
    assert flagged.select("word", "decade").rows() == [("network", 1990)]


def test_a_cell_under_100_drawn_usages_is_never_flagged(gate: SimpleNamespace) -> None:
    cells = read_cells(gate)

    small = cells["network", 2000]
    assert small["drawn"] == 30 < GATED_CELL_USAGES
    assert small["predicted_junk"] == 30
    assert not small["gated"]
    assert not small["flagged"]
    assert small["tail_probability"] is None
    assert all(row["gated"] == (row["drawn"] >= 100) for row in cells.values())


def test_the_ocr_score_leaves_out_the_target_and_is_null_under_five_words(
    gate: SimpleNamespace,
) -> None:
    scores = pl.read_parquet(gate.catalog.usage_ocr_scores.path)
    by_usage = {(row["work_id"], row["word"]): row for row in scores.iter_rows(named=True)}

    # "Attention improves recall in every network we tried.": five words, the target capitalised
    assert by_usage[work_id(61), "attention"]["ocr_words"] == 5
    assert by_usage[work_id(61), "attention"]["ocr_unknown_share"] == 0.0
    # The same sentence for "network": the target left out, four words, no score
    assert by_usage[work_id(61), "network"]["ocr_words"] == 4
    assert by_usage[work_id(61), "network"]["ocr_unknown_share"] is None
    # The junk sentence for "network": tbe, qnick, brovvn, wvrds unknown, every known
    assert by_usage[work_id(1), "network"]["ocr_words"] == 5
    assert by_usage[work_id(1), "network"]["ocr_unknown_share"] == pytest.approx(0.8)
    assert scores.height == 320 + 120 + 150 + 30


def test_the_metrics_name_held_out_words_by_their_sealed_names_only(
    gate: SimpleNamespace,
) -> None:
    text = gate.catalog.junk_gate_metrics.path.read_text()
    metrics = json.loads(text)

    assert "network" not in text
    assert [cell["word"] for cell in metrics["cells"]["flagged_cells"]] == [SEALED]
    assert [(cell["word"], cell["decade"]) for cell in metrics["cells"]["ungated_cells"]] == [
        (SEALED, 2000)
    ]


def test_the_metrics_hold_the_classifiers_bar_the_counts_and_the_ocr_check(
    gate: SimpleNamespace,
) -> None:
    metrics = json.loads(gate.catalog.junk_gate_metrics.path.read_text())

    assert metrics["records"] == {
        "behind_drawn_usages": 470,
        "called_junk": metrics["records"]["called_junk"],
        "labelled_sample": 150,
        "labelled_sample_junk": 30,
        "enrichment": 30,
        "enrichment_junk": 30,
        "with_disagreeing_items": 1,
    }
    classifier = metrics["classifier"]
    assert classifier["meets_bar"]
    assert classifier["sensitivity"]["of"] == 30
    assert classifier["specificity"]["of"] == 120
    assert set(classifier["false_alarms_by_decade"]) == {"1990", "2000"}
    assert set(classifier["false_alarms_by_field"]) == {"ai", "compsci"}
    assert classifier["correction_by_decade"]["1990"]["items"] == 102
    assert metrics["ocr"]["testable"]
    assert metrics["ocr"]["auc"] == 1.0
    assert metrics["ocr"]["without_score"] == 1
    assert metrics["cells"]["gated"] == 3
    assert metrics["usages"]["drawn"] == 620


def test_l_records_keep_their_held_out_scores_and_folds(gate: SimpleNamespace) -> None:
    scores = pl.read_parquet(gate.catalog.junk_scores.path)
    labelled = {label.work_id for label in gate.labels.items.values() if label.set == "labelled"}

    with_fold = set(scores.filter(pl.col("fold").is_not_null())["work_id"])
    assert with_fold == labelled
    assert scores.columns == ["work_id", *FEATURE_NAMES, "score", "predicted_junk", "fold"]
    junk = {work_id(number) for number in [*range(1, 61), *range(1001, 1031)]}
    called = set(scores.filter("predicted_junk")["work_id"])
    assert len(called ^ junk) <= 0.03 * scores.height


def test_the_saved_classifier_gives_the_scores_of_the_records_outside_l(
    gate: SimpleNamespace,
) -> None:
    classifier = parse_classifier(gate.catalog.junk_classifier.path.read_text())
    scores = pl.read_parquet(gate.catalog.junk_scores.path).filter(pl.col("fold").is_null())
    vectors = (
        load_record_vectors(gate.catalog)
        .join(scores.select("work_id").lazy(), on="work_id", how="semi")
        .sort("work_id")
        .collect()
    )

    recomputed = classifier.scorer.compute_scores(
        vectors["vector"].to_numpy(), scores.select(FEATURE_NAMES).to_numpy()
    )

    np.testing.assert_array_equal(recomputed, scores["score"].to_numpy())
    np.testing.assert_array_equal(recomputed >= classifier.cut_off, scores["predicted_junk"])


def test_the_catalogs_feature_columns_are_the_classifiers_features() -> None:
    assert SURFACE_FEATURES == FEATURE_NAMES


def test_the_gates_settings_in_params_yaml_are_the_preregistered_ones() -> None:
    sections = yaml.safe_load((REPOSITORY / "params.yaml").read_text())["junk_gate"]

    assert sections["classifier"] == {
        "regularisation": 1.0,
        "class_weight": "balanced",
        "folds": 5,
        "specificity_target": 0.97,
        "max_iterations": 1000,
    }
    assert sections["screen"] == {"min_drawn_usages": GATED_CELL_USAGES, "alpha": 0.01}
    assert sections["ocr"] == {"min_letters": 3, "min_words": 5, "min_known_works": 20}
    assert ClassifierSettings() == CLASSIFIER
    assert ScreenSettings() == SCREEN
