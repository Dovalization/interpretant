"""The junk gate's calls on synthetic records: labels, folds, cut-offs, agreement and quality."""

from __future__ import annotations

import hashlib

import numpy as np
import polars as pl
import pytest

from interpretant.catalog.junk_gate import JunkLabel, JunkLabels
from interpretant.review.junk_calls import (
    CutOffs,
    build_folds,
    build_record_labels,
    build_sources,
    call_by_agreement,
    call_records,
    compute_corrections,
    compute_cut_off,
    compute_false_alarms,
    compute_quality,
    set_cut_offs,
)


def label(work: int, item_set: str, verdict: str) -> JunkLabel:
    return JunkLabel(
        set=item_set,
        stratum=None if item_set != "enrichment" else "list_like",
        work_id=f"https://openalex.org/W{work}",
        part="ai_1990",
        decade=1990,
        word="attention",
        content_key="0" * 20,
        verdict=verdict,
        kind="none" if verdict != "junk" else "contents_or_list",
        sentence="own",
        damaged="no",
    )


def test_a_record_takes_its_smallest_items_label_and_disagreements_are_counted() -> None:
    labels = JunkLabels(
        codebook_sha256="c" * 64,
        items={
            "S0009": label(1, "labelled", "abstract"),
            "S0002": label(1, "labelled", "junk"),
            "S0003": label(2, "labelled", "mixed"),
            "S0004": label(2, "labelled", "abstract"),
            "S0005": label(3, "enrichment", "junk"),
            "C0001": label(4, "confirmation", "junk"),
        },
    )

    records = build_record_labels(labels)

    assert records.table.to_dicts() == [
        {"work_id": "https://openalex.org/W1", "is_junk": True, "in_labelled_sample": True},
        {"work_id": "https://openalex.org/W2", "is_junk": False, "in_labelled_sample": True},
        {"work_id": "https://openalex.org/W3", "is_junk": True, "in_labelled_sample": False},
    ]
    assert records.disagreeing_count == 1


def test_a_record_in_both_sets_counts_as_labelled() -> None:
    labels = JunkLabels(
        codebook_sha256="c" * 64,
        items={"S0001": label(1, "enrichment", "junk"), "S0002": label(1, "labelled", "junk")},
    )

    assert build_record_labels(labels).table["in_labelled_sample"].to_list() == [True]


def test_a_source_is_the_issn_then_the_source_name_then_the_work() -> None:
    records = pl.DataFrame(
        {
            "work_id": ["W1", "W2", "W3", "W4"],
            "issn_l": ["1234-5678", None, None, ""],
            "source_name": ["Journal A", "Journal B", None, " "],
        }
    )

    assert build_sources(records) == ["1234-5678", "Journal B", "W3", "W4"]


def test_folds_are_the_sources_hash_and_never_split_a_source() -> None:
    sources = [f"source-{index % 7}" for index in range(70)]

    outer = build_folds(sources, folds=5, depth=0)
    inner = build_folds(sources, folds=5, depth=1)

    expected = int(hashlib.sha256(b"source-3").hexdigest(), 16) % 5
    assert outer[3] == expected
    for source in set(sources):
        positions = [index for index, name in enumerate(sources) if name == source]
        assert len(set(outer[positions])) == 1
        assert len(set(inner[positions])) == 1


@pytest.mark.parametrize("count", [1, 33, 100, 200])
def test_the_cut_off_is_the_lowest_score_clearing_97_percent_of_the_abstracts(
    count: int,
) -> None:
    generator = np.random.default_rng(count)
    scores = np.round(generator.random(count), 2)

    cut_off = compute_cut_off(scores, target=0.97)

    assert (scores < cut_off).mean() >= 0.97
    assert (scores < np.nextafter(cut_off, -np.inf)).mean() < 0.97


def test_each_folds_cut_off_is_set_on_the_other_folds_abstracts_alone() -> None:
    generator = np.random.default_rng(3)
    probability = generator.random(500)
    is_junk = generator.random(500) < 0.1
    folds = np.arange(500) % 5

    cut_offs = set_cut_offs(probability, is_junk, folds, fold_count=5, target=0.97)

    for fold in range(5):
        others = probability[(folds != fold) & ~is_junk]
        assert cut_offs.by_fold[fold] == compute_cut_off(others, target=0.97)
    assert cut_offs.final == compute_cut_off(probability[~is_junk], target=0.97)
    assert cut_offs.lowest == min([*cut_offs.by_fold.values(), cut_offs.final])


def test_ls_records_are_called_by_their_folds_cut_off_and_the_others_by_the_final_one() -> None:
    cut_offs = CutOffs(by_fold={0: 0.2, 1: 0.8}, final=0.5)
    probability = np.array([0.3, 0.3, 0.3, 0.9, np.nan])
    folds = np.array([0, 1, -1, -1, -1])

    called = call_records(probability, folds, cut_offs)

    assert called.tolist() == [True, False, False, True, False]


def test_a_record_of_l_whose_fold_has_no_cut_off_is_refused() -> None:
    with pytest.raises(ValueError, match="no cut-off"):
        call_records(np.array([0.5]), np.array([2]), CutOffs(by_fold={0: 0.2}, final=0.5))


def test_a_record_is_junk_only_when_both_models_reach_their_cut_offs() -> None:
    screen_cut_offs = CutOffs(by_fold={0: 0.4}, final=0.5)
    confirm_cut_offs = CutOffs(by_fold={0: 0.6}, final=0.7)
    screen = np.array([0.45, 0.45, 0.9, 0.9, 0.9])
    confirm = np.array([0.65, 0.5, 0.65, 0.8, np.nan])
    folds = np.array([0, 0, -1, -1, -1])

    called = call_by_agreement(screen, confirm, folds, screen_cut_offs, confirm_cut_offs)

    assert called.tolist() == [True, False, False, True, False]


def test_the_quality_bar_needs_both_point_estimates() -> None:
    is_junk = np.array([True] * 10 + [False] * 100)
    called = is_junk.copy()
    called[:3] = False
    called[10:13] = True

    quality = compute_quality(is_junk, called)

    assert quality.sensitivity.share == pytest.approx(0.7)
    assert quality.specificity.share == pytest.approx(0.97)
    assert quality.meets_bar
    called[3] = False
    assert not compute_quality(is_junk, called).meets_bar


def test_false_alarms_count_only_abstracts_and_each_of_a_records_fields() -> None:
    records = pl.DataFrame(
        {
            "is_junk": [False, False, False, True],
            "predicted_junk": [True, False, False, True],
            "decade": [1990, 1990, 2000, 2000],
            "fields": [["ai", "compsci"], ["ai"], ["compsci"], ["ai"]],
        }
    )

    by_field = compute_false_alarms(records, "fields")
    by_decade = compute_false_alarms(records, "decade")

    assert {field: (rate.count, rate.total) for field, rate in by_field.items()} == {
        "ai": (1, 2),
        "compsci": (1, 2),
    }
    assert {decade: (rate.count, rate.total) for decade, rate in by_decade.items()} == {
        "1990": (1, 2),
        "2000": (0, 1),
    }


def test_a_decades_correction_is_the_mean_label_minus_call_with_its_interval() -> None:
    items = pl.DataFrame(
        {
            "decade": [1990] * 4 + [2000] * 2,
            "is_junk_usage": [True, True, False, False, False, False],
            "predicted_junk": [True, False, False, False, True, False],
        }
    )

    corrections = compute_corrections(items)

    assert corrections[1990].mean == pytest.approx(0.25)
    assert corrections[1990].items == 4
    margin = 1.959963984540054 * np.std([0, 1, 0, 0], ddof=1) / 2
    assert corrections[1990].low == pytest.approx(0.25 - margin)
    assert corrections[1990].high == pytest.approx(0.25 + margin)
    assert corrections[2000].mean == pytest.approx(-0.5)
