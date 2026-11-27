"""The junk gate's scorer (`scripts/review/score_junk_gate.py`): adjudication, labels, gate."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import polars as pl
import pytest

from interpretant.catalog.config import read_catalog
from interpretant.catalog.junk_gate import JunkLabel, JunkLabels, Run3Verdict, save_junk_labels
from interpretant.catalog.junk_gate_tables import (
    JUNK_CELLS_TABLE_COLUMNS,
    JUNK_SCORES_COLUMNS,
    SURFACE_FEATURES,
    save_junk_cells,
    save_junk_scores,
)
from interpretant.review.sampling_plans import ConfusionMatrix
from tests.junk_gate_records import SEALED, SyntheticRecord, work_id, write_records, write_words

REPOSITORY = Path(__file__).resolve().parents[1]
CODEBOOK_SHA256 = "c" * 64


def load_scorer() -> ModuleType:
    """The scoring script as a module."""
    path = REPOSITORY / "scripts" / "review" / "score_junk_gate.py"
    spec = importlib.util.spec_from_file_location("score_junk_gate_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["score_junk_gate_script"] = module
    spec.loader.exec_module(module)
    return module


scorer = load_scorer()


def answer(verdict: str = "abstract", sentence: str = "none", kind: str = "none") -> dict[str, str]:
    return {
        "note": "the judge's note",
        "verdict": verdict,
        "kind": kind,
        "sentence": sentence,
        "damaged": "no",
    }


def write_folder(
    folder: Path,
    key: dict[str, Any],
    first: dict[str, dict[str, str]],
    second: dict[str, dict[str, str]] | None = None,
    adjudicated: dict[str, dict[str, str]] | None = None,
    codebook_sha256: str = CODEBOOK_SHA256,
) -> None:
    """A draw folder: its key, the judges' answers and manifests, and the adjudicator's."""
    verdicts = folder / "verdicts"
    verdicts.mkdir(parents=True)
    (folder / "key.json").write_text(json.dumps(key))
    answers = {"judge_A_part_00": first, "judge_B_part_00": second or first}
    if adjudicated is not None:
        answers["opus_adjudication"] = adjudicated
    for name, answered in answers.items():
        (verdicts / f"{name}.json").write_text(json.dumps(answered))
        manifest = {"model": "claude-sonnet-5-5", "codebook_sha256": codebook_sha256}
        (verdicts / f"{name}.manifest.json").write_text(json.dumps(manifest))


RUN3_IDS = [f"r{number:03d}" for number in range(200)]
NOT_ABSTRACTS = RUN3_IDS[100:130]
UNSURE = RUN3_IDS[130:133]


def run3_author() -> list[Run3Verdict]:
    """Run 3's 200 records: 30 not abstracts, 3 unsure, the rest (r018 among them) abstracts."""
    verdicts = []
    for item_id in RUN3_IDS:
        verdict = "not_abstract" if item_id in NOT_ABSTRACTS else "abstract"
        if item_id in UNSURE:
            verdict = "unsure"
        verdicts.append(
            Run3Verdict(id=item_id, work_id=f"W{item_id}", decade=1950, field="ai", verdict=verdict)
        )
    return verdicts


def write_run3(folder: Path, junk: set[str], mixed: frozenset[str] = frozenset()) -> None:
    """Run 3's folder where both judges call ``junk`` junk and ``mixed`` mixed."""
    key = {item_id: {"set": "run3", "work_id": f"W{item_id}"} for item_id in RUN3_IDS}
    answers = {
        item_id: answer("junk" if item_id in junk else "mixed" if item_id in mixed else "abstract")
        for item_id in RUN3_IDS
    }
    write_folder(folder, key, answers)


# The judges find r018 and 27 of the 30 other junk, call 4 abstracts junk and 3 unsure junk
JUDGED_JUNK = {"r018", *NOT_ABSTRACTS[:27], "r150", "r151", "r152", "r153", *UNSURE}


def test_the_three_bars_are_counted_on_the_197_records(tmp_path: Path) -> None:
    write_run3(tmp_path, JUDGED_JUNK)

    results = scorer.calibrate(tmp_path, run3_author(), {})

    assert results["records"] == 197
    assert results["bars"] == {
        "agreement": {"count": 190, "of": 197, "bar": 178, "passes": True},
        "junk_found": {"count": 28, "of": 31, "bar": 27, "passes": True},
        "abstracts_called_junk": {"count": 4, "of": 166, "bar": 6, "passes": True},
    }
    assert results["passes"]
    assert results["disputed"] == ["r127", "r128", "r129", "r150", "r151", "r152", "r153"]


def test_a_bar_fails_when_too_little_junk_is_found(tmp_path: Path) -> None:
    write_run3(tmp_path, JUDGED_JUNK - {NOT_ABSTRACTS[0], NOT_ABSTRACTS[1]})

    results = scorer.calibrate(tmp_path, run3_author(), {})

    assert results["bars"]["junk_found"] == {"count": 26, "of": 31, "bar": 27, "passes": False}
    assert not results["passes"]


def test_r018_counts_as_junk_whatever_the_verdicts_file_says(tmp_path: Path) -> None:
    write_run3(tmp_path, JUDGED_JUNK - {"r018"})

    results = scorer.calibrate(tmp_path, run3_author(), {})

    assert "r018" in results["disputed"]
    assert results["bars"]["junk_found"]["count"] == 27


def test_the_unsure_records_are_left_out_and_a_mixed_verdict_is_not_junk(tmp_path: Path) -> None:
    write_run3(tmp_path, JUDGED_JUNK, mixed=frozenset({"r160", NOT_ABSTRACTS[29]}))

    results = scorer.calibrate(tmp_path, run3_author(), {})

    assert not set(UNSURE) & set(results["disputed"])
    assert "r160" not in results["disputed"]
    assert NOT_ABSTRACTS[29] in results["disputed"]
    assert results["bars"]["agreement"]["count"] == 190


def test_a_reference_other_than_step_ones_is_refused(tmp_path: Path) -> None:
    write_run3(tmp_path, JUDGED_JUNK)

    with pytest.raises(SystemExit, match="196 records and 30 junk"):
        scorer.calibrate(
            tmp_path,
            [verdict for verdict in run3_author() if verdict.id != NOT_ABSTRACTS[0]],
            {},
        )


def test_the_second_look_stands_where_given_and_the_first_verdict_elsewhere(
    tmp_path: Path,
) -> None:
    write_run3(tmp_path, JUDGED_JUNK)
    second_look = {"r150": "junk", NOT_ABSTRACTS[27]: "abstract", "r151": "unsure"}

    results = scorer.calibrate(tmp_path, run3_author(), second_look)

    assert results["second_look"] == {
        "records": 197,
        "second_looks": 2,
        "confusion": {
            "true_positives": 29,
            "false_negatives": 2,
            "false_positives": 3,
            "true_negatives": 163,
        },
    }
    assert results["bars"]["junk_found"]["count"] == 28


def test_the_adjudicator_settles_only_junk_and_other_sentence_differences() -> None:
    assert scorer.needs_adjudicator(answer("junk"), answer("abstract"))
    assert scorer.needs_adjudicator(answer("junk"), answer("mixed"))
    assert scorer.needs_adjudicator(answer(sentence="other"), answer(sentence="own"))
    assert not scorer.needs_adjudicator(answer("abstract"), answer("mixed"))
    assert not scorer.needs_adjudicator(answer(sentence="own"), answer(sentence="none"))
    assert not scorer.needs_adjudicator(answer("junk", kind="fragment"), answer("junk"))


def sample_key(count: int) -> dict[str, dict[str, Any]]:
    return {
        f"S{number:04d}": {
            "set": "labelled",
            "work_id": f"W{number}",
            "part": "ai_1990",
            "decade": 1990,
            "word": "attention",
            "content_key": f"{number:020d}",
        }
        for number in range(1, count + 1)
    }


def test_pending_lists_the_items_the_adjudicator_must_settle(tmp_path: Path) -> None:
    key = sample_key(4)
    first = {
        "S0001": answer("junk", "other"),
        "S0002": answer("abstract", "own"),
        "S0003": answer("abstract", "other"),
        "S0004": answer("abstract", "own"),
    }
    second = {
        "S0001": answer("abstract", "other"),
        "S0002": answer("mixed", "own"),
        "S0003": answer("abstract", "own"),
        "S0004": answer("abstract", "own", kind="other"),
    }
    write_folder(tmp_path, key, first, second)
    items = [{"id": item_id, "text": f"text of {item_id}"} for item_id in key]
    (tmp_path / "batch_00.jsonl").write_text("".join(json.dumps(item) + "\n" for item in items))

    summary = scorer.write_pending(tmp_path)

    rows = [json.loads(line) for line in (tmp_path / "adjudicate.jsonl").read_text().splitlines()]
    assert [row["id"] for row in rows] == ["S0001", "S0003"]
    assert rows[0]["judge_1"] == first["S0001"]
    assert rows[0]["judge_2"] == second["S0001"]
    assert summary == "judges agree on 2 of 4; 2 items for the adjudicator"
    final, _, _ = scorer.final_answers(tmp_path, key)
    assert final["S0002"] == first["S0002"]


def test_labels_hold_each_final_answer_without_its_note(tmp_path: Path) -> None:
    key = sample_key(2)
    key["S0002"] |= {"set": "enrichment", "stratum": "list_like"}
    first = {"S0001": answer("junk", "other"), "S0002": answer("abstract", "own")}
    second = {"S0001": answer("abstract", "own"), "S0002": answer("abstract", "own")}
    write_folder(tmp_path / "sample", key, first, second, {"S0001": answer("mixed", "other")})
    confirmation = {
        "items": {"C0001": sample_key(1)["S0001"] | {"set": "confirmation"}},
        "cells": [{"word": "attention", "decade": 1990, "items": ["C0001", "S0001"]}],
    }
    write_folder(tmp_path / "confirmation", confirmation, {"C0001": answer()})

    labels = scorer.build_labels([tmp_path / "sample", tmp_path / "confirmation"], CODEBOOK_SHA256)

    assert labels.codebook_sha256 == CODEBOOK_SHA256
    assert sorted(labels.items) == ["C0001", "S0001", "S0002"]
    assert labels.items["S0001"].verdict == "mixed"
    assert labels.items["S0002"].stratum == "list_like"
    assert labels.items["S0001"].stratum is None
    assert "note" not in labels.items["S0001"].model_dump()


def test_labels_are_refused_while_an_item_waits_for_the_adjudicator(tmp_path: Path) -> None:
    write_folder(tmp_path, sample_key(1), {"S0001": answer("junk")}, {"S0001": answer()})

    with pytest.raises(SystemExit, match=r"1 items .* still need the adjudicator"):
        scorer.build_labels([tmp_path], CODEBOOK_SHA256)


def test_labels_are_refused_for_answers_made_under_another_codebook(tmp_path: Path) -> None:
    write_folder(tmp_path, sample_key(1), {"S0001": answer()}, codebook_sha256="d" * 64)

    with pytest.raises(SystemExit, match="another codebook"):
        scorer.build_labels([tmp_path], CODEBOOK_SHA256)


def test_labels_are_refused_without_a_judges_manifest(tmp_path: Path) -> None:
    write_folder(tmp_path, sample_key(1), {"S0001": answer()})
    for manifest in (tmp_path / "verdicts").glob("*.manifest.json"):
        manifest.unlink()

    with pytest.raises(SystemExit, match="no judge's manifest"):
        scorer.build_labels([tmp_path], CODEBOOK_SHA256)


def label(number: int, item_set: str, verdict: str, work: int | None = None) -> JunkLabel:
    return JunkLabel(
        set=item_set,
        stratum="list_like" if item_set == "enrichment" else None,
        work_id=f"W{number if work is None else work}",
        part="ai_1990",
        decade=1990,
        word="attention",
        content_key=f"{number:020d}",
        verdict=verdict,
        kind="none",
        sentence="own",
        damaged="no",
    )


def fresh_labels() -> JunkLabels:
    """L with 20 junk, 20 abstracts and 5 mixed; E with 20 abstracts and 20 junk."""
    verdicts = [("labelled", "junk")] * 20 + [("labelled", "abstract")] * 20
    verdicts += [("labelled", "mixed")] * 5
    verdicts += [("enrichment", "abstract")] * 20 + [("enrichment", "junk")] * 20
    items = {
        f"S{number:04d}": label(number, item_set, verdict)
        for number, (item_set, verdict) in enumerate(verdicts, start=1)
    }
    return JunkLabels(codebook_sha256=CODEBOOK_SHA256, items=items)


def test_the_fresh_draw_takes_each_groups_size_the_same_way_each_time() -> None:
    labels = fresh_labels()

    drawn = scorer.draw_fresh(labels, scorer.FRESH_SEED)

    groups = [entry["group"] for entry in drawn]
    assert {group: groups.count(group) for group in set(groups)} == {
        "labelled_junk": 12,
        "labelled_abstract": 8,
        "enrichment_abstract": 10,
    }
    assert drawn == scorer.draw_fresh(labels, scorer.FRESH_SEED)
    assert drawn != scorer.draw_fresh(labels, scorer.REDRAW_SEED)
    for entry in drawn:
        chosen = labels.items[entry["id"]]
        assert (chosen.verdict == "junk") == (entry["group"] == "labelled_junk")
        assert (chosen.set == "enrichment") == (entry["group"] == "enrichment_abstract")
    assert groups != [group.name for group in scorer.FRESH_GROUPS for _ in range(group.size)]


def test_the_fresh_draw_takes_each_record_once() -> None:
    labels = fresh_labels()
    shared = {
        item_id: item.model_copy(update={"work_id": "W1"})
        for item_id, item in labels.items.items()
        if item.set == "labelled" and item.verdict == "junk"
    }
    labels = labels.model_copy(update={"items": labels.items | shared})

    with pytest.raises(SystemExit, match="labelled_junk has 1 records, not 12"):
        scorer.draw_fresh(labels, scorer.FRESH_SEED)


def test_the_authors_fresh_answers_pass_at_27_of_30() -> None:
    labels = fresh_labels()
    drawn = scorer.draw_fresh(labels, scorer.FRESH_SEED)
    author = {
        entry["id"]: "junk" if labels.items[entry["id"]].verdict == "junk" else "abstract"
        for entry in drawn
    }
    first, second, third, fourth = (entry["id"] for entry in drawn[:4])
    author[first] = "junk" if author[first] == "abstract" else "abstract"
    author[second] = "unsure"

    results = scorer.score_fresh(drawn, labels, author)

    assert (results["items"], results["agree"], results["unsure"]) == (30, 28, 1)
    assert results["passes"]
    assert results["disagreements"] == sorted([first, second])
    author[third] = author[fourth] = "unsure"
    assert not scorer.score_fresh(drawn, labels, author)["passes"]


def test_the_authors_fresh_answers_must_be_on_the_drawn_records() -> None:
    labels = fresh_labels()
    drawn = scorer.draw_fresh(labels, scorer.FRESH_SEED)

    with pytest.raises(SystemExit, match="not on the drawn items"):
        scorer.score_fresh(drawn, labels, {"S0001": "junk"})


def test_the_authors_answers_are_read_as_verdicts_or_the_pages_answers(tmp_path: Path) -> None:
    path = tmp_path / "answers.json"
    path.write_text(json.dumps({"r001": "junk", "r002": {"verdict": "abstract", "note": "x"}}))

    assert scorer.load_author_answers(path) == {"r001": "junk", "r002": "abstract"}
    assert scorer.load_author_answers(tmp_path / "missing.json") == {}


# The gate's scoring: the decade rule, the cells' confirmation and the exclusions

DECADES = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]
PERFECT_JUDGES = ConfusionMatrix(
    true_positives=28, false_negatives=0, false_positives=0, true_negatives=169
)


def decade_labels(junk_by_decade: dict[int, int]) -> dict[str, JunkLabel]:
    """L's 500 usages in each decade, the first ones junk: half by verdict, half by sentence."""
    items = {}
    for decade, junk in junk_by_decade.items():
        for index in range(500):
            is_junk_record = index < junk and index % 2 == 0
            is_other_sentence = index < junk and index % 2 == 1
            items[f"L{decade}{index:03d}"] = JunkLabel(
                set="labelled",
                stratum=None,
                work_id=f"https://openalex.org/W{decade}{index:03d}",
                part=f"ai_{decade}",
                decade=decade,
                word="attention",
                content_key=f"{index:020d}",
                verdict="junk" if is_junk_record else "abstract",
                kind="fragment" if is_junk_record else "none",
                sentence="other" if is_other_sentence else "own",
                damaged="no",
            )
    return items


def every_decade_at(junk: int, **changes: int) -> JunkLabels:
    counts = dict.fromkeys(DECADES, junk) | {
        int(name[1:]): count for name, count in changes.items()
    }
    return JunkLabels(codebook_sha256=CODEBOOK_SHA256, items=decade_labels(counts))


def test_a_decade_passes_the_bar_at_52_junk_usages_and_fails_at_53() -> None:
    passing = scorer.score_decades(every_decade_at(45, d1950=52), PERFECT_JUDGES)
    failing = scorer.score_decades(every_decade_at(45, d1950=53), PERFECT_JUDGES)

    assert passing["decades"]["1950"]["junk"] == 52
    assert passing["decades"]["1950"]["passes"]
    assert passing["passes"]
    assert failing["decades"]["1950"]["junk"] == 53
    assert not failing["decades"]["1950"]["passes_bar"]
    assert failing["decades"]["1950"]["passes_comparison"]
    assert not failing["passes"]


def test_a_decade_within_the_bar_fails_when_fishers_test_finds_more_junk_than_elsewhere() -> None:
    results = scorer.score_decades(every_decade_at(10, d1990=50), PERFECT_JUDGES)

    nineties = results["decades"]["1990"]
    assert nineties["passes_bar"]
    assert nineties["fisher_p"] < 0.05 / 8
    assert not nineties["passes_comparison"]
    assert not results["passes"]
    assert results["decades"]["1950"]["passes"]


def test_each_decades_share_has_its_interval_and_its_correction_for_the_judges() -> None:
    labels = every_decade_at(40)

    results = scorer.score_decades(labels, PERFECT_JUDGES)

    fifties = results["decades"]["1950"]
    assert fifties["share"] == 0.08
    assert fifties["interval"][0] < 0.08 < fifties["interval"][1]
    assert (
        fifties["corrected_interval"][0] < fifties["corrected"] < fifties["corrected_interval"][1]
    )
    assert abs(fifties["corrected"] - 0.08) < 0.02
    assert results == scorer.score_decades(labels, PERFECT_JUDGES)


def test_a_decade_short_of_its_500_labelled_usages_is_refused() -> None:
    labels = every_decade_at(10)
    labels = labels.model_copy(update={"items": dict(list(labels.items.items())[1:])})

    with pytest.raises(SystemExit, match="without 500 labelled usages"):
        scorer.score_decades(labels, PERFECT_JUDGES)


GATE_RECORDS = [
    SyntheticRecord(number, 1995, is_junk=number <= 4, has_network=number <= 6)
    for number in range(1, 11)
] + [
    SyntheticRecord(number, 2005, is_junk=number <= 1003, has_network=number <= 1005)
    for number in range(1001, 1011)
]


def confirmation_labels(junk: int) -> dict[str, JunkLabel]:
    return {
        f"C{number:04d}": JunkLabel(
            set="confirmation",
            stratum=None,
            work_id=f"https://openalex.org/W9{number:04d}",
            part="ai_1990",
            decade=1990,
            word=SEALED,
            content_key=f"{number:020d}",
            verdict="junk" if number <= junk else "abstract",
            kind="fragment" if number <= junk else "none",
            sentence="own",
            damaged="no",
        )
        for number in range(1, 41)
    }


def write_gate(folder: Path, *, nineties_junk: int, confirmed_junk: int) -> SimpleNamespace:
    """A checkout's catalog with the stage's outputs, the labels, the calibration and the key.

    The screen flags *network* (held out) in the 1990s; the 1990s hold ``nineties_junk`` junk
    usages of L's 500 (the other decades 20), and the flagged cell ``confirmed_junk`` of its 40.
    """
    shutil.copy(REPOSITORY / "catalog.yaml", folder / "catalog.yaml")
    catalog = read_catalog(folder / "catalog.yaml")
    write_words(catalog)
    write_records(catalog, GATE_RECORDS)
    save_junk_scores(
        catalog,
        pl.DataFrame(
            {
                "work_id": sorted(work_id(record.number) for record in GATE_RECORDS),
                **{name: [0.0] * len(GATE_RECORDS) for name in SURFACE_FEATURES},
                "score": [0.5] * len(GATE_RECORDS),
                "predicted_junk": [
                    record.is_junk
                    for record in sorted(GATE_RECORDS, key=lambda record: work_id(record.number))
                ],
                "fold": [None] * len(GATE_RECORDS),
            },
            schema=JUNK_SCORES_COLUMNS,
        ),
    )
    cells = [("attention", 1990, 10, 4), ("attention", 2000, 10, 3)]
    cells += [("network", 1990, 6, 4), ("network", 2000, 5, 3)]
    save_junk_cells(
        catalog,
        pl.DataFrame(
            [
                (word, decade, drawn, junk, 0.4, None, word == "network" and decade == 1990, True)
                for word, decade, drawn, junk in cells
            ],
            schema=JUNK_CELLS_TABLE_COLUMNS,
            orient="row",
        ),
    )
    labels = every_decade_at(20, d1990=nineties_junk).items | confirmation_labels(confirmed_junk)
    save_junk_labels(catalog, JunkLabels(codebook_sha256=CODEBOOK_SHA256, items=labels))
    catalog.junk_gate_metrics.path.parent.mkdir(parents=True, exist_ok=True)
    catalog.junk_gate_metrics.path.write_text(
        json.dumps(
            {
                "classifier": {
                    "sensitivity": {"share": 0.8},
                    "specificity": {"share": 0.98},
                    "bar": {"sensitivity": 0.7, "specificity": 0.97},
                    "meets_bar": True,
                }
            }
        )
    )
    gate_folder = folder / "data" / "review" / "junk_gate"
    gate_folder.mkdir(parents=True, exist_ok=True)
    (gate_folder / "calibration.json").write_text(
        json.dumps({"second_look": {"confusion": vars(PERFECT_JUDGES)}})
    )
    confirmation = gate_folder / "confirmation"
    confirmation.mkdir()
    (confirmation / "key.json").write_text(
        json.dumps(
            {
                "items": {},
                "cells": [
                    {"word": SEALED, "decade": 1990, "items": sorted(confirmation_labels(0))}
                ],
            }
        )
    )
    return SimpleNamespace(catalog=catalog, gate_folder=gate_folder, confirmation=confirmation)


def test_a_failed_cell_excludes_its_words_predicted_junk_in_every_decade_and_nothing_else(
    tmp_path: Path,
) -> None:
    gate = write_gate(tmp_path, nineties_junk=20, confirmed_junk=9)

    results, exclusions = scorer.score_gate(gate.catalog, gate.gate_folder, gate.confirmation)

    assert results["cells"]["confirmed"] == [
        {"word": SEALED, "decade": 1990, "junk": 9, "of": 40, "fails": True}
    ]
    assert exclusions.rows() == sorted(
        (work_id(number), "network") for number in [1, 2, 3, 4, 1001, 1002, 1003]
    )
    assert results["exclusions"]["scope"] == "the failed cells' words"
    assert results["exclusions"]["by_word_and_decade"] == {SEALED: {"1990": 4, "2000": 3}}
    assert results["decade_rule"]["passes"]
    assert not results["passes"]


def test_a_failed_decade_excludes_predicted_junk_of_every_word_in_every_decade(
    tmp_path: Path,
) -> None:
    gate = write_gate(tmp_path, nineties_junk=60, confirmed_junk=8)

    results, exclusions = scorer.score_gate(gate.catalog, gate.gate_folder, gate.confirmation)

    assert not results["decade_rule"]["decades"]["1990"]["passes"]
    assert not results["cells"]["confirmed"][0]["fails"]
    assert results["exclusions"]["scope"] == "every word"
    junk = [1, 2, 3, 4, 1001, 1002, 1003]
    assert exclusions.rows() == sorted(
        (work_id(number), word) for number in junk for word in ("attention", "network")
    )
    assert results["exclusions"]["by_word_and_decade"] == {
        "attention": {"1990": 4, "2000": 3},
        SEALED: {"1990": 4, "2000": 3},
    }
    assert "network" not in json.dumps(results)


def test_the_gate_passes_with_every_decade_and_cell_passing_and_excludes_nothing(
    tmp_path: Path,
) -> None:
    gate = write_gate(tmp_path, nineties_junk=20, confirmed_junk=8)

    results, exclusions = scorer.score_gate(gate.catalog, gate.gate_folder, gate.confirmation)

    assert results["passes"]
    assert exclusions.height == 0
    assert results["exclusions"] == {"scope": "none", "usages": 0, "by_word_and_decade": {}}


def test_flagged_cells_need_their_confirmation(tmp_path: Path) -> None:
    gate = write_gate(tmp_path, nineties_junk=20, confirmed_junk=8)

    with pytest.raises(SystemExit, match="1 cells are flagged"):
        scorer.score_gate(gate.catalog, gate.gate_folder, None)


def git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def commit(root: Path, files: dict[str, str]) -> str:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
        git(root, "add", name)
    git(root, "commit", "--quiet", "-m", f"write {', '.join(files)}")
    return git(root, "rev-parse", "HEAD")


def test_the_gate_is_not_scored_by_an_uncommitted_scorer_nor_twice(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    git(root, "init", "--quiet")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test")
    sample = scorer.SAMPLE_FOLDER
    commit(
        root,
        {
            "docs/junk_gate.md": "the plan\n",
            "curated/judging/junk_gate/prompt.md": "the codebook\n",
            f"{sample}/key.json": "{}\n",
            f"{sample}/counts.json": "{}\n",
        },
    )
    amendment = commit(root, {"docs/junk_gate.md": "the plan, amended\n"})
    scorer_path = root / "scripts" / "review" / "score_junk_gate.py"
    scorer_path.parent.mkdir(parents=True)
    scorer_path.write_text("score\n")

    with pytest.raises(SystemExit, match=r"score_junk_gate\.py is not committed unchanged"):
        scorer.check_preregistration(root, amendment, None)

    commit(root, {"scripts/review/score_junk_gate.py": "score\n"})
    scorer.check_preregistration(root, amendment, None)
    (root / scorer.GATE_FOLDER / "results.json").write_text("{}\n")
    with pytest.raises(SystemExit, match="the gate is scored"):
        scorer.check_preregistration(root, amendment, None)
