"""The junk gate's scorer (`scripts/review/score_junk_gate.py`): adjudication, bars and labels."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from interpretant.catalog.junk_gate import JunkLabel, JunkLabels, Run3Verdict

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
