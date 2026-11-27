"""The judges' merge (`review/verdicts.py`): part files, the adjudicator, the pending items."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from interpretant.review.judging import render
from interpretant.review.verdicts import VerdictsError, build_pending, load_parts, merge_verdicts

RIGHT = {"verdict": "right", "reason": "none", "note": "a use"}
WRONG = {"verdict": "wrong", "reason": "name", "note": "a name"}


def verdicts_differ(first_answer: dict[str, Any], second_answer: dict[str, Any]) -> bool:
    """The usages audit's rule: the adjudicator settles a difference of verdict."""
    return first_answer["verdict"] != second_answer["verdict"]


def test_a_judge_is_read_from_one_file_and_its_parts_without_the_manifests(tmp_path: Path) -> None:
    (tmp_path / "judge_A.json").write_text(json.dumps({"item-1": RIGHT}))
    (tmp_path / "judge_A_part_00.json").write_text(json.dumps({"item-2": WRONG}))
    (tmp_path / "judge_A_part_01.json").write_text(json.dumps({"item-3": RIGHT}))
    (tmp_path / "judge_A_part_00.manifest.json").write_text(json.dumps({"judged": 1}))
    (tmp_path / "judge_B_part_00.json").write_text(json.dumps({"item-4": RIGHT}))

    answers = load_parts(tmp_path, "judge_A")

    assert answers == {"item-1": RIGHT, "item-2": WRONG, "item-3": RIGHT}


def test_an_item_judged_twice_is_refused(tmp_path: Path) -> None:
    (tmp_path / "judge_A_part_00.json").write_text(json.dumps({"item-1": RIGHT}))
    (tmp_path / "judge_A_part_01.json").write_text(json.dumps({"item-1": WRONG}))

    with pytest.raises(VerdictsError, match="judge_A judged item-1 twice"):
        load_parts(tmp_path, "judge_A")


def test_an_adjudicator_who_answered_nothing_has_no_answers(tmp_path: Path) -> None:
    assert load_parts(tmp_path, "opus_adjudication") == {}


def test_agreement_keeps_the_first_judges_answer() -> None:
    second = RIGHT | {"note": "the second judge's note"}

    final, agreed_count, pending_ids = merge_verdicts(
        {"item-1": RIGHT}, {"item-1": second}, {}, differ=verdicts_differ
    )

    assert final == {"item-1": RIGHT}
    assert agreed_count == 1
    assert pending_ids == []


def test_a_difference_the_plan_leaves_alone_keeps_the_first_judges_answer() -> None:
    first = WRONG | {"reason": "name"}
    second = WRONG | {"reason": "other_word"}

    final, agreed_count, pending_ids = merge_verdicts(
        {"item-1": first}, {"item-1": second}, {"item-1": RIGHT}, differ=verdicts_differ
    )

    assert final == {"item-1": first}
    assert agreed_count == 1
    assert pending_ids == []


def test_a_difference_is_settled_by_the_adjudicator() -> None:
    settled = WRONG | {"note": "the adjudicator's note"}

    final, agreed_count, pending_ids = merge_verdicts(
        {"item-1": RIGHT}, {"item-1": WRONG}, {"item-1": settled}, differ=verdicts_differ
    )

    assert final == {"item-1": settled}
    assert agreed_count == 0
    assert pending_ids == []


def test_a_difference_without_the_adjudicators_answer_is_pending() -> None:
    final, agreed_count, pending_ids = merge_verdicts(
        {"item-1": RIGHT, "item-2": RIGHT},
        {"item-1": WRONG, "item-2": RIGHT},
        {},
        differ=verdicts_differ,
    )

    assert final == {"item-2": RIGHT}
    assert agreed_count == 1
    assert pending_ids == ["item-1"]


def test_items_are_merged_in_the_first_judges_order() -> None:
    first = {"item-3": RIGHT, "item-1": RIGHT, "item-2": RIGHT}
    second = {"item-1": WRONG, "item-2": RIGHT, "item-3": WRONG}

    final, _, pending_ids = merge_verdicts(first, second, {"item-1": WRONG}, differ=verdicts_differ)

    assert list(final) == ["item-1", "item-2"]
    assert pending_ids == ["item-3"]


def test_judges_who_answered_different_items_are_refused() -> None:
    with pytest.raises(VerdictsError, match=r"1 items have one judge's answer only.*item-2"):
        merge_verdicts(
            {"item-1": RIGHT, "item-2": RIGHT}, {"item-1": RIGHT}, {}, differ=verdicts_differ
        )


def test_pending_items_carry_both_judges_answers_in_the_order_asked() -> None:
    items = [{"id": "item-1", "text": "one"}, {"id": "item-2", "text": "two"}, {"id": "item-3"}]
    first = {"item-1": RIGHT, "item-2": RIGHT, "item-3": RIGHT}
    second = {"item-1": WRONG, "item-2": WRONG, "item-3": RIGHT}

    pending = build_pending(items, first, second, ["item-2", "item-1"])

    assert pending == [
        {"id": "item-2", "text": "two", "judge_1": RIGHT, "judge_2": WRONG},
        {"id": "item-1", "text": "one", "judge_1": RIGHT, "judge_2": WRONG},
    ]
    assert items[0] == {"id": "item-1", "text": "one"}


def test_a_pending_item_is_rendered_with_both_judges_answers() -> None:
    item = {
        "id": "item-1",
        "kind": "kept",
        "word": "network",
        "parts_of_speech": ["NOUN"],
        "sentence": "A ⟨t⟩network⟨/t⟩ of sensors.",
    }

    [pending] = build_pending([item], {"item-1": RIGHT}, {"item-1": WRONG}, ["item-1"])
    text = render(pending, "usages")

    assert f"Judge 1 answered: {json.dumps(RIGHT)}" in text
    assert f"Judge 2 answered: {json.dumps(WRONG)}" in text
