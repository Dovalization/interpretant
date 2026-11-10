"""Sense-label instrument on synthetic inputs: sense lists, blocks, answer checks and measures.
No labeller is called."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from interpretant.drift.labels import (
    REPEAT_SUFFIX,
    Item,
    Labeller,
    Sense,
    SenseList,
    blocks,
    distribution,
    divergence,
    load_sense_lists,
    parsed,
    prompt,
    shown,
)
from interpretant.drift.pairs import Usage

SENSES = SenseList(
    "attention",
    (Sense("cognitive", "a mind's focus", "pay attention"), Sense("mechanism", "in nets", "self")),
)


def items(count: int) -> list[Item]:
    return [
        Item(f"u{index}", Usage(f"usage {index} attention here", 8, 17)) for index in range(count)
    ]


def test_sense_lists_load_from_toml_and_always_offer_other_and_unclear(tmp_path: Path) -> None:
    path = tmp_path / "lists.toml"
    path.write_text(
        '[attention]\nword = "attention"\n'
        '[[attention.senses]]\nkey = "cognitive"\ndefinition = "focus"\nexample = "pay it"\n',
        encoding="utf-8",
    )
    lists = load_sense_lists(path)
    assert lists["attention"].labels == ["cognitive", "other", "unclear"]


def test_the_shown_usage_marks_the_word_and_keeps_about_the_limit_around_it() -> None:
    text = " ".join(f"w{index}" for index in range(300)) + " target " + "x " * 300
    start = text.index("target")
    result = shown(Usage(text, start, start + len("target")), limit=11).split()
    assert result[5] == "⟦target⟧"
    assert len(result) == 11


def test_blocks_hold_every_item_once_and_each_repeat_outside_its_own_block() -> None:
    result = blocks(items(95), size=20, repeat_share=0.1, seed=1)
    keys = [item.key for block in result for item in block]
    originals = [key for key in keys if not key.endswith(REPEAT_SUFFIX)]
    repeats = [key for key in keys if key.endswith(REPEAT_SUFFIX)]
    assert sorted(originals) == sorted(f"u{index}" for index in range(95))
    assert len(repeats) == 10
    home = {item.key: index for index, block in enumerate(result) for item in block}
    assert all(home[key] != home[key.removesuffix(REPEAT_SUFFIX)] for key in repeats)


def test_blocks_are_reproducible_from_their_seed() -> None:
    first = blocks(items(50), size=20, repeat_share=0.1, seed="a")
    second = blocks(items(50), size=20, repeat_share=0.1, seed="a")
    assert [[i.key for i in b] for b in first] == [[i.key for i in b] for b in second]


def test_the_prompt_numbers_the_usages_and_lists_every_sense() -> None:
    text = prompt(SENSES, items(2))
    assert "- mechanism: in nets (Example: self)" in text
    assert "1. usage 0 ⟦attention⟧ here" in text
    assert "2. usage 1 ⟦attention⟧ here" in text


def answer(labels: list[tuple[int, str]]) -> dict[str, object]:
    return {"structured_output": {"labels": [{"item": n, "label": label} for n, label in labels]}}


def test_a_complete_answer_is_read_in_item_order() -> None:
    assert parsed(answer([(2, "other"), (1, "cognitive")]), SENSES, 2) == ["cognitive", "other"]


@pytest.mark.parametrize(
    "labels",
    [
        [(1, "cognitive")],
        [(1, "cognitive"), (1, "mechanism")],
        [(1, "cognitive"), (2, "visual")],
        [(1, "cognitive"), (3, "mechanism")],
    ],
)
def test_an_answer_missing_repeating_or_inventing_items_is_rejected(
    labels: list[tuple[int, str]],
) -> None:
    assert parsed(answer(labels), SENSES, 2) is None


def test_an_error_response_is_rejected() -> None:
    assert parsed({"is_error": True}, SENSES, 1) is None


def test_a_stored_valid_response_is_reused_without_calling_the_labeller(tmp_path: Path) -> None:
    record = tmp_path / "block.json"
    record.write_text(json.dumps([answer([(1, "mechanism"), (2, "unclear")])]), encoding="utf-8")
    labeller = Labeller(model="unused", effort="low", attempts=0)
    assert labeller.label(SENSES, items(2), record) == {"u0": "mechanism", "u1": "unclear"}


def test_the_distribution_leaves_out_unclear_and_failed_labels() -> None:
    shares = distribution(["cognitive", "cognitive", "other", "unclear", "failed"], SENSES)
    assert shares.tolist() == pytest.approx([2 / 3, 0.0, 1 / 3])


def test_divergence_is_zero_for_equal_distributions_and_one_for_disjoint_ones() -> None:
    assert divergence(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == pytest.approx(0.0)
    assert divergence(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)
