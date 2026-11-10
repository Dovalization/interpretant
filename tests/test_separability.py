"""Tests for field separability by classifier, on synthetic fields."""

from __future__ import annotations

import random
from pathlib import Path

from interpretant.drift.separability import (
    assignment_shares,
    decade_samples,
    fit_fields,
    pair_accuracy,
    separability_runs,
    split,
)

VOCABULARY = {
    "mind": ["consciousness", "perception", "intentionality", "phenomenology", "experience"],
    "machine": ["algorithm", "network", "training", "dataset", "optimisation"],
    "society": ["community", "institution", "survey", "interview", "participant"],
}
SHARED = ["the", "of", "study", "we", "show"]


def abstracts(field: str, count: int, seed: int, borrowed: str | None = None) -> list[str]:
    """Synthetic abstracts from a field's words, optionally half borrowed from another field."""
    rng = random.Random(f"{field}{seed}{borrowed}")
    words = VOCABULARY[field] + (VOCABULARY[borrowed] if borrowed else [])
    return [" ".join(rng.choices(words + SHARED, k=20)) + f" n{n}" for n in range(count)]


def write_decade(root: Path, decade: int, borrowed: dict[str, str] | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for field in VOCABULARY:
        lines = abstracts(field, 120, decade, (borrowed or {}).get(field))
        (root / f"{field}_{decade}.txt").write_text("\n".join(lines), encoding="utf-8")


def test_decade_samples_removes_shared_abstracts_and_skips_small_fields(tmp_path: Path) -> None:
    write_decade(tmp_path, 1990)
    shared = (tmp_path / "mind_1990.txt").read_text().splitlines()[0]
    with (tmp_path / "machine_1990.txt").open("a") as cell:
        cell.write("\n" + shared)
    (tmp_path / "tiny_1990.txt").write_text("one abstract\n")

    samples = decade_samples(tmp_path, 1990, seed=1, per_field=100)

    assert sorted(samples) == ["machine", "mind", "society"]
    assert all(len(texts) == 100 for texts in samples.values())
    assert shared not in samples["mind"] + samples["machine"]
    assert samples == decade_samples(tmp_path, 1990, seed=1, per_field=100)


def test_distinct_fields_separate_and_identical_ones_do_not() -> None:
    mind, machine = abstracts("mind", 100, 1), abstracts("machine", 100, 1)

    assert pair_accuracy(mind, machine, seed=1) > 0.95
    assert pair_accuracy(mind, abstracts("mind", 100, 2), seed=1) < 0.75


def test_assignment_shares_sum_to_one_per_field() -> None:
    samples = {field: abstracts(field, 100, 1) for field in VOCABULARY}
    train, test = split(samples)

    rows = assignment_shares(fit_fields(train), test)

    for field in VOCABULARY:
        shares = {row["assigned_field"]: row["share"] for row in rows if row["text_field"] == field}
        assert sum(shares.values()) == 1.0
        assert shares[field] > 0.9


def test_runs_cover_every_pair_and_the_cross_decade_classifier_sees_a_move(tmp_path: Path) -> None:
    write_decade(tmp_path, 1990)
    write_decade(tmp_path, 2020, borrowed={"mind": "society"})

    pairs, assignments = separability_runs(
        tmp_path, [1990, 2020], seeds=[1], cross=(1990, 2020), per_field=100
    )

    assert len(pairs) == 2 * 3
    cross = assignments[(assignments.train_decade == 1990) & (assignments.test_decade == 2020)]
    within = assignments[(assignments.train_decade == 1990) & (assignments.test_decade == 1990)]

    def share(table: object, text: str, assigned: str) -> float:
        rows = table[(table.text_field == text) & (table.assigned_field == assigned)]  # type: ignore[index]
        return float(rows.share.iloc[0])

    assert share(cross, "mind", "society") > share(within, "mind", "society")
    accuracy = pairs.set_index(["decade", "field_a", "field_b"]).accuracy
    assert accuracy[(2020, "mind", "society")] < accuracy[(1990, "mind", "society")]
