"""Score round W (W1-W5): both instruments in one frame of reference, on whole cells and on
the venue-stable cell, as committed in docs/predictions_cross_perplexity.md.

Committed before it was run. Rebuilds HCI_v as round K' did (the venues with at least 20
cleaned HCI documents in both decades), refits round N's 1990s classifier and stops if its
assignment shares differ from round N's table, runs the planted check on 1990s abstracts for
each instrument, then scores the contrast. The cost tables are written to data/drift.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from interpretant.corpus.lemmas import LemmaMap
from interpretant.corpus.openalex import OpenAlexSource
from interpretant.corpus.phrases import PhraseModel
from interpretant.drift.direction import (
    ClassifierCosts,
    FieldCosts,
    LanguageModelCosts,
    assignment_runs,
    cost_matrix,
    frame_runs,
    planted_runs,
    present,
    relative_movement,
)
from interpretant.drift.perplexity import read_cell, remove_shared
from interpretant.predictions import report

RAW = Path("data/raw/openalex")
TEXTS = Path("data/phrased/openalex/texts")
OUTPUT = Path("data/drift")
SEEDS = (1, 2, 3, 4, 5)
EARLY, LATE = 1990, 2020
SHARED_MIN = 20
BEHAVIOURAL_SOCIAL = ("psychology", "cogpsych", "neuroscience", "sociology", "communication")
COMPUTING = ("compsci", "ai")
INSTRUMENTS: dict[str, Callable[[dict[str, list[str]]], FieldCosts]] = {
    "classifier": ClassifierCosts,
    "language models": LanguageModelCosts,
}


def phrased_lines(decade: int) -> tuple[list[str], list[str]]:
    """The HCI cell's documents as the lemmatised, phrased lines the models read, with ids."""
    cell = OpenAlexSource(RAW, "hci").cell(decade)
    lemmas = LemmaMap.load(Path("data/processed/lemmas.tsv"))
    phrases = PhraseModel.load(Path("data/lemmatised/phrases.model"))
    lines = [phrases.apply(lemmas.apply(document)) for document in cell.documents]
    _, on_disk = read_cell(TEXTS / f"hci_{decade}.txt")
    if set(lines) != set(on_disk):
        raise SystemExit(f"rebuilt {decade}s lines differ from {TEXTS}/hci_{decade}.txt")
    return lines, cell.ids


def sources(decade: int) -> dict[str, str | None]:
    path = RAW / "hci" / f"sources_{decade}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {row["id"]: row["source_id"] for row in rows}


def venue_stable() -> dict[int, list[str]]:
    """HCI_v per decade: abstracts from round K''s shared venues, shared abstracts removed."""
    lines = {decade: phrased_lines(decade) for decade in (EARLY, LATE)}
    venue_of = {decade: sources(decade) for decade in (EARLY, LATE)}
    counts = {
        decade: Counter(venue_of[decade].get(work_id) for work_id in lines[decade][1])
        for decade in (EARLY, LATE)
    }
    shared = {
        source
        for source in set(counts[EARLY]) & set(counts[LATE])
        if source and counts[EARLY][source] >= SHARED_MIN and counts[LATE][source] >= SHARED_MIN
    }
    stable: dict[int, list[str]] = {}
    for decade in (EARLY, LATE):
        contents = remove_shared(
            dict(read_cell(path) for path in sorted(TEXTS.glob(f"*_{decade}.txt")))
        )
        kept = set(contents["hci"])
        stable[decade] = [
            line
            for line, work_id in zip(*lines[decade], strict=True)
            if venue_of[decade].get(work_id) in shared and line in kept
        ]
    print(f"HCI_v: {len(shared)} venues; {len(stable[EARLY])} / {len(stable[LATE])} abstracts")
    return stable


def check_round_n() -> None:
    """Stop unless the refitted 1990s classifier reproduces round N's assignment shares."""
    keys = ["seed", "train_decade", "test_decade", "text_field", "assigned_field"]
    refitted = assignment_runs(TEXTS, [EARLY], [LATE], SEEDS).sort_values(keys)
    recorded = pd.read_parquet(OUTPUT / "direction_assignments.parquet")
    recorded = recorded[
        (recorded.train_decade == EARLY) & recorded.test_decade.isin([EARLY, LATE])
    ].sort_values(keys)
    same_keys = refitted[keys].reset_index(drop=True).equals(recorded[keys].reset_index(drop=True))
    if not same_keys or not np.allclose(refitted.share, recorded.share, rtol=0, atol=1e-12):
        raise SystemExit("the refitted classifier does not reproduce round N's shares")
    print("Round N's 1990s classifier reproduced: every assignment share matches.")


def movements(
    table: pd.DataFrame, key: str, early: object, late: object, towards: tuple[str, ...]
) -> list[float]:
    """HCI's relative movement towards ``towards`` over the other group, one value per seed;
    ``key`` names the column that tells the early rows from the late ones."""
    over = BEHAVIOURAL_SOCIAL if towards == COMPUTING else COMPUTING
    return [
        relative_movement(
            cost_matrix(table[(table.seed == seed) & (table[key] == early)]),
            cost_matrix(table[(table.seed == seed) & (table[key] == late)]),
            "hci",
            towards,
            over,
        )
        for seed in SEEDS
    ]


def describe(values: list[float]) -> str:
    return (
        f"mean {statistics.mean(values):+.4f}, sd {statistics.stdev(values):.4f}, "
        f"seeds {', '.join(f'{value:+.4f}' for value in values)}"
        f" ({'present' if present(values) else 'absent'})"
    )


def planted_check(name: str, table: pd.DataFrame) -> bool:
    """Whether an instrument finds both planted moves and none in the unplanted halves."""

    def found(cell: str, towards: tuple[str, ...]) -> list[float]:
        return movements(table[table.cell == cell], "half", "early", "late", towards)

    results = {
        "computer science planted, towards computing": found("compsci", COMPUTING),
        "psychology planted, towards behavioural/social": found("psychology", BEHAVIOURAL_SOCIAL),
        "unplanted, towards computing": found("none", COMPUTING),
        "unplanted, towards behavioural/social": found("none", BEHAVIOURAL_SOCIAL),
    }
    flags = [present(values) for values in results.values()]
    passed = flags[0] and flags[1] and not flags[2] and not flags[3]
    print(f"\nPlanted check, {name}: {'PASSED' if passed else 'FAILED'}")
    for label, values in results.items():
        print(f"  {label}: {describe(values)}")
    return passed


def w1(costs: dict[str, pd.DataFrame]) -> None:
    failures: list[str] = []
    for name, table in costs.items():
        whole = table[(table.cell == "whole") & (table.test_decade == EARLY)]
        for seed in SEEDS:
            matrix = cost_matrix(whole[whole.seed == seed])
            failures.extend(
                f"{name}, seed {seed}: {field} (cheapest {min(row, key=row.__getitem__)})"
                for field, row in matrix.items()
                if min(row, key=row.__getitem__) != field
            )
    detail = "every field's own model cheapest" if not failures else "; ".join(failures)
    report("W1 both instruments see the fields", not failures, detail)


def direction(
    name: str,
    costs: pd.DataFrame,
    passed: bool,
    cell: str,
    towards: tuple[str, ...],
) -> None:
    values = movements(costs[costs.cell == cell], "test_decade", EARLY, LATE, towards)
    detail = describe(values)
    if not passed:
        report(name, None, f"instrument failed the planted check; {detail}")
        return
    report(name, present(values), detail)


def main() -> None:
    stable = venue_stable()
    check_round_n()
    costs: dict[str, pd.DataFrame] = {}
    gate: dict[str, bool] = {}
    for name, fit in INSTRUMENTS.items():
        print(f"\n{name}:", flush=True)
        planted = planted_runs(TEXTS, EARLY, SEEDS, fit, ["compsci", "psychology"], log=print)
        costs[name] = frame_runs(TEXTS, EARLY, LATE, SEEDS, fit=fit, stable=stable, log=print)
        slug = name.replace(" ", "_")
        planted.to_parquet(OUTPUT / f"reference_frame_planted_{slug}.parquet", index=False)
        costs[name].to_parquet(OUTPUT / f"reference_frame_costs_{slug}.parquet", index=False)
        gate[name] = planted_check(name, planted)

    w1(costs)
    classifier, language_models = costs["classifier"], costs["language models"]
    direction(
        "W2 the classifier's contrast agrees with round N",
        classifier,
        gate["classifier"],
        "whole",
        COMPUTING,
    )
    direction(
        "W3 the language models see the opposite",
        language_models,
        gate["language models"],
        "whole",
        BEHAVIOURAL_SOCIAL,
    )
    direction(
        "W4 inside its venues, the classifier still finds the move towards computing",
        classifier,
        gate["classifier"],
        "stable",
        COMPUTING,
    )
    direction(
        "W5 inside its venues, the language models find the behavioural/social move",
        language_models,
        gate["language models"],
        "stable",
        BEHAVIOURAL_SOCIAL,
    )


if __name__ == "__main__":
    main()
