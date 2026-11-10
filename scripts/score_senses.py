"""Score round C (C1-C5): does the sense instrument find senses whose history is known?

Committed before the runs. Reads the tables `interpretant drift senses` writes, one per design
and encoder, and scores every prediction in docs/predictions_senses.md for each encoder, then
prints each encoder's mechanism sense and AI sense of *intelligence* with their top context
words and nearest examples (seed 1).
"""

from __future__ import annotations

import statistics

import pandas as pd

from interpretant.drift.direction import present
from interpretant.drift.senses import (
    context_words,
    distribution,
    examples,
    jensen_shannon,
    sense_shares,
)
from interpretant.predictions import report

ENCODERS = {"SciBERT": "scibert", "ModernBERT": "modernbert"}
SEEDS = (1, 2, 3, 4, 5)
MIN_OCCURRENCES = 40
MIN_MARKED = 20
MARKS = (
    "self-attention",
    "self attention",
    "cross-attention",
    "transformer",
    "multi-head",
    "multihead",
    "attention mechanism",
    "attention layer",
    "attention weight",
    "attention head",
    "attention module",
)
SPREAD_FIELDS = ("communication", "histphilsci", "sociology", "philosophy")


def marked(context: str) -> bool:
    lowered = context.lower()
    return any(mark in lowered for mark in MARKS)


def cell(rows: pd.DataFrame, field: str, decade: int) -> pd.DataFrame:
    return rows[(rows.field == field) & (rows.decade == decade)]


def too_small(table: pd.DataFrame) -> list[str]:
    """Word, field and decade of every cell under the minimum, in any seed."""
    sizes = table.groupby(["seed", "word", "field", "decade"]).size()
    small = sizes[sizes < MIN_OCCURRENCES].reset_index()
    return sorted({f"{row.word} {row.field} {row.decade}s" for row in small.itertuples()})


def share(rows: pd.DataFrame, field: str, decade: int, sense: int) -> float:
    found = cell(rows, field, decade)
    return float((found.sense == sense).mean())


def mechanism_sense(rows: pd.DataFrame) -> tuple[int, int]:
    """The sense holding the most marked uses among AI's 2020s occurrences, and their count."""
    ai = cell(rows, "ai", 2020)
    marks = ai[ai.context.map(marked)]
    if marks.empty:
        return -1, 0
    return int(marks.sense.mode()[0]), len(marks)


def change(table: pd.DataFrame, word: str, field: str) -> float:
    """Mean over seeds of the Jensen-Shannon divergence between the field's 1990s and 2020s
    sense distributions."""
    values = []
    for seed in SEEDS:
        shares = sense_shares(table[(table.seed == seed) & (table.word == word)])
        values.append(
            jensen_shannon(distribution(shares, field, 1990), distribution(shares, field, 2020))
        )
    return statistics.mean(values)


def score_attention(name: str, table: pd.DataFrame) -> None:
    small = too_small(table)
    attention = table[table.word == "attention"]
    senses = {seed: mechanism_sense(attention[attention.seed == seed]) for seed in SEEDS}
    controls = sorted(set(table.word) - {"attention"})

    c1: dict[int, tuple[float, float, float]] = {}
    for seed, (sense, count) in senses.items():
        rows = attention[attention.seed == seed]
        ai = cell(rows, "ai", 2020)
        marks = ai[ai.context.map(marked)]
        purity = float((marks.sense == sense).mean()) if count else 0.0
        c1[seed] = (purity, share(rows, "ai", 1990, sense), share(rows, "ai", 2020, sense))
    detail = [
        f"seed {seed}: {senses[seed][1]} marked, {purity:.2f} in the mechanism sense; "
        f"its share of AI's 1990s {early:.3f}, 2020s {late:.3f}"
        for seed, (purity, early, late) in c1.items()
    ]
    c1_passed = all(
        purity >= 0.8 and early <= 0.05 and late >= 0.30 for purity, early, late in c1.values()
    )
    untestable = small or any(count < MIN_MARKED for _, count in senses.values())
    if small:
        detail.append(f"cells under {MIN_OCCURRENCES}: {small}")
    report(
        f"C1 [{name}] the instrument isolates the new sense",
        None if untestable else c1_passed,
        "\n  ".join(detail),
    )

    psychology_shares = [
        (share(rows, "psychology", 1990, sense), share(rows, "psychology", 2020, sense))
        for seed, (sense, _) in senses.items()
        for rows in [attention[attention.seed == seed]]
    ]
    psychology_change = change(table, "attention", "psychology")
    control_changes = {word: change(table, word, "psychology") for word in controls}
    median_control = statistics.median(control_changes.values())
    c2_passed = all(max(pair) < 0.05 for pair in psychology_shares) and (
        psychology_change <= median_control
    )
    report(
        f"C2 [{name}] psychology's attention did not change",
        None if small else c2_passed,
        f"mechanism sense's share of psychology (1990s, 2020s) by seed: "
        f"{', '.join(f'({a:.3f}, {b:.3f})' for a, b in psychology_shares)}\n"
        f"  psychology's change in attention {psychology_change:.4f}; "
        f"median of the controls {median_control:.4f}",
    )

    ai_change = change(table, "attention", "ai")
    ai_controls = {word: change(table, word, "ai") for word in controls}
    below = sum(value < ai_change for value in ai_controls.values())
    ranked = sorted(ai_controls.items(), key=lambda item: -item[1])[:5]
    report(
        f"C3 [{name}] AI's attention changed more than typical words",
        None if small else below >= 27,
        f"AI's change in attention {ai_change:.4f}, larger than {below} of {len(controls)} "
        f"controls; largest controls: {', '.join(f'{w} {v:.4f}' for w, v in ranked)}",
    )

    rows = attention[attention.seed == 1]
    sense = senses[1][0]
    print(f"\n  [{name}] mechanism sense, seed 1: {context_words(rows, sense)}")
    for context in examples(rows, sense, top=5):
        print(f"    - {context}")


def score_intelligence(name: str, table: pd.DataFrame) -> None:
    small = too_small(table)
    rises: dict[str, list[float]] = {field: [] for field in (*SPREAD_FIELDS, "cogpsych")}
    senses: dict[int, int] = {}
    for seed in SEEDS:
        rows = table[table.seed == seed]
        senses[seed] = int(cell(rows, "ai", 2020).sense.mode()[0])
        for field, values in rises.items():
            values.append(
                share(rows, field, 2020, senses[seed]) - share(rows, field, 2000, senses[seed])
            )

    def describe(values: list[float]) -> str:
        return (
            f"mean {statistics.mean(values):+.3f}, sd {statistics.stdev(values):.3f}"
            f" ({'present' if present(values) else 'absent'})"
        )

    detail = "\n  ".join(f"{field}: {describe(rises[field])}" for field in SPREAD_FIELDS)
    report(
        f"C4 [{name}] AI's intelligence spread into other fields",
        None if small else all(present(rises[field]) for field in SPREAD_FIELDS),
        detail + (f"\n  cells under {MIN_OCCURRENCES}: {small}" if small else ""),
    )
    report(
        f"C5 [{name}] not into psychology's intelligence",
        None if small else not present(rises["cogpsych"]),
        f"cognitive psychology: {describe(rises['cogpsych'])}",
    )

    rows = table[table.seed == 1]
    print(f"\n  [{name}] AI sense of intelligence, seed 1: {context_words(rows, senses[1])}")
    for context in examples(rows, senses[1], top=5):
        print(f"    - {context}")


def main() -> None:
    for name, slug in ENCODERS.items():
        score_attention(name, pd.read_parquet(f"data/drift/senses_attention_{slug}.parquet"))
        score_intelligence(name, pd.read_parquet(f"data/drift/senses_intelligence_{slug}.parquet"))


if __name__ == "__main__":
    main()
