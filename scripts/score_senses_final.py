"""Score round C″ (C″1-C″2): the repaired sense-set rule on *prompt*, the last calibration.

Committed before the run. Reads the table `interpretant drift senses` writes with round C′'s
options for *prompt*, forms the meaning set (every sense whose share of marked uses, over all
its occurrences, is at least half the highest), scores the predictions in
docs/predictions_senses.md and prints the meaning set's context words and nearest distinct
examples (seed 1).
"""

from __future__ import annotations

import statistics

import pandas as pd

from interpretant.drift.direction import present
from interpretant.drift.senses import context_words
from interpretant.predictions import report

SEEDS = (1, 2, 3, 4, 5)
MIN_OCCURRENCES = 40
MIN_MARKED = 20
JOIN = 0.5
CELLS = (("ai", 2000), ("ai", 2020), ("psychology", 2000), ("psychology", 2020))
MARKS = (
    "language model",
    "large language",
    "llm",
    "chatgpt",
    "gpt",
    "generative ai",
    "chatbot",
    "few-shot",
    "zero-shot",
    "in-context",
    "prompt engineering",
    "prompt tuning",
    "prompt learning",
)


def marked(context: str) -> bool:
    lowered = context.lower()
    return any(mark in lowered for mark in MARKS)


def cell(rows: pd.DataFrame, field: str, decade: int) -> pd.DataFrame:
    return rows[(rows.field == field) & (rows.decade == decade)]


def meaning_set(rows: pd.DataFrame) -> tuple[set[int], pd.Series]:
    """Every sense whose marked share, over all its occurrences, is at least ``JOIN`` times the
    highest, and every sense's marked share."""
    shares = rows.groupby("sense").context.agg(lambda contexts: contexts.map(marked).mean())
    return {int(sense) for sense in shares.index[shares >= JOIN * shares.max()]}, shares


def main() -> None:
    table = pd.read_parquet("data/drift/senses_finer_prompt.parquet")
    sizes = table.groupby(["seed", "field", "decade"]).size()
    small = sorted(
        {
            f"{f} {d}s"
            for seed in SEEDS
            for f, d in CELLS
            if sizes.get((seed, f, d), 0) < MIN_OCCURRENCES
        }
    )
    results = {}
    for seed in SEEDS:
        rows = table[table.seed == seed]
        senses, marked_shares = meaning_set(rows)
        count = int(cell(rows, "ai", 2020).context.map(marked).sum())
        values = {c: float(cell(rows, *c).sense.isin(senses).mean()) for c in CELLS}
        results[seed] = (senses, marked_shares, count, values)
    untestable = bool(small) or any(count < MIN_MARKED for _, _, count, _ in results.values())

    passed = all(
        values[("ai", 2020)] >= 0.50
        and values[("ai", 2000)] <= 0.05
        and values[("psychology", 2000)] < 0.05
        for _, _, _, values in results.values()
    )
    detail = [
        f"seed {seed}: {count} marked in AI's 2020s; meaning set {sorted(senses)} (marked shares "
        + " ".join(f"{s}:{v:.2f}" for s, v in shares.sort_values(ascending=False).items())
        + "); shares "
        + ", ".join(f"{f} {d}s {v:.3f}" for (f, d), v in values.items())
        for seed, (senses, shares, count, values) in results.items()
    ]
    if small:
        detail.append(f"cells under {MIN_OCCURRENCES}: {small}")
    report(
        "C″1 the repaired instrument finds a third known sense",
        None if untestable else passed,
        "\n  ".join(detail),
    )

    rise = [
        values[("psychology", 2020)] - values[("psychology", 2000)]
        for _, _, _, values in results.values()
    ]
    report(
        "C″2 the language-model prompt has reached psychology",
        None if untestable else present(rise),
        f"rise: mean {statistics.mean(rise):+.3f}, sd {statistics.stdev(rise):.3f}, seeds "
        + ", ".join(f"{value:+.3f}" for value in rise)
        + f" ({'present' if present(rise) else 'absent'})",
    )

    rows = table[table.seed == 1]
    senses = results[1][0]
    labelled = rows.assign(sense=rows.sense.isin(senses).astype(int))
    print(f"\n  Meaning set, seed 1, senses {sorted(senses)}: {context_words(labelled, 1)}")
    nearest = rows[rows.sense.isin(senses)].drop_duplicates("context").nsmallest(5, "distance")
    for context in nearest.context:
        print(f"    - {context}")


if __name__ == "__main__":
    main()
