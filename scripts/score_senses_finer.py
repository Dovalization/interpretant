"""Score round C′ (C′1-C′4): the finer sense instrument on *attention*, where it was chosen,
and on *hallucination*, a known answer it has not seen.

Committed before the runs. Reads the two tables `interpretant drift senses` writes with
round C′'s options (SciBERT, eight senses, sampling with replacement, plurals, quality rules)
and scores every prediction in docs/predictions_senses.md, then prints each sense set with its
top context words and nearest examples (seed 1).
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
COVER = 0.8
MECHANISM = (
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
LANGUAGE_MODEL = (
    "language model",
    "llm",
    "chatgpt",
    "gpt",
    "generative ai",
    "chatbot",
    "factual",
    "fabricat",
)


def marked(context: str, marks: tuple[str, ...]) -> bool:
    lowered = context.lower()
    return any(mark in lowered for mark in marks)


def cell(rows: pd.DataFrame, field: str, decade: int) -> pd.DataFrame:
    return rows[(rows.field == field) & (rows.decade == decade)]


def sense_set(rows: pd.DataFrame, marks: tuple[str, ...]) -> tuple[set[int], int]:
    """The fewest senses holding ``COVER`` of the marked uses among AI's 2020s occurrences,
    taken greedily by marked count, and the number of marked uses."""
    ai = cell(rows, "ai", 2020)
    counts = ai[ai.context.map(lambda context: marked(context, marks))].sense.value_counts()
    chosen: set[int] = set()
    covered = 0
    for sense, count in zip(counts.index, counts, strict=True):
        if covered >= COVER * counts.sum():
            break
        chosen.add(int(sense))
        covered += int(count)
    return chosen, int(counts.sum())


def share(rows: pd.DataFrame, field: str, decade: int, senses: set[int]) -> float:
    return float(cell(rows, field, decade).sense.isin(senses).mean())


def small_cells(table: pd.DataFrame, cells: list[tuple[str, int]]) -> list[str]:
    sizes = table.groupby(["seed", "field", "decade"]).size()
    return sorted(
        {
            f"{field} {decade}s"
            for seed in SEEDS
            for field, decade in cells
            if sizes.get((seed, field, decade), 0) < MIN_OCCURRENCES
        }
    )


def show(name: str, rows: pd.DataFrame, senses: set[int]) -> None:
    labelled = rows.assign(sense=rows.sense.isin(senses).astype(int))
    print(f"\n  {name}, seed 1, senses {sorted(senses)}: {context_words(labelled, 1)}")
    for context in rows[rows.sense.isin(senses)].nsmallest(5, "distance").context:
        print(f"    - {context}")


def c_prime_1(table: pd.DataFrame) -> None:
    cells = [("ai", 1990), ("ai", 2020), ("psychology", 1990), ("psychology", 2020)]
    small = small_cells(table, cells)
    results = {}
    for seed in SEEDS:
        rows = table[table.seed == seed]
        senses, count = sense_set(rows, MECHANISM)
        results[seed] = (senses, count, [share(rows, f, d, senses) for f, d in cells])
    passed = all(
        values[0] <= 0.05 and values[1] >= 0.30 and values[2] < 0.05 and values[3] < 0.05
        for _, _, values in results.values()
    )
    untestable = small or any(count < MIN_MARKED for _, count, _ in results.values())
    detail = [
        f"seed {seed}: {count} marked, mechanism set {sorted(senses)}; shares AI 1990s "
        f"{values[0]:.3f}, AI 2020s {values[1]:.3f}, psychology 1990s {values[2]:.3f}, "
        f"2020s {values[3]:.3f}"
        for seed, (senses, count, values) in results.items()
    ]
    if small:
        detail.append(f"cells under {MIN_OCCURRENCES}: {small}")
    report(
        "C′1 the fix works where it was chosen", None if untestable else passed, "\n  ".join(detail)
    )
    show("Mechanism set", table[table.seed == 1], results[1][0])


def c_prime_2_to_4(table: pd.DataFrame) -> None:
    cells = [
        ("ai", 2020),
        ("psychology", 2000),
        ("psychology", 2020),
        ("neuroscience", 2000),
        ("neuroscience", 2020),
    ]
    small = small_cells(table, cells)
    results = {}
    for seed in SEEDS:
        rows = table[table.seed == seed]
        senses, count = sense_set(rows, LANGUAGE_MODEL)
        results[seed] = (senses, count, {c: share(rows, *c, senses) for c in cells})
    untestable = small or any(count < MIN_MARKED for _, count, _ in results.values())

    def shares_of(field: str, decade: int) -> list[float]:
        return [values[(field, decade)] for _, _, values in results.values()]

    passed = all(
        values[("ai", 2020)] >= 0.50
        and values[("psychology", 2000)] < 0.05
        and values[("neuroscience", 2000)] < 0.05
        for _, _, values in results.values()
    )
    detail = [
        f"seed {seed}: {count} marked, language-model set {sorted(senses)}; shares "
        + ", ".join(f"{f} {d}s {v:.3f}" for (f, d), v in values.items())
        for seed, (senses, count, values) in results.items()
    ]
    if small:
        detail.append(f"cells under {MIN_OCCURRENCES}: {small}")
    report(
        "C′2 the finer instrument finds a sense it has not seen",
        None if untestable else passed,
        "\n  ".join(detail),
    )

    late = {
        field: statistics.mean(shares_of(field, 2020)) for field in ("psychology", "neuroscience")
    }
    report(
        "C′3 the perceptual sense still dominates where it lives",
        None if untestable else all(value < 0.10 for value in late.values()),
        ", ".join(f"{field} 2020s: {value:.3f}" for field, value in late.items()),
    )
    rise = [
        b - a
        for a, b in zip(shares_of("psychology", 2000), shares_of("psychology", 2020), strict=True)
    ]
    report(
        "C′4 but the new sense has reached psychology",
        None if untestable else present(rise),
        f"rise in psychology: mean {statistics.mean(rise):+.3f}, sd {statistics.stdev(rise):.3f}, "
        f"seeds {', '.join(f'{value:+.3f}' for value in rise)}"
        f" ({'present' if present(rise) else 'absent'})",
    )
    show("Language-model set", table[table.seed == 1], results[1][0])


def main() -> None:
    c_prime_1(pd.read_parquet("data/drift/senses_finer_attention.parquet"))
    c_prime_2_to_4(pd.read_parquet("data/drift/senses_finer_hallucination.parquet"))


if __name__ == "__main__":
    main()
