"""Score round 5 (E1-E6): does the grid find changes whose history is known?

Committed before it was run. Seed 42's ranks come from the lemmatised drift tables
(data/drift/within_field.parquet, cross_field.parquet); seed 43's are computed with the same
functions on models/twec_grid_seed43, as round 4 did. A candidate is in the top 5% of its pair
on both cosine drift and neighbourhood shift.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import pandas as pd

from interpretant.drift.grid import add_noise_floor, compare_cells
from interpretant.predictions import load_vectors, report

SEED43 = Path("models/twec_grid_seed43")
DRIFT = Path("data/drift")
LAYER = "openalex"
KNOWN = {
    "E1 attention in AI, 2010s to 2020s": ("ai", 2010, "attention"),
    "E2 platform in communication, 2000s to 2010s": ("communication", 2000, "platform"),
    "E3 network in AI, 1980s to 1990s": ("ai", 1980, "network"),
    "E4 alignment in AI, 2010s to 2020s": ("ai", 2010, "alignment"),
}
QUIET = (("ai", 1990, "attention"), ("psychology", 2010, "attention"))
CROSS_FIELDS = ("psychology", "cogpsych", "neuroscience")
CROSS_FLOOR = 90.0


@cache
def table(name: str) -> pd.DataFrame:
    return pd.read_parquet(DRIFT / f"{name}.parquet")


@cache
def pair43(field_a: str, decade_a: int, field_b: str, decade_b: int) -> pd.DataFrame:
    return add_noise_floor(
        compare_cells(
            load_vectors(SEED43, LAYER, field_a, decade_a),
            load_vectors(SEED43, LAYER, field_b, decade_b),
        )
    )


def within(seed: int, field: str, decade: int, word: str) -> pd.Series | None:
    """A word's row for ``field``'s ``decade`` → ``decade + 10`` transition in one seed."""
    if seed == 42:
        rows = table("within_field")
        rows = rows[
            (rows.layer == LAYER)
            & (rows.field_a == field)
            & (rows.decade_a == decade)
            & (rows.word == word)
        ]
    else:
        pair = pair43(field, decade, field, decade + 10)
        rows = pair[pair.word == word]
    return rows.iloc[0] if len(rows) else None


def cross(seed: int, field_a: str, field_b: str, decade: int, word: str) -> float | None:
    """A word's cross-field cosine rank between two fields in one decade and seed."""
    first, second = sorted([field_a, field_b])
    if seed == 42:
        rows = table("cross_field")
        rows = rows[
            (rows.field_a == first)
            & (rows.field_b == second)
            & (rows.decade_a == decade)
            & (rows.word == word)
        ]
    else:
        pair = pair43(first, decade, second, decade)
        rows = pair[pair.word == word]
    return float(rows.iloc[0]["cosine_rank"]) if len(rows) else None


def describe(row: pd.Series | None) -> str:
    if row is None:
        return "not measured"
    flag = "candidate" if row["candidate"] else "not a candidate"
    return f"cosine rank {row['cosine_rank']:.0f}, neighbour rank {row['shift_rank']:.0f} ({flag})"


def known() -> None:
    for name, (field, decade, word) in KNOWN.items():
        rows = {seed: within(seed, field, decade, word) for seed in (42, 43)}
        if any(row is None for row in rows.values()):
            report(name, None, "; ".join(f"seed {s}: {describe(r)}" for s, r in rows.items()))
            continue
        passed = all(bool(row["candidate"]) for row in rows.values() if row is not None)
        report(name, passed, "; ".join(f"seed {s}: {describe(r)}" for s, r in rows.items()))


def quiet() -> None:
    lines, flagged, missing = [], False, False
    for field, decade, word in QUIET:
        for seed in (42, 43):
            row = within(seed, field, decade, word)
            missing = missing or row is None
            flagged = flagged or (row is not None and bool(row["candidate"]))
            lines.append(
                f"{word} in {field} {decade}s → {decade + 10}s, seed {seed}: {describe(row)}"
            )
    report(
        "E5 quiet where nothing documented changed",
        None if missing else not flagged,
        "\n  ".join(lines),
    )


def cross_field() -> None:
    lines, passed, missing = [], True, False
    for other in CROSS_FIELDS:
        for seed in (42, 43):
            early = cross(seed, "ai", other, 1990, "attention")
            late = cross(seed, "ai", other, 2020, "attention")
            if early is None or late is None:
                missing = True
                lines.append(f"AI × {other}, seed {seed}: not measured")
                continue
            passed = passed and late > early and late >= CROSS_FLOOR
            lines.append(f"AI × {other}, seed {seed}: 1990s {early:.0f}, 2020s {late:.0f}")
    report("E6 across fields, the new sense shows", None if missing else passed, "\n  ".join(lines))


if __name__ == "__main__":
    known()
    quiet()
    cross_field()
