"""Score the corrected cross-perplexity predictions (D0-D5) in docs/predictions_cross_perplexity.md.

Reads data/drift/cross_perplexity_corrected.parquet (shared abstracts removed, 200,000 training
tokens everywhere, seeds 1 to 5).
"""

from __future__ import annotations

import pandas as pd

from interpretant.predictions import report

# Paths moved on 2026-11-10 when the lemmatised grid took the defaults; the evidence is unchanged.
TABLE = pd.read_parquet("data/drift/unlemmatised/cross_perplexity_corrected.parquet")
SEEDS = sorted(TABLE.seed.unique())
MEAN = TABLE.groupby(["decade", "model_field", "text_field"], as_index=False).excess.mean()


def excess(table: pd.DataFrame, decade: int, model_field: str, text_field: str) -> float | None:
    rows = table[
        (table.decade == decade)
        & (table.model_field == model_field)
        & (table.text_field == text_field)
    ]
    return float(rows.iloc[0].excess) if len(rows) else None


def decades_with(*fields: str) -> list[int]:
    return sorted(d for d, g in MEAN.groupby("decade") if set(fields) <= set(g.text_field))


def enough(hits: int, decades: list[int]) -> bool:
    return bool(decades) and hits >= min(6, len(decades))


def d0() -> None:
    shares = {}
    for seed in SEEDS:
        rows = TABLE[TABLE.seed == seed]
        groups = rows.groupby(["decade", "text_field"])
        wins = sum(g.loc[g.cross_entropy.idxmin(), "model_field"] == t for (_, t), g in groups)
        shares[seed] = wins / groups.ngroups
    report("D0 own model best", all(s >= 0.95 for s in shares.values()), str(shares))


def d1() -> None:
    lines, passed = [], True
    for parent, child in (("compsci", "ai"), ("psychology", "cogpsych")):
        decades = decades_with(parent, child)
        hits = [
            d for d in decades if excess(MEAN, d, parent, child) < excess(MEAN, d, child, parent)
        ]
        passed &= enough(len(hits), decades)
        lines.append(f"{parent}->{child} lower in {len(hits)}/{len(decades)} decades {decades}")
    report("D1 parents predict subfields", passed, "\n  ".join(lines))


def d2() -> None:
    lines, passed = [], True
    for parent, child in (("compsci", "ai"), ("psychology", "cogpsych")):
        decades = decades_with(parent, child)
        hits = []
        for decade in decades:
            others = MEAN[
                (MEAN.decade == decade) & (MEAN.text_field == child) & (MEAN.model_field != child)
            ]
            if others.loc[others.excess.idxmin(), "model_field"] == parent:
                hits.append(decade)
        passed &= enough(len(hits), decades)
        lines.append(f"{parent} closest to {child} in {len(hits)}/{len(decades)} decades")
    report("D2 subfields closest to parent", passed, "\n  ".join(lines))


def all_seeds(name: str, model_field: str, text_field: str, early: int, late: int) -> None:
    changes = {}
    for seed in SEEDS:
        rows = TABLE[TABLE.seed == seed]
        before, after = (
            excess(rows, early, model_field, text_field),
            excess(rows, late, model_field, text_field),
        )
        changes[seed] = None if before is None or after is None else round(after - before, 3)
    passed = None if None in changes.values() else all(c < 0 for c in changes.values())
    report(
        name,
        passed,
        f"E({model_field} -> {text_field}) {late}s minus {early}s, per seed: {changes}",
    )


def d5() -> None:
    changes = {}
    for seed in SEEDS:
        rows = TABLE[TABLE.seed == seed]
        fields = set(rows[rows.decade == 1980].text_field) & set(
            rows[rows.decade == 2010].text_field
        )
        means = []
        for decade in (1980, 2010):
            pairs = rows[
                (rows.decade == decade)
                & rows.model_field.isin(fields)
                & rows.text_field.isin(fields)
                & (rows.model_field != rows.text_field)
            ]
            means.append(float(pairs.excess.mean()))
        changes[seed] = (len(fields), round(means[1] - means[0], 3))
    passed = all(change > 0 for _, change in changes.values())
    report(
        "D5 fields grow more specialised",
        passed,
        f"(fields, 2010s minus 1980s) per seed: {changes}",
    )


if __name__ == "__main__":
    d0()
    d1()
    d2()
    all_seeds("D3 philosophy absorbs AI language", "ai", "philosophy", 2010, 2020)
    all_seeds("D4 HCI moves towards philosophy", "philosophy", "hci", 1990, 2020)
    d5()
