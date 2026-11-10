"""Score docs/predictions_cross_perplexity.md (C0-C5) against the cross-perplexity table."""

from __future__ import annotations

import pandas as pd

from interpretant.predictions import report

# Paths moved on 2026-11-10 when the lemmatised grid took the defaults; the evidence is unchanged.
TABLE = pd.read_parquet("data/drift/unlemmatised/cross_perplexity.parquet")


def excess(decade: int, model_field: str, text_field: str) -> float | None:
    rows = TABLE[
        (TABLE.decade == decade)
        & (TABLE.model_field == model_field)
        & (TABLE.text_field == text_field)
    ]
    return float(rows.iloc[0].excess) if len(rows) else None


def decades_with(*fields: str) -> list[int]:
    return sorted(
        decade for decade, group in TABLE.groupby("decade") if set(fields) <= set(group.text_field)
    )


def enough(hits: int, decades: list[int]) -> bool:
    return hits >= min(6, len(decades)) and bool(decades)


def c0() -> None:
    wins, total = 0, 0
    for (_decade, _text), group in TABLE.groupby(["decade", "text_field"]):
        total += 1
        wins += group.loc[group.cross_entropy.idxmin(), "model_field"] == group.text_field.iloc[0]
    share = wins / total if total else 0.0
    report("C0 own model best", share >= 0.95, f"{wins} of {total} cells ({share:.0%})")


def c1() -> None:
    lines, passed = [], True
    for parent, child in (("compsci", "ai"), ("psychology", "cogpsych")):
        decades = decades_with(parent, child)
        hits = [d for d in decades if excess(d, parent, child) < excess(d, child, parent)]
        passed &= enough(len(hits), decades)
        detail = {
            d: (round(excess(d, parent, child), 2), round(excess(d, child, parent), 2))
            for d in decades
        }
        lines.append(
            f"{parent}->{child} < {child}->{parent} in {len(hits)}/{len(decades)}: {detail}"
        )
    report("C1 parents predict subfields", passed, "\n  ".join(lines))


def c2() -> None:
    lines, passed = [], True
    for parent, child in (("compsci", "ai"), ("psychology", "cogpsych")):
        decades = decades_with(parent, child)
        hits = []
        for decade in decades:
            others = TABLE[
                (TABLE.decade == decade)
                & (TABLE.text_field == child)
                & (TABLE.model_field != child)
            ]
            if others.loc[others.excess.idxmin(), "model_field"] == parent:
                hits.append(decade)
        passed &= enough(len(hits), decades)
        lines.append(f"{parent} closest to {child} in {len(hits)}/{len(decades)} decades {hits}")
    report("C2 subfields closest to parent", passed, "\n  ".join(lines))


def change(name: str, model_field: str, text_field: str, early: int, late: int) -> None:
    before, after = excess(early, model_field, text_field), excess(late, model_field, text_field)
    passed = None if before is None or after is None else after < before
    report(name, passed, f"E({model_field} -> {text_field}): {early}s {before}, {late}s {after}")


def c5() -> None:
    fields = set(TABLE[TABLE.decade == 1980].text_field) & set(
        TABLE[TABLE.decade == 2010].text_field
    )
    means = {}
    for decade in (1980, 2010):
        rows = TABLE[
            (TABLE.decade == decade)
            & TABLE.model_field.isin(fields)
            & TABLE.text_field.isin(fields)
            & (TABLE.model_field != TABLE.text_field)
        ]
        means[decade] = float(rows.excess.mean())
    report(
        "C5 fields grow more specialised",
        means[2010] > means[1980],
        f"{len(fields)} fields; mean excess {means}",
    )


if __name__ == "__main__":
    c0()
    c1()
    c2()
    change("C3 philosophy absorbs AI language", "ai", "philosophy", 2010, 2020)
    change("C4 HCI moves towards philosophy", "philosophy", "hci", 1990, 2020)
    c5()
