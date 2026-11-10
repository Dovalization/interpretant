"""Score round M (a held-out check on every cell) as committed in
docs/predictions_cross_perplexity.md, with its method amendment, from the table
`interpretant drift heldout` writes.
"""

from __future__ import annotations

import statistics

import pandas as pd

from interpretant.predictions import report

TABLE = pd.read_parquet("data/drift/heldout.parquet")
CELLS = (
    TABLE.groupby(["layer", "field", "decade"])[["gap", "training_tokens", "documents"]]
    .mean()
    .reset_index()
)
REFERENCE = statistics.median(
    CELLS[(CELLS.layer == "openalex") & CELLS.decade.isin([2000, 2010, 2020])].gap
)
CELLS["flagged"] = CELLS.gap > 1.5 * REFERENCE
FLAGGED = CELLS[CELLS.flagged]


def is_flagged(layer: str, field: str, decade: int) -> bool:
    rows = CELLS[(CELLS.layer == layer) & (CELLS.field == field) & (CELLS.decade == decade)]
    if rows.empty:
        raise KeyError(f"no cell {layer}/{field}_{decade}")
    return bool(rows.flagged.iloc[0])


def gap_of(layer: str, field: str, decade: int) -> float:
    rows = CELLS[(CELLS.layer == layer) & (CELLS.field == field) & (CELLS.decade == decade)]
    return float(rows.gap.iloc[0])


def m0() -> None:
    rho = CELLS.training_tokens.corr(CELLS.gap, method="spearman")
    report("M0 size drives the gap", rho <= -0.8, f"Spearman ρ = {rho:.3f} over {len(CELLS)} cells")


def m1() -> None:
    cells = [("hci", 1950), ("hci", 1960), ("hci", 1970), ("ai", 1950)]
    found = {cell: is_flagged("openalex", *cell) for cell in cells}
    detail = ", ".join(
        f"{field} {decade}s gap {gap_of('openalex', field, decade):.2f}"
        f"{' (flagged)' if flagged else ' (not flagged)'}"
        for (field, decade), flagged in found.items()
    )
    report("M1 the check has teeth", all(found.values()), detail)


def m2() -> None:
    cells = (
        [("openalex_full", "philosophy", decade) for decade in (1980, 1990, 2000, 2010, 2020)]
        + [("openalex_full", "hci", decade) for decade in (2000, 2010, 2020)]
        + [("openalex", "ai", decade) for decade in (1980, 2000, 2010, 2020)]
        + [
            ("openalex", field, decade)
            for decade in (1980, 1990, 2010, 2020)
            for field in sorted(CELLS[CELLS.layer == "openalex"].field.unique())
        ]
    )
    flagged = sorted({cell for cell in cells if is_flagged(*cell)})
    names = ", ".join(f"{layer}/{field}_{decade}" for layer, field, decade in flagged)
    detail = f"{len(set(cells))} finding cells, {len(flagged)} flagged" + (
        f": {names}" if flagged else ""
    )
    report("M2 no robust finding rests on a flagged cell", not flagged, detail)


if __name__ == "__main__":
    threshold = 1.5 * REFERENCE
    print(f"Reference median gap (main grid, 2000s–2020s): {REFERENCE:.3f} bits")
    print(f"Flag above {threshold:.3f} bits")
    m0()
    m1()
    m2()
    print(f"\nFlagged cells ({len(FLAGGED)}):")
    for row in FLAGGED.sort_values(["layer", "field", "decade"]).itertuples():
        cell = f"{row.layer}/{row.field}_{row.decade}"
        print(f"  {cell}: gap {row.gap:.2f}, {int(row.documents):,} documents")
