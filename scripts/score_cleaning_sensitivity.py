"""Score round S1 of docs/predictions_cleaning.md: does the identifiable junk move static drift?

Committed before the full run. Reads the tables `interpretant measure-cleaning-sensitivity`
writes and applies the round's decision rule as fixed before any code ran:

- M(V) is a version's mean flag turnover over the comparisons with at least 200 measured words
  in P;
- C moves the measurement if M(C) is above M of every placebo (rank test, p = (1 + k)/(n + 1),
  k the placebos at or above M(C)) and M(C) is at least half the median M of the bootstrap
  samples.

Also reports the secondary measures (rank disagreement, words entering and leaving, the targets'
combined rank change, the targets' share of tokens inside C's removal) and bet S1c (whether the
comparisons that involve the 1950s and 1960s move most under C). Prints Markdown.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

from interpretant.predictions import report

MIN_MEASURED = 200
SIZE_FRACTION = 0.5
EARLY_DECADES = (1950, 1960)


def mean_turnover(movements: pl.DataFrame) -> pl.DataFrame:
    """Each version's M: the mean flag turnover over the eligible comparisons."""
    return (
        movements.filter(pl.col("base_measured") >= MIN_MEASURED)
        .group_by("version", "replicate")
        .agg(
            M=pl.col("flag_turnover").mean(),
            rank_disagreement=pl.col("rank_disagreement").mean(),
            entering=pl.col("entering").sum(),
            leaving=pl.col("leaving").sum(),
            target_rank_change=pl.col("target_rank_change").sum(),
            comparisons=pl.len(),
        )
        .sort("version", "replicate")
    )


def markdown_table(frame: pl.DataFrame) -> str:
    """The frame as a Markdown table, floats to four places."""

    def cell(value: object) -> str:
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |" for row in frame.iter_rows()]
    return "\n".join(lines)


def verdict(summary: pl.DataFrame) -> tuple[bool, float, float, float, float]:
    """Whether C moves the measurement, with the figures the rule reads.

    Returns M(C), the rank test's p, the largest placebo M and the median bootstrap M after the
    verdict.
    """
    treatment = float(summary.filter(pl.col("version") == "treatment")["M"].item())
    placebos = [float(value) for value in summary.filter(pl.col("version") == "placebo")["M"]]
    bootstraps = sorted(
        float(value) for value in summary.filter(pl.col("version") == "bootstrap")["M"]
    )
    at_or_above = sum(1 for value in placebos if value >= treatment)
    p_value = (1 + at_or_above) / (len(placebos) + 1)
    noise = float(np.median(bootstraps))
    moves = at_or_above == 0 and treatment >= SIZE_FRACTION * noise
    return moves, treatment, p_value, max(placebos), noise


def main(out_dir: Path) -> None:
    """Apply the rule to the round's tables and report each bet and the verdict."""
    movements = pl.read_parquet(out_dir / "movements.parquet")
    doses = pl.read_parquet(out_dir / "doses.parquet")
    targets = pl.read_parquet(out_dir / "target_doses.parquet")
    summary = mean_turnover(movements)
    moves, treatment, p_value, top_placebo, noise = verdict(summary)
    treated = movements.filter(pl.col("version") == "treatment")
    eligible = treated.filter(pl.col("base_measured") >= MIN_MEASURED)
    tokens, removed = int(doses["tokens"].sum()), int(doses["removed_tokens"].sum())
    sys.stdout.write(
        f"Comparisons with at least {MIN_MEASURED} measured words in P: {eligible.height} of "
        f"{treated.height}. C removes {removed:,} of P's {tokens:,} tokens "
        f"({removed / tokens:.2%}): {int(doses['dropped_works'].sum()):,} works dropped, "
        f"{int(doses['verbless_works'].sum()):,} of them with no finite verb.\n\n"
    )
    sys.stdout.write(markdown_table(summary) + "\n")
    report(
        "S1a (M(C) above every placebo)",
        p_value <= 1 / (summary.filter(pl.col("version") == "placebo").height + 1),
        f"M(C) = {treatment:.4f}, largest placebo {top_placebo:.4f}, rank test p = {p_value:.2f}",
    )
    report(
        "S1b (M(C) under half the median bootstrap M)",
        treatment < SIZE_FRACTION * noise,
        f"M(C) = {treatment:.4f}, half the median bootstrap M = {SIZE_FRACTION * noise:.4f}",
    )
    early = float(
        eligible.filter(pl.col("decade_a").is_in(EARLY_DECADES))["flag_turnover"].to_numpy().mean()
    )
    rest = float(
        eligible.filter(~pl.col("decade_a").is_in(EARLY_DECADES))["flag_turnover"].to_numpy().mean()
    )
    report(
        "S1c (the comparisons from the 1950s and 1960s move most under C)",
        early > rest,
        f"C's mean flag turnover {early:.4f} there, {rest:.4f} in the others",
    )
    removed_targets, target_tokens = (
        int(targets["removed_tokens"].sum()),
        int(targets["tokens"].sum()),
    )
    largest = float((targets["removed_tokens"] / targets["tokens"]).to_numpy().max())
    sys.stdout.write(
        f"\nTargets' tokens inside C's removal: {removed_targets:,} of {target_tokens:,}; the "
        f"largest share for one target {largest:.2%}.\n"
    )
    report("Round S1", "moves" if moves else "does not move", "C against the decision rule")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/cleaning_sensitivity"))
