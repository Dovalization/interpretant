"""Score round S1b of docs/predictions_cleaning.md: does the junk move bootstrap-averaged drift?

Committed before the run. Reads the tables `interpretant measure-cleaning-sensitivity --averaged`
writes. Over the comparisons with at least 200 measured words in averaged P, D(junk) is the mean
flag turnover between averaged P and averaged C, and D(noise) the mean flag turnover between
averaged P and averaged P', a second, independent average of P. C moves the averaged measurement
if D(junk) >= 1.5 x D(noise), as the amendment fixed before any code ran.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

from interpretant.predictions import report

MIN_MEASURED = 200
NOISE_MULTIPLE = 1.5


def mean_of(movements: pl.DataFrame, version: str, column: str) -> float:
    """A column's mean over the eligible comparisons of one version."""
    values = movements.filter(
        (pl.col("version") == version) & (pl.col("base_measured") >= MIN_MEASURED)
    )[column]
    return float(values.to_numpy().mean())


def main(out_dir: Path) -> None:
    """Apply round S1b's rule and report its bet and verdict."""
    movements = pl.read_parquet(out_dir / "movements.parquet")
    eligible = movements.filter(
        (pl.col("version") == "averaged_treatment") & (pl.col("base_measured") >= MIN_MEASURED)
    )
    junk = mean_of(movements, "averaged_treatment", "flag_turnover")
    noise = mean_of(movements, "averaged_noise", "flag_turnover")
    treated = movements.filter(pl.col("version") == "averaged_treatment")
    noise_rows = movements.filter(
        (pl.col("version") == "averaged_noise") & (pl.col("base_measured") >= MIN_MEASURED)
    )
    junk_ranks = mean_of(movements, "averaged_treatment", "rank_disagreement")
    noise_ranks = mean_of(movements, "averaged_noise", "rank_disagreement")
    junk_targets = float(eligible["target_rank_change"].to_numpy().sum())
    noise_targets = float(noise_rows["target_rank_change"].to_numpy().sum())
    sys.stdout.write(
        f"Comparisons with at least {MIN_MEASURED} measured words in averaged P: "
        f"{eligible.height} of {treated.height}.\n"
        f"D(junk) = {junk:.4f}, D(noise) = {noise:.4f}, ratio {junk / noise:.2f}; rank "
        f"disagreement {junk_ranks:.4f} against {noise_ranks:.4f}; targets' combined rank "
        f"change {junk_targets:,.0f} against {noise_targets:,.0f}.\n"
    )
    report(
        "S1b-a (D(junk) under 1.5 x D(noise))",
        junk < NOISE_MULTIPLE * noise,
        f"D(junk) = {junk:.4f}, 1.5 x D(noise) = {NOISE_MULTIPLE * noise:.4f}",
    )
    report(
        "Round S1b",
        "moves" if junk >= NOISE_MULTIPLE * noise else "does not move",
        "averaged C against the amendment's rule",
    )


if __name__ == "__main__":
    main(
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("data/processed/cleaning_sensitivity_averaged")
    )
