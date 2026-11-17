"""Score round L1 of docs/predictions_lemmas.md: do the lemma exceptions move averaged drift?

Committed before the run. Reads the tables `interpretant measure-lemma-sensitivity` writes. Over
the comparisons with at least 200 measured words in averaged A, D(B) is the mean rank
disagreement (1 - Spearman's rho of the within-bin ranks, over the words measured in both)
between averaged A and averaged B, and D(A') the same between averaged A and a second,
independent average of A. The exceptions move the averaged measurement if D(B) >= 1.5 x D(A').
Prints totals only: no target is named, and only the combined target rank change is shown.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import polars as pl

from interpretant.predictions import report

MIN_MEASURED = 200
NOISE_MULTIPLE = 1.5
TREATMENT = "averaged_treatment"
NOISE = "averaged_noise"


def eligible(movements: pl.DataFrame, version: str) -> pl.DataFrame:
    """One version's comparisons with at least MIN_MEASURED words in averaged A."""
    return movements.filter(
        (pl.col("version") == version) & (pl.col("base_measured") >= MIN_MEASURED)
    )


def mean_of(movements: pl.DataFrame, version: str, column: str) -> float:
    """A column's mean over one version's eligible comparisons.

    Raises:
        SystemExit: if a value is missing (a comparison with fewer than three shared words).
    """
    values = eligible(movements, version)[column].to_numpy()
    if any(math.isnan(value) for value in values):
        raise SystemExit(f"{version} has a comparison without a {column}; the run is void")
    return float(values.mean())


def main(out_dir: Path) -> None:
    """Apply round L1's rule and report its bets and verdict."""
    movements = pl.read_parquet(out_dir / "movements.parquet")
    doses = pl.read_parquet(out_dir / "doses.parquet")
    changed = int(doses["changed_tokens"].sum())
    tokens = int(doses["tokens"].sum())
    if changed == 0:
        raise SystemExit("the two views do not differ; the run is void")
    treated, noise = eligible(movements, TREATMENT), eligible(movements, NOISE)
    lemma = mean_of(movements, TREATMENT, "rank_disagreement")
    noisy = mean_of(movements, NOISE, "rank_disagreement")
    flags = mean_of(movements, TREATMENT, "flag_turnover")
    noisy_flags = mean_of(movements, NOISE, "flag_turnover")
    targets_b = float(treated["target_rank_change"].to_numpy().sum())
    targets_noise = float(noise["target_rank_change"].to_numpy().sum())
    total = movements.filter(pl.col("version") == TREATMENT).height
    sys.stdout.write(
        f"Comparisons with at least {MIN_MEASURED} measured words in averaged A: "
        f"{treated.height} of {total}. Tokens the exceptions change: {changed:,} of {tokens:,} "
        f"({changed / tokens:.2%}).\n"
        f"D(B) = {lemma:.4f}, D(A') = {noisy:.4f}, ratio {lemma / noisy:.2f}; flag turnover "
        f"{flags:.4f} against {noisy_flags:.4f}; words entering {int(treated['entering'].sum()):,} "
        f"and leaving {int(treated['leaving'].sum()):,} against "
        f"{int(noise['entering'].sum()):,} and {int(noise['leaving'].sum()):,}; targets' combined "
        f"rank change {targets_b:,.0f} against {targets_noise:,.0f} "
        f"({int(treated['targets_compared'].sum())} and {int(noise['targets_compared'].sum())} "
        f"target comparisons).\n"
    )
    report(
        "L1-a (D(B) under 1.5 x D(A'))",
        lemma < NOISE_MULTIPLE * noisy,
        f"D(B) = {lemma:.4f}, 1.5 x D(A') = {NOISE_MULTIPLE * noisy:.4f}",
    )
    report("L1-b (D(B) / D(A') over 1.0)", lemma > noisy, f"ratio {lemma / noisy:.2f}")
    report(
        "L1-c (B's combined target rank change over A''s)",
        targets_b > targets_noise,
        f"{targets_b:,.0f} against {targets_noise:,.0f}",
    )
    report(
        "Round L1",
        "moves" if lemma >= NOISE_MULTIPLE * noisy else "does not move",
        "averaged B against the pre-registered rule",
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/lemma_sensitivity"))
