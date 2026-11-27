"""The junk gate's rules on cells (`docs/junk_gate.md`): the classifier screens, the judges confirm.

The screen asks, for each word and decade with enough drawn usages, whether more of them come from
records the classifier calls junk than the decade's share would give by chance: a one-sided
binomial tail at the decade's rate. A flagged cell goes to the judges' confirmation sample.
"""

from __future__ import annotations

import polars as pl
from scipy.stats import binom

CELLS_SCHEMA = pl.Schema(
    {
        "word": pl.String(),
        "decade": pl.Int64(),
        "drawn": pl.Int64(),
        "predicted_junk": pl.Int64(),
        "decade_rate": pl.Float64(),
        "tail_probability": pl.Float64(),
        "flagged": pl.Boolean(),
        "gated": pl.Boolean(),
    }
)


def screen_cells(drawn: pl.DataFrame, *, min_drawn_usages: int, alpha: float) -> pl.DataFrame:
    """Each cell's drawn usages, those from records called junk, and whether the screen flags it.

    ``drawn`` has one row per drawn usage: ``word``, ``decade`` and ``predicted_junk`` (whether
    the classifier calls its record junk). A cell is a word and a decade over all fields. q, the
    decade's rate, is the share of the decade's drawn usages (all words) from records called junk.
    A cell with at least ``min_drawn_usages`` drawn usages is gated: its tail probability is the
    chance of x or more such usages in a binomial draw of its size at rate q, and it is flagged
    when that is under ``alpha``. A cell under the size is never flagged and has no tail
    probability. One row per cell (``CELLS_SCHEMA``), sorted by word and decade.
    """
    usages = drawn.select("word", pl.col("decade").cast(pl.Int64), "predicted_junk")
    rates = usages.group_by("decade").agg(
        pl.col("predicted_junk").mean().cast(pl.Float64).alias("decade_rate")
    )
    cells = (
        usages.group_by("word", "decade")
        .agg(
            pl.len().cast(pl.Int64).alias("drawn"),
            pl.col("predicted_junk").sum().cast(pl.Int64),
        )
        .join(rates, on="decade")
        .sort("word", "decade")
        .with_columns((pl.col("drawn") >= min_drawn_usages).alias("gated"))
    )
    tails = binom.sf(
        cells["predicted_junk"].to_numpy() - 1,
        cells["drawn"].to_numpy(),
        cells["decade_rate"].to_numpy(),
    )
    screened = cells.with_columns(
        pl.when("gated").then(pl.Series(tails, dtype=pl.Float64())).alias("tail_probability")
    ).with_columns((pl.col("gated") & (pl.col("tail_probability") < alpha)).alias("flagged"))
    return screened.select(list(CELLS_SCHEMA)).cast(CELLS_SCHEMA)
