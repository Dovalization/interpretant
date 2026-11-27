"""The junk gate's rules (`docs/junk_gate.md`): the decade rule, the cell screen, the exclusions.

- **The decade rule** reads L alone: a decade passes with at most 52 junk usages among its labelled
  usages, and when Fisher's exact test does not find more junk in it than in the other decades
  together (one-sided, at 0.05/8).
- **The screen** asks, for each word and decade with enough drawn usages, whether more of them come
  from records the classifier calls junk than the decade's share would give by chance: a one-sided
  binomial tail at the decade's rate. A flagged cell goes to the judges' confirmation sample, and
  fails at 9 or more junk usages of its 40.
- **The exclusions** are the drawn usages from records the classifier calls junk, of the words of
  the failed cells in all their decades, or of every word when a decade fails.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

import polars as pl
from scipy.stats import binom, fisher_exact

# A decade passes with at most this many junk usages among its 500 labelled usages
DECADE_JUNK_BAR = 52
# Fisher's exact test of each decade against the other seven, Bonferroni-corrected over the eight
DECADE_COMPARISON_ALPHA = 0.05 / 8
# A flagged cell fails at this many junk usages among its confirmation sample, or more
CELL_FAILING_JUNK = 9

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


@dataclass(frozen=True)
class DecadeVerdict:
    """One decade under the decade rule: its junk usages of its labelled ones, and Fisher's p.

    ``fisher_p`` is the one-sided p-value of more junk in the decade than in the other decades
    together.
    """

    decade: int
    junk: int
    labelled: int
    fisher_p: float

    @property
    def passes_bar(self) -> bool:
        """Whether the decade has at most 52 junk usages."""
        return self.junk <= DECADE_JUNK_BAR

    @property
    def passes_comparison(self) -> bool:
        """Whether the comparison with the other decades stays at or above 0.05/8."""
        return self.fisher_p >= DECADE_COMPARISON_ALPHA

    @property
    def passes(self) -> bool:
        """Whether the decade passes both the bar and the comparison."""
        return self.passes_bar and self.passes_comparison


def is_junk_usage(verdict: str, sentence: str) -> bool:
    """Whether a label makes its usage junk: its record is junk or its sentence another text's."""
    return verdict == "junk" or sentence == "other"


def compute_decade_verdicts(counts: Mapping[int, tuple[int, int]]) -> list[DecadeVerdict]:
    """Each decade's verdict from its junk and labelled usages (``counts``), sorted by decade.

    Each decade's count is tested against the other decades' pooled counts with Fisher's exact
    test, one-sided (scipy's ``fisher_exact``, ``alternative="greater"``).
    """
    total_junk = sum(junk for junk, _ in counts.values())
    total = sum(labelled for _, labelled in counts.values())
    verdicts = []
    for decade in sorted(counts):
        junk, labelled = counts[decade]
        other_junk, others = total_junk - junk, total - labelled
        table = [[junk, labelled - junk], [other_junk, others - other_junk]]
        fisher_p = float(fisher_exact(table, alternative="greater").pvalue)
        verdicts.append(DecadeVerdict(decade, junk, labelled, fisher_p))
    return verdicts


def build_exclusions(drawn: pl.DataFrame, words: Collection[str]) -> pl.DataFrame:
    """The drawn usages of ``words`` whose records the classifier calls junk, in every decade.

    ``drawn`` has one row per drawn usage: ``work_id``, ``word`` and ``predicted_junk``. A row per
    excluded usage, a work and a word, sorted.
    """
    excluded = pl.DataFrame({"word": sorted(words)}, schema={"word": pl.String()})
    return (
        drawn.filter("predicted_junk")
        .join(excluded, on="word", how="semi")
        .select("work_id", "word")
        .unique()
        .sort("work_id", "word")
    )
