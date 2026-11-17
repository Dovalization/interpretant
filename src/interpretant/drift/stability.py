"""How much one version of a comparison's drift table moved from another (round S1).

Both tables are a comparison's flagged drift table (``add_shift_floor``): one row per measured
word with its within-bin ``shift_rank`` and its ``candidate`` flag. The primary statistic of
``docs/predictions_cleaning.md`` is the flag turnover; the rank disagreement and the words
entering and leaving the measured set are reported beside it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl
from scipy.stats import spearmanr

# Fewer shared words than this give no rank correlation worth reporting.
_MIN_SHARED_WORDS = 3


@dataclass(frozen=True)
class Movement:
    """How a version's flagged drift table differs from the base's, for one comparison."""

    flag_turnover: float
    rank_disagreement: float
    entering: int
    leaving: int


def movement(base: pl.DataFrame, version: pl.DataFrame) -> Movement:
    """The flag turnover, the rank disagreement and the words entering and leaving."""
    base_words, version_words = set(base["word"]), set(version["word"])
    return Movement(
        flag_turnover=flag_turnover(base, version),
        rank_disagreement=rank_disagreement(base, version),
        entering=len(version_words - base_words),
        leaving=len(base_words - version_words),
    )


def flag_turnover(base: pl.DataFrame, version: pl.DataFrame) -> float:
    """1 minus the Jaccard index of the two flag sets; 0 when neither flags a word.

    The flags are taken over the words measured in either table, so a flagged word that is no
    longer measured counts as turnover.
    """
    base_flags = set(base.filter(pl.col("candidate"))["word"])
    version_flags = set(version.filter(pl.col("candidate"))["word"])
    union = base_flags | version_flags
    if not union:
        return 0.0
    return 1.0 - len(base_flags & version_flags) / len(union)


def rank_disagreement(base: pl.DataFrame, version: pl.DataFrame) -> float:
    """1 minus Spearman's rho of the within-bin ranks over the words measured in both.

    NaN when fewer than three words are shared.
    """
    joined = base.select("word", "shift_rank").join(
        version.select("word", "shift_rank"), on="word", suffix="_version"
    )
    if joined.height < _MIN_SHARED_WORDS:
        return math.nan
    rho = spearmanr(joined["shift_rank"].to_numpy(), joined["shift_rank_version"].to_numpy())[0]
    return 1.0 - float(rho)
