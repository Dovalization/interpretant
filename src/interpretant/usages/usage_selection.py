"""Which usages are in scope, and the one usage drawn per abstract and word (agenda 4.1, 4.3).

A usage is in scope when its part of speech is one its word table gives and it does not sit inside a
name (*medium* is a noun, not the adjective of *medium effect size*; *network* in *Cable News
Network* is a name). The usages table keeps every match; measures read in-scope usages by default.

One usage is drawn per abstract and word, among its in-scope usages in sentences no exclusion
keeps out (not removed, not cut off): the one with the smallest draw key, a cryptographic hash of
the seed, the work, the word and the content key (`usages/identifiers.py`). So abstracts stay
independent observations, and a draw never depends on other abstracts, other words or row order.
"""

from __future__ import annotations

from collections.abc import Collection

import polars as pl

from interpretant.catalog.words import Word

# The entity labels that are names (the author, 2026-11-18): every spaCy label that is not a
# number, date, time, percentage, money, quantity or ordinal.
NAME_LABELS = frozenset(
    {
        "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW",
        "LANGUAGE",
    }
)  # fmt: skip


def is_in_scope(pos: str, word: Word, entity: str | None, name_labels: Collection[str]) -> bool:
    """Whether a usage has a part of speech its word allows and sits inside no name."""
    return pos in word.pos and entity not in name_labels


def mark_drawn(usages: pl.DataFrame) -> pl.DataFrame:
    """The usages with ``drawn`` set on the smallest-keyed eligible usage per work and word.

    Eligible means in scope and in a sentence without an exclusion; the table needs the columns
    ``work_id``, ``word``, ``draw_key``, ``in_scope`` and ``sentence_exclusion``.
    """
    eligible = pl.col("in_scope") & pl.col("sentence_exclusion").is_null()
    smallest = pl.col("draw_key").filter(eligible).min().over("work_id", "word")
    return usages.with_columns((eligible & (pl.col("draw_key") == smallest)).alias("drawn"))
