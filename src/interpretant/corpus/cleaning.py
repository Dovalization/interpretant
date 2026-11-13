"""Corpus-level rules (decision agenda 2.3), run on the abstracts table after the spaCy pass.

Template sentences and repeated sentences add labelled removal ranges, as the pass does; an
abstract made mostly of template sentences, and works over an author's share of a cell, are
dropped with their reason. The text is never edited (2.2).
"""

from __future__ import annotations

import hashlib
from collections import Counter

import polars as pl

TEMPLATE_SENTENCE = "template_sentence"
REPEATED_SENTENCE = "repeated_sentence"
TEMPLATE = "template"
AUTHOR_CAP = "author_cap"
# "An abstract made mostly of them is dropped": more than half its sentence characters.
TEMPLATE_DROP_SHARE = 0.5


def mark_template_sentences(abstracts: pl.DataFrame, *, min_abstracts: int) -> pl.DataFrame:
    """Remove every sentence shared by at least ``min_abstracts`` abstracts, digits masked.

    Sentences are spaCy's, compared after replacing each run of digits with ``0``, and counted
    over every source among the abstracts not already dropped. An abstract whose template
    sentences hold more than half of its sentence characters is dropped as ``template``.
    """
    masked = pl.col("sentence").str.replace_all(r"\d+", "0")
    sentences = _sentence_rows(abstracts.filter(pl.col("dropped_reason").is_null())).with_columns(
        (pl.col("work_id").n_unique().over(masked) >= min_abstracts).alias("is_template")
    )
    marked = _add_removals(abstracts, sentences.filter("is_template"), reason=TEMPLATE_SENTENCE)
    template_shares = sentences.group_by("work_id").agg(
        (pl.col("length").filter("is_template").sum() / pl.col("length").sum()).alias("share")
    )
    mostly_templates = template_shares.filter(pl.col("share") > TEMPLATE_DROP_SHARE)["work_id"]
    return _mark_dropped(marked, mostly_templates, reason=TEMPLATE)


def mark_repeated_sentences(abstracts: pl.DataFrame) -> pl.DataFrame:
    """Keep each sentence once per abstract: later copies of the same text are removed.

    Without this rule two mentions of a word in a repeated sentence would share a usage id
    (decision agenda 4.5).
    """
    sentences = _sentence_rows(abstracts)
    repeats = sentences.filter(pl.col("start").rank("ordinal").over("work_id", "sentence") > 1)
    return _add_removals(abstracts, repeats, reason=REPEATED_SENTENCE)


def mark_author_cap(abstracts: pl.DataFrame, *, max_share: float, floor: int) -> pl.DataFrame:
    """Drop works so no author holds more than ``max(floor, max_share × n)`` of a cell's n works.

    A cell is a source, field and decade. Each cell is capped on all its works not already
    dropped, independently of the other cells, and a work over the cap in any cell leaves all of
    them: no cell's result depends on another's, so the order of the cells cannot matter.
    Within a cell, the most frequent authors go first, and an author over the cap keeps the works
    that come first when ordered by a hash of the work id: a stable order, so which works an
    author keeps does not change when other works are added or removed (as for drawn mentions,
    decision agenda 4.3). Dropping a co-authored work also lowers its co-authors' counts in that
    cell.
    """
    kept = abstracts.filter(pl.col("dropped_reason").is_null())
    authors_by_work = {
        work_id: frozenset(author_ids)
        for work_id, author_ids in zip(kept["work_id"], kept["author_ids"], strict=True)
    }
    works_by_cell: dict[tuple[str, str, int], list[str]] = {}
    for row in kept.select("work_id", "source", "fields", "decade").iter_rows(named=True):
        for field in row["fields"]:
            works_by_cell.setdefault((row["source"], field, row["decade"]), []).append(
                row["work_id"]
            )
    dropped: set[str] = set()
    for cell_works in works_by_cell.values():
        dropped |= _cap_cell(cell_works, authors_by_work, max_share=max_share, floor=floor)
    return _mark_dropped(
        abstracts, pl.Series(sorted(dropped), dtype=pl.String()), reason=AUTHOR_CAP
    )


def _cap_cell(
    cell_works: list[str],
    authors_by_work: dict[str, frozenset[str]],
    *,
    max_share: float,
    floor: int,
) -> set[str]:
    """The works one cell drops so that no author holds more than its limit."""
    limit = max(floor, int(max_share * len(cell_works)))
    author_counts = Counter(author for work_id in cell_works for author in authors_by_work[work_id])
    dropped: set[str] = set()
    for author, count in sorted(author_counts.items(), key=lambda item: (-item[1], item[0])):
        if count <= limit:
            break
        author_works = sorted(
            (
                work_id
                for work_id in cell_works
                if work_id not in dropped and author in authors_by_work[work_id]
            ),
            key=_stable_rank,
        )
        dropped.update(author_works[limit:])
    return dropped


def _sentence_rows(abstracts: pl.DataFrame) -> pl.DataFrame:
    """One row per sentence: work id, start, end, length and the sentence's text."""
    return (
        abstracts.select("work_id", "text", "sentences")
        .explode("sentences")
        .unnest("sentences")
        .select(
            "work_id",
            "start",
            "end",
            (pl.col("end") - pl.col("start")).alias("length"),
            pl.col("text")
            .str.slice(pl.col("start"), pl.col("end") - pl.col("start"))
            .alias("sentence"),
        )
    )


def _add_removals(abstracts: pl.DataFrame, ranges: pl.DataFrame, *, reason: str) -> pl.DataFrame:
    """Add a removal for each (work id, start, end) not already removed, keeping ranges sorted."""
    existing = (
        abstracts.select("work_id", "removals").explode("removals").unnest("removals")
    ).select("work_id", "start", "end")
    added = (
        ranges.select("work_id", "start", "end")
        .unique()
        .join(existing, on=["work_id", "start", "end"], how="anti")
        .with_columns(pl.lit(reason).alias("reason"))
        .group_by("work_id")
        .agg(pl.struct("start", "end", "reason").alias("added"))
    )
    return (
        abstracts.join(added, on="work_id", how="left", maintain_order="left")
        .with_columns(
            pl.when(pl.col("added").is_null())
            .then(pl.col("removals"))
            .otherwise(
                pl.concat_list("removals", "added").list.eval(
                    pl.element().sort_by(
                        pl.element().struct.field("start"), pl.element().struct.field("end")
                    )
                )
            )
            .alias("removals")
        )
        .drop("added")
    )


def _mark_dropped(abstracts: pl.DataFrame, work_ids: pl.Series, *, reason: str) -> pl.DataFrame:
    """Set ``reason`` as the dropped reason of the given works not already dropped."""
    return abstracts.with_columns(
        pl.when(pl.col("dropped_reason").is_null() & pl.col("work_id").is_in(work_ids.implode()))
        .then(pl.lit(reason))
        .otherwise(pl.col("dropped_reason"))
        .alias("dropped_reason")
    )


def _stable_rank(work_id: str) -> str:
    return hashlib.sha256(work_id.encode()).hexdigest()
