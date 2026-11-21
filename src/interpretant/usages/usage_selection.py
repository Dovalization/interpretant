"""Which usages are in scope, and the one usage drawn per abstract and word (agenda 4.1, 4.3).

A usage is in scope when its part of speech is one its word table gives and it does not sit inside a
name (*medium* is a noun, not the adjective of *medium effect size*; *network* in *Cable News
Network* is a name, while a generic term the recognizer labels a name is not, as
`entities.entity_over_target` decides). The usages table keeps every match; measures read in-scope
usages by default.

One usage is drawn per abstract and word, among its in-scope usages in sentences no exclusion
keeps out (not removed, not cut off): the one with the smallest draw key, a cryptographic hash of
the seed, the work, the word and the content key (`usages/identifiers.py`). So abstracts stay
independent observations, and a draw never depends on other abstracts, other words or row order.
A work that copies an earlier one (three or more target sentences at the same places) keeps its
usages, marked with the work it copies, but none of them is drawn.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from itertools import combinations

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


def is_in_scope(pos: str, word: Word, *, inside_name: bool, hyphen_part: bool) -> bool:
    """Whether a usage has a part of speech its word allows and sits inside no name.

    Whether an entity puts the target inside a name is decided from the entity itself
    (`entities.entity_over_target`). A target that is one part of a hyphenated word is in scope
    whatever that word's tag (agenda 4.2: compounds are in scope by default): the tagger tags the
    whole word, so *attention* in *attention-based* would read as an adjective. The exception is a
    word the tagger reads as foreign (X): the usages audit's hyphen parts tagged X were all French
    (*lui-même*; `docs/usages_audit.md`).
    """
    allowed = pos in word.pos or (hyphen_part and pos != "X")
    return allowed and not inside_name


def mark_drawn(usages: pl.DataFrame) -> pl.DataFrame:
    """The usages with ``drawn`` set on the smallest-keyed eligible usage per work and word.

    Eligible means in scope and in a sentence without an exclusion; the table needs the columns
    ``work_id``, ``word``, ``draw_key``, ``in_scope`` and ``sentence_exclusion``.
    """
    eligible = pl.col("in_scope") & pl.col("sentence_exclusion").is_null()
    smallest = pl.col("draw_key").filter(eligible).min().over("work_id", "word")
    return usages.with_columns((eligible & (pl.col("draw_key") == smallest)).alias("drawn"))


# Works sharing this many target sentences at the same places are one work twice: the same paper
# indexed twice below the corpus's near-duplicate threshold (`docs/usages_audit.md`, 2026-11-20).
COPY_SHARED_SENTENCES = 3


def find_copies(usages: pl.DataFrame, years: Mapping[str, int]) -> dict[str, str]:
    """Each copy and the work it copies.

    Two works are linked when they share ``COPY_SHARED_SENTENCES`` or more target sentences at the
    same places (usages with one content key, counted by sentence); in each group of linked works
    the earliest (by year, then by id) is kept and the others are its copies. The table needs the
    columns ``work_id``, ``content_key`` and ``sentence_number``.
    """
    shared = (
        usages.group_by("content_key", "work_id")
        .agg(pl.col("sentence_number").min())
        .group_by("content_key")
        .agg(pl.col("work_id"), pl.col("sentence_number"))
        .filter(pl.col("work_id").list.len() > 1)
    )
    sentences: dict[tuple[str, str], set[int]] = defaultdict(set)
    for ids, numbers in shared.select("work_id", "sentence_number").iter_rows():
        for (first, number), (second, _) in combinations(sorted(zip(ids, numbers, strict=True)), 2):
            sentences[(first, second)].add(number)
    group = _Groups()
    for (first, second), numbers in sentences.items():
        if len(numbers) >= COPY_SHARED_SENTENCES:
            group.join(first, second)
    copies: dict[str, str] = {}
    for members in group.members():
        kept = min(members, key=lambda work: (years[work], work))
        copies |= {work: kept for work in members if work != kept}
    return copies


def mark_copies(usages: pl.DataFrame, copies: Mapping[str, str]) -> pl.DataFrame:
    """The usages with ``copy_of`` set from the copies, and a copy's usages not drawn."""
    copy_of = pl.col("work_id").replace_strict(dict(copies), default=None, return_dtype=pl.String)
    return usages.with_columns(copy_of.alias("copy_of")).with_columns(
        (pl.col("drawn") & pl.col("copy_of").is_null()).alias("drawn")
    )


class _Groups:
    """Works joined into groups (union-find)."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def root(self, work: str) -> str:
        """The work standing for the work's group."""
        self.parent.setdefault(work, work)
        while self.parent[work] != work:
            self.parent[work] = self.parent[self.parent[work]]
            work = self.parent[work]
        return work

    def join(self, first: str, second: str) -> None:
        """Put the two works' groups together."""
        self.parent[self.root(first)] = self.root(second)

    def members(self) -> list[list[str]]:
        """Every group's works."""
        groups: dict[str, list[str]] = defaultdict(list)
        for work in list(self.parent):
            groups[self.root(work)].append(work)
        return list(groups.values())
