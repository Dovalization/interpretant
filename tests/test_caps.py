"""Tests for author caps."""

from __future__ import annotations

from collections import Counter

from interpretant.corpus.caps import cap_author_works


def test_cap_author_works_limits_each_author_share() -> None:
    authors_per_work = [["consortium"] for _ in range(30)] + [[f"author{n}"] for n in range(70)]

    kept = cap_author_works(authors_per_work, max_share=0.05)

    counts = Counter(author for index in kept for author in authors_per_work[index])
    assert counts["consortium"] == 5
    assert all(counts[f"author{n}"] == 1 for n in range(70))


def test_cap_author_works_keeps_everything_when_no_author_dominates() -> None:
    authors_per_work = [[f"author{n}", f"coauthor{n}"] for n in range(200)]

    assert cap_author_works(authors_per_work, max_share=0.01) == set(range(200))
