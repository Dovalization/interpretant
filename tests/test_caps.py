"""Tests for author caps."""

from __future__ import annotations

from collections import Counter

from interpretant.corpus.caps import cap_author_tokens, cap_author_works, largest_author


def test_cap_author_tokens_keeps_small_authors_whole_and_samples_large_ones() -> None:
    documents = [
        ("James", " ".join(["psychology"] * 3_000)),
        ("James", " ".join(["pragmatism"] * 3_000)),
        ("Peirce", " ".join(["sign"] * 1_500)),
    ]

    kept = cap_author_tokens(documents, cap_tokens=2_000)

    assert " ".join(["sign"] * 1_500) in kept
    james_tokens = sum(len(doc.split()) for doc in kept if not doc.startswith("sign"))
    assert james_tokens == 2_000
    assert kept == cap_author_tokens(documents, cap_tokens=2_000)


def test_cap_author_works_limits_each_author_share() -> None:
    authors_per_work = [["consortium"] for _ in range(30)] + [[f"author{n}"] for n in range(70)]

    kept = cap_author_works(authors_per_work, max_share=0.05)

    counts = Counter(author for index in kept for author in authors_per_work[index])
    assert counts["consortium"] == 5
    assert all(counts[f"author{n}"] == 1 for n in range(70))


def test_cap_author_works_keeps_everything_when_no_author_dominates() -> None:
    authors_per_work = [[f"author{n}", f"coauthor{n}"] for n in range(200)]

    assert cap_author_works(authors_per_work, max_share=0.01) == set(range(200))


def test_largest_author_counts_tokens_across_books() -> None:
    documents = [("Brown", "a b c"), ("Mill", "a b"), ("Mill", "c d")]

    assert largest_author(documents) == "Mill"
