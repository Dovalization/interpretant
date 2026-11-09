"""Author caps — keep any single author from dominating a field × decade cell.

Drift between two decades should not be a change of authors. Books are capped by tokens per
author (``BOOK_AUTHOR_TOKENS``, about two median books), so a capped author contributes a
seeded sample of chunks from all their books. OpenAlex cells are capped by share of works
(``MAX_AUTHOR_SHARE``): ordinary prolific authors stay under it, while merged author records
and consortia credited on hundreds of papers do not.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict

BOOK_AUTHOR_TOKENS = 250_000
MAX_AUTHOR_SHARE = 0.01
CHUNK_TOKENS = 1_000


def cap_author_tokens(
    documents: list[tuple[str, str]], cap_tokens: int = BOOK_AUTHOR_TOKENS, seed: int = 42
) -> list[str]:
    """Return documents with each author limited to ``cap_tokens`` tokens.

    Args:
        documents: ``(author, text)`` pairs, one per book.
        cap_tokens: Token budget per author.
        seed: Seed for choosing which chunks a capped author keeps.

    Returns:
        Uncapped authors' books unchanged, then each capped author's sampled chunks, each
        chunk as its own document.
    """
    by_author: dict[str, list[list[str]]] = defaultdict(list)
    for author, text in documents:
        by_author[author].append(text.split())
    kept: list[str] = []
    rng = random.Random(seed)
    for books in by_author.values():
        if sum(len(book) for book in books) <= cap_tokens:
            kept.extend(" ".join(book) for book in books)
            continue
        chunks = [
            book[start : start + CHUNK_TOKENS]
            for book in books
            for start in range(0, len(book), CHUNK_TOKENS)
        ]
        rng.shuffle(chunks)
        total = 0
        for chunk in chunks:
            if total + len(chunk) > cap_tokens:
                break
            kept.append(" ".join(chunk))
            total += len(chunk)
    return kept


def cap_author_works(
    authors_per_work: list[list[str]], max_share: float = MAX_AUTHOR_SHARE, seed: int = 42
) -> set[int]:
    """Return the indices of works to keep so no author appears in more than ``max_share``.

    Works are dropped at random from each over-represented author, most frequent first;
    dropping a co-authored work also lowers its co-authors' counts.
    """
    limit = max(1, int(max_share * len(authors_per_work)))
    kept = set(range(len(authors_per_work)))
    rng = random.Random(seed)
    for author, initial_count in Counter(
        author for authors in authors_per_work for author in set(authors)
    ).most_common():
        if initial_count <= limit:
            break
        works = sorted(index for index in kept if author in authors_per_work[index])
        for index in rng.sample(works, max(0, len(works) - limit)):
            kept.discard(index)
    return kept


def largest_author(documents: list[tuple[str, str]]) -> str:
    """Return the author with the most tokens, for leave-one-author-out checks."""
    tokens: Counter[str] = Counter()
    for author, text in documents:
        tokens[author] += len(text.split())
    return tokens.most_common(1)[0][0]
