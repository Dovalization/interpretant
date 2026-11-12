"""Author caps — keep any single author from dominating a field × decade cell.

Drift between two decades should not be a change of authors. OpenAlex cells are capped by share
of works (``MAX_AUTHOR_SHARE``): ordinary prolific authors stay under it, while merged author
records and consortia credited on hundreds of papers do not.
"""

from __future__ import annotations

import random
from collections import Counter

MAX_AUTHOR_SHARE = 0.01


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
