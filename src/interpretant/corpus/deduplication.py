"""Near-duplicates (decision agenda 2.5), found by MinHash.

Exact duplicates, the same work in two cells, are merged earlier (``corpus.works``).
Near-duplicates (a preprint and its published version) are found with
MinHash and locality-sensitive hashing (``datasketch``) on word 5-shingles of the kept text, the
design's starting point; the earliest work of each group is kept and every later one is dropped
with the work it duplicates recorded.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import polars as pl
from datasketch import MinHash, MinHashLSH

from interpretant.corpus.removals import Removal, kept_text

NEAR_DUPLICATE = "near_duplicate"
SHINGLE_WORDS = 5
PERMUTATION_COUNT = 128


def build_shingles(text: str, removals: Sequence[Removal]) -> set[bytes]:
    """The word 5-shingles of the text outside the removed ranges, lowercased.

    Words are split on whitespace: the shingles only need to repeat between near-identical
    texts, so punctuation attached to a word does no harm. A text shorter than five words is one
    shingle; an empty one has none.
    """
    words = kept_text(text, removals).lower().split()
    if len(words) < SHINGLE_WORDS:
        return {" ".join(words).encode()} if words else set()
    return {
        " ".join(words[start : start + SHINGLE_WORDS]).encode()
        for start in range(len(words) - SHINGLE_WORDS + 1)
    }


def mark_near_duplicates(
    abstracts: pl.DataFrame,
    *,
    threshold: float,
    advance: Callable[[int], None] | None = None,
) -> pl.DataFrame:
    """Drop each abstract too similar to an earlier kept one, recording that work.

    An abstract is a near-duplicate when its estimated Jaccard similarity to a kept abstract
    reaches ``threshold``; the kept work goes into ``duplicate_of``. Only abstracts not already
    dropped take part. They are visited earliest first (year, then work id); each is compared
    with the kept abstracts the LSH index proposes and, if it matches, is marked a duplicate of
    its most similar match (the earliest on a tie); otherwise it is kept and indexed. Exact
    duplicates are the special case of similarity 1. ``advance`` is called with 1 as each
    abstract is visited, for a progress display.
    """
    candidates = abstracts.filter(pl.col("dropped_reason").is_null()).sort("year", "work_id")
    order = {work_id: position for position, work_id in enumerate(candidates["work_id"])}
    lsh_index = MinHashLSH(threshold=threshold, num_perm=PERMUTATION_COUNT)
    # Every signature reuses the first MinHash's permutations and scheme, the same for one seed:
    # made again for each abstract they took 29% of the corpus build (run 7c's profile), and
    # datasketch takes them from another MinHash for faster initialization (with its scheme
    # named, since version 2.0.0).
    first = MinHash(num_perm=PERMUTATION_COUNT)
    kept: dict[str, MinHash] = {}
    duplicate_of: dict[str, str] = {}
    # Each abstract's shingles are built as it is visited and only its signature is kept: held
    # all at once, the shingles of the whole corpus take about 15 GB.
    rows = zip(candidates["work_id"], candidates["text"], candidates["removals"], strict=True)
    for work_id, text, removals in rows:
        if advance is not None:
            advance(1)
        shingles = build_shingles(text, [Removal(**removal) for removal in removals])
        if not shingles:
            continue
        minhash = MinHash(
            num_perm=PERMUTATION_COUNT, permutations=first.permutations, scheme=first.scheme
        )
        minhash.update_batch(list(shingles))
        similarities = [(minhash.jaccard(kept[key]), key) for key in lsh_index.query(minhash)]
        matches = [match for match in similarities if match[0] >= threshold]
        if matches:
            _, original = max(matches, key=lambda match: (match[0], -order[match[1]]))
            duplicate_of[work_id] = original
            continue
        lsh_index.insert(work_id, minhash)
        kept[work_id] = minhash
    found = pl.DataFrame(
        {"work_id": list(duplicate_of), "original": list(duplicate_of.values())},
        schema={"work_id": pl.String(), "original": pl.String()},
    )
    return (
        abstracts.join(found, on="work_id", how="left", maintain_order="left")
        .with_columns(
            pl.when(pl.col("original").is_not_null())
            .then(pl.lit(NEAR_DUPLICATE))
            .otherwise(pl.col("dropped_reason"))
            .alias("dropped_reason"),
            pl.coalesce("original", "duplicate_of").alias("duplicate_of"),
        )
        .drop("original")
    )
