"""Near-duplicates (decision agenda 2.5), found by MinHash.

Exact duplicates, the same work in two cells, are merged earlier (``corpus.works``).
Near-duplicates (a preprint and its published version) are found on word 5-shingles of the kept
text: MinHash and locality-sensitive hashing (``datasketch``) propose candidate pairs, and each
candidate's exact Jaccard similarity decides, as datasketch's documentation advises for exact
results (decided 2026-11-16, the author: estimated from 128 permutations, a similarity of 0.8 has
a standard error of about 0.035, so decisions near the threshold changed with the hash). The
earliest work of each group is kept and every later one is dropped with the work it duplicates
recorded. Shingles are hashed with MurmurHash3 (``mmh3``), the faster
hash datasketch's documentation shows in place of its default SHA-1, which took 24% of the
corpus build. Each abstract's signature depends on its text alone, so signatures are made apart
(``minhash_signatures``, by part in worker processes) and only the comparison runs in order.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from typing import Any

import mmh3
import numpy as np
import polars as pl
from datasketch import LeanMinHash, MinHash, MinHashLSH

from interpretant.corpus.removals import Removal, kept_text

NEAR_DUPLICATE = "near_duplicate"
SHINGLE_WORDS = 5
PERMUTATION_COUNT = 128
# datasketch's default seed and permutation scheme, stated because a signature made apart is
# read back with them.
SEED = 1
SCHEME = "affine32"
_SHINGLE_CACHE_SIZE = 50_000


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


def _jaccard(first: frozenset[bytes], second: frozenset[bytes]) -> float:
    return len(first & second) / len(first | second)


def minhash_signatures(
    texts: Sequence[str], removals: Sequence[Sequence[Mapping[str, Any]]]
) -> tuple[np.ndarray, np.ndarray]:
    """Each text's MinHash signature, a row per text, and whether the text has any shingles.

    The shingles are those of the text outside its removed ranges (``build_shingles``); a text
    without any has an unused row.
    """
    minhash = MinHash(
        num_perm=PERMUTATION_COUNT, seed=SEED, hashfunc=mmh3.mmh3_32_uintdigest, scheme=SCHEME
    )
    signatures = np.empty((len(texts), PERMUTATION_COUNT), dtype=minhash.hashvalues.dtype)
    has_shingles = np.zeros(len(texts), dtype=bool)
    for row, (text, text_removals) in enumerate(zip(texts, removals, strict=True)):
        shingles = build_shingles(text, [Removal(**removal) for removal in text_removals])
        if not shingles:
            continue
        # One MinHash reset for each text: making one per text parsed its permutations again
        # each time (29% of the corpus build before datasketch could reuse them).
        minhash.clear()
        minhash.update_batch(list(shingles))
        signatures[row] = minhash.hashvalues
        has_shingles[row] = True
    return signatures, has_shingles


def mark_near_duplicates(
    abstracts: pl.DataFrame,
    *,
    threshold: float,
    candidate_threshold: float,
    signatures: Mapping[str, np.ndarray],
    advance: Callable[[int], None] | None = None,
) -> pl.DataFrame:
    """Drop each abstract too similar to an earlier kept one, recording that work.

    An abstract is a near-duplicate when the exact Jaccard similarity of its shingles with a kept
    abstract's reaches ``threshold``; the kept work goes into ``duplicate_of``. Only abstracts
    not already dropped take part, each with its signature from ``signatures``
    (``minhash_signatures``, by work id; an abstract without shingles has none and is left as it
    is). They are visited earliest first (year, then work id); the LSH index, tuned for
    ``candidate_threshold``, proposes kept abstracts, and if any reaches ``threshold`` the
    abstract is marked a duplicate of the most similar (the earliest on a tie); otherwise it is
    kept and indexed. Exact duplicates are the special case of similarity 1. ``advance`` is
    called with 1 as each abstract is visited, for a progress display.

    Raises:
        ValueError: if ``candidate_threshold`` is above ``threshold``.
    """
    if candidate_threshold > threshold:
        raise ValueError("the candidate threshold must not be above the near-duplicate threshold")
    candidates = abstracts.filter(pl.col("dropped_reason").is_null()).sort("year", "work_id")
    order = {work_id: position for position, work_id in enumerate(candidates["work_id"])}
    rows = dict(
        zip(
            candidates["work_id"],
            zip(candidates["text"], candidates["removals"], strict=True),
            strict=True,
        )
    )

    # Each abstract's shingles are made when a comparison needs them; a bounded cache keeps those
    # of the abstracts most recently compared, where held for every abstract they took 15 GB.
    @lru_cache(maxsize=_SHINGLE_CACHE_SIZE)
    def shingles_of(work_id: str) -> frozenset[bytes]:
        text, removals = rows[work_id]
        return frozenset(build_shingles(text, [Removal(**removal) for removal in removals]))

    lsh_index = MinHashLSH(threshold=candidate_threshold, num_perm=PERMUTATION_COUNT)
    duplicate_of: dict[str, str] = {}
    for work_id in candidates["work_id"]:
        if advance is not None:
            advance(1)
        signature = signatures.get(work_id)
        if signature is None:
            continue
        minhash = LeanMinHash(seed=SEED, hashvalues=signature, scheme=SCHEME)
        proposed = lsh_index.query(minhash)
        if proposed:
            own = shingles_of(work_id)
            similarities = [(_jaccard(own, shingles_of(key)), key) for key in proposed]
            matches = [match for match in similarities if match[0] >= threshold]
            if matches:
                _, original = max(matches, key=lambda match: (match[0], -order[match[1]]))
                duplicate_of[work_id] = original
                continue
        lsh_index.insert(work_id, minhash)
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
