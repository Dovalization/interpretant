"""Duplicates (decision agenda 2.5): one row per work, and near-duplicates found by MinHash.

The same work in two cells is one row listing every field it was sampled for, so it is kept in
every cell it belongs to. Near-duplicates (a preprint and its published version) are found with
MinHash and locality-sensitive hashing (``datasketch``) on word 5-shingles of the kept text, the
design's starting point; the earliest work of each group is kept and every later one is dropped
with the work it duplicates recorded.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import polars as pl
from datasketch import MinHash, MinHashLSH

from interpretant.catalog import RawWork
from interpretant.corpus.annotation import RawAbstract, Removal, kept_text

NEAR_DUPLICATE = "near_duplicate"
SHINGLE_WORDS = 5
PERMUTATION_COUNT = 128


def merge_cells(
    cells: Iterable[tuple[str, Iterable[RawWork]]], *, source: str
) -> list[RawAbstract]:
    """One abstract per work id, in order of first appearance, listing every field it came from.

    A work's text, year, type and authors come from its first appearance; a work fetched twice
    carries the same OpenAlex record.
    """
    first_seen: dict[str, RawWork] = {}
    fields_by_work: dict[str, set[str]] = {}
    for field, works in cells:
        for work in works:
            first_seen.setdefault(work.work_id, work)
            fields_by_work.setdefault(work.work_id, set()).add(field)
    return [
        RawAbstract(
            work_id=work_id,
            year=work.year,
            fields=tuple(sorted(fields_by_work[work_id])),
            source=source,
            document_type=work.document_type,
            author_ids=work.author_ids,
            text=work.text,
        )
        for work_id, work in first_seen.items()
    ]


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
        minhash = MinHash(num_perm=PERMUTATION_COUNT)
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
