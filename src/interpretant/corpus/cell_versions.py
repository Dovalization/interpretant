"""A cell's text under a build's removals, and the versions round S1 compares.

A work's static tokens (``corpus/static_view.py``) keep their character offsets, so a build's
removed ranges apply to them here: tokens inside a range are dropped and a range ends the line,
so no window crosses it. A version of a cell is its works with some left out and some spans cut;
placebos match a treatment's removal in works (within tenths of token length) and in tokens, and
bootstrap samples draw the cell's works with replacement. Every draw is seeded per version,
replicate and cell.
"""

from __future__ import annotations

import hashlib
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

# Tenths of token length within which a placebo replaces a dropped work.
LENGTH_BANDS = 10


@dataclass(frozen=True)
class Cut:
    """A version of a cell: the works left out, and a span of tokens cut from some works.

    A span is (first token, token count) in the work's kept token sequence.
    """

    dropped: frozenset[str] = frozenset()
    spans: Mapping[str, tuple[int, int]] = field(default_factory=dict)


# The version that cuts nothing.
NO_CUT = Cut()


def kept_segments(
    tokens: Sequence[str], starts: Sequence[int], removals: Sequence[tuple[int, int]]
) -> list[list[str]]:
    """The work's tokens outside the removed ranges, split wherever a range lies between two."""
    merged = _merged(removals)
    range_starts = [start for start, _ in merged]
    segments: list[list[str]] = []
    previous_key: int | None = None
    for token, start in zip(tokens, starts, strict=True):
        key = bisect_right(range_starts, start) - 1
        if key >= 0 and start < merged[key][1]:
            continue
        if key != previous_key or not segments:
            segments.append([])
            previous_key = key
        segments[-1].append(token)
    return segments


def cell_lines(
    works: Mapping[str, list[list[str]]], cut: Cut = NO_CUT, sample: Sequence[str] | None = None
) -> list[str]:
    """The cell's lines: each kept segment of each work, the cut applied.

    ``sample`` gives the works to use, repeats allowed (a bootstrap sample); by default every work
    once, in the mapping's order.
    """
    chosen = list(works) if sample is None else sample
    lines: list[str] = []
    for work_id in chosen:
        if work_id in cut.dropped:
            continue
        segments = works[work_id]
        if work_id in cut.spans:
            segments = _cut_span(segments, *cut.spans[work_id])
        lines.extend(" ".join(segment) for segment in segments if segment)
    return lines


def token_count(segments: list[list[str]]) -> int:
    """A work's kept tokens."""
    return sum(len(segment) for segment in segments)


def placebo_cut(
    works: Mapping[str, list[list[str]]],
    dropped: frozenset[str],
    passage_sizes: Mapping[str, int],
    rng: np.random.Generator,
) -> Cut:
    """A random cut matching a treatment's, from the works the treatment leaves alone.

    Each work the treatment drops is matched by a random work from the same tenth of token length
    (the nearest tenth that still has one), and each passage it removes by a span of the same
    token count at a random position in a random work long enough to hold it.
    """
    lengths = {work_id: token_count(segments) for work_id, segments in works.items()}
    edges = np.quantile(list(lengths.values()), np.linspace(0, 1, LENGTH_BANDS + 1)[1:-1])
    touched = set(dropped) | set(passage_sizes)
    pools: dict[int, list[str]] = {}
    for work_id in sorted(set(works) - touched):
        pools.setdefault(_band(lengths[work_id], edges), []).append(work_id)
    for band, pool in pools.items():
        pools[band] = [pool[index] for index in rng.permutation(len(pool))]
    placebo_dropped = {
        _take(pools, _band(lengths[work_id], edges)) for work_id in sorted(dropped & set(works))
    }
    spans: dict[str, tuple[int, int]] = {}
    for work_id in sorted(set(passage_sizes) & set(works)):
        size = passage_sizes[work_id]
        host = _take_long_enough(pools, lengths, size, rng)
        spans[host] = (int(rng.integers(0, lengths[host] - size + 1)), size)
    return Cut(dropped=frozenset(placebo_dropped), spans=spans)


def bootstrap_sample(work_ids: Sequence[str], rng: np.random.Generator) -> list[str]:
    """As many works as the cell has, drawn with replacement (Antoniak and Mimno 2018)."""
    return [work_ids[index] for index in rng.integers(0, len(work_ids), size=len(work_ids))]


def seeded(base_seed: int, replicate: int, cell: str) -> np.random.Generator:
    """The generator of one replicate of one cell: the same draws for the same three values.

    The cell enters as a hash of its whole name; until 2026-11-17 it entered as its first 16
    bytes, so *communication_1950* to *_1990* shared one stream and *_2000* to *_2020* another.
    """
    cell_key = int.from_bytes(hashlib.sha256(cell.encode("utf-8")).digest(), "big")
    return np.random.default_rng([base_seed, replicate, cell_key])


def _merged(removals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(removals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _cut_span(segments: list[list[str]], first: int, size: int) -> list[list[str]]:
    """The segments with tokens ``first`` to ``first + size`` removed; the cut ends a segment."""
    result: list[list[str]] = []
    position = 0
    for segment in segments:
        before = segment[: max(0, first - position)]
        after = segment[max(0, first + size - position) :]
        if position + len(segment) <= first or position >= first + size:
            result.append(segment)
        else:
            result.extend(part for part in (before, after) if part)
        position += len(segment)
    return result


def _band(length: int, edges: np.ndarray) -> int:
    return int(np.searchsorted(edges, length, side="right"))


def _take(pools: dict[int, list[str]], band: int) -> str:
    """A work from the band's pool, or from the nearest band that still has one."""
    for distance in range(LENGTH_BANDS + 1):
        for candidate in (band - distance, band + distance):
            if pools.get(candidate):
                return pools[candidate].pop()
    raise ValueError("the cell has too few untouched works for a placebo")


def _take_long_enough(
    pools: dict[int, list[str]], lengths: Mapping[str, int], size: int, rng: np.random.Generator
) -> str:
    """A random untouched work with at least ``size`` tokens."""
    candidates = [
        (band, index)
        for band, pool in sorted(pools.items())
        for index, work_id in enumerate(pool)
        if lengths[work_id] >= size
    ]
    if not candidates:
        raise ValueError(f"no untouched work has {size} tokens for a placebo span")
    band, index = candidates[int(rng.integers(0, len(candidates)))]
    return pools[band].pop(index)
