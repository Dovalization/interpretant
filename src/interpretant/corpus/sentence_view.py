"""The contextual sentence view's rows (decision agenda 2.2; `docs/sentence_view.md`).

The contextual pipeline draws its usages from sentences. The spaCy pass already split each text
into sentences and the language stage read each one's language, so this module only decides what
each sentence of a kept work may be used for:

- **removed:** every non-space character lies inside the work's removed ranges, merged, so a
  sentence spanning two adjacent ranges counts as removed;
- **cut_off:** the last sentence of a work flagged truncated, whose text stops mid-sentence: its
  words are real uses up to the cut, so it stays in the view, but no usage is drawn from it;
- otherwise none. A partly covered sentence stays whole, so offsets never move; a mention inside a
  removed range is never a usage, which matching decides.

Sentences of undetermined language and of malformed works stay, with their language recorded.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import polars as pl

from interpretant.catalog.sentence_view import CUT_OFF, REMOVED, SENTENCE_VIEW_COLUMNS
from interpretant.corpus.cleaning import TRUNCATED


def merge_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """The ranges sorted by start, with overlapping or touching ranges joined into one."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def is_inside_removal(text: str, start: int, end: int, merged: Sequence[tuple[int, int]]) -> bool:
    """Whether every non-space character of ``text[start:end]`` lies inside the merged ranges.

    A span with no non-space character is not inside a removal: there is nothing to remove.
    """
    if not text[start:end].strip():
        return False
    position = start
    for low, high in merged:
        if high <= position:
            continue
        if low >= end:
            break
        if text[position:low].strip():
            return False
        position = max(position, high)
        if position >= end:
            return True
    return not text[position:end].strip()


def word_count(sentence: str) -> int:
    """The number of whitespace-separated pieces of the sentence that hold a letter."""
    return sum(1 for piece in sentence.split() if any(character.isalpha() for character in piece))


def build_sentence_view(abstracts: pl.DataFrame) -> pl.DataFrame:
    """One row per sentence of every kept work, in the table's order, with its exclusion."""
    rows = []
    kept = abstracts.filter(pl.col("dropped_reason").is_null())
    for work_id, text, sentences, removals, flags in kept.select(
        "work_id", "text", "sentences", "removals", "flags"
    ).iter_rows():
        merged = merge_ranges((removal["start"], removal["end"]) for removal in removals)
        last = len(sentences) - 1 if TRUNCATED in flags else None
        for number, sentence in enumerate(sentences):
            start, end = sentence["start"], sentence["end"]
            exclusion = None
            if merged and is_inside_removal(text, start, end, merged):
                exclusion = REMOVED
            elif number == last:
                exclusion = CUT_OFF
            rows.append(
                (
                    work_id,
                    number,
                    start,
                    end,
                    sentence["language"],
                    sentence["confidence"],
                    word_count(text[start:end]),
                    exclusion,
                )
            )
    return pl.DataFrame(rows, schema=SENTENCE_VIEW_COLUMNS, orient="row")
