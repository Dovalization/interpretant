"""The contextual sentence view's rows (decision agenda 2.2; `docs/sentence_view.md`).

The contextual pipeline draws its usages from sentences. The spaCy pass already split each text
into sentences and the language stage read each one's language. This module joins the cuts the
merge rules find wrong and decides what each sentence of a kept work may be used for:

- **joined:** two consecutive sentences, neither removed, become one when a merge rule says the
  cut between them falls where no sentence ends (`join_rule`). The rules came from a scan of every
  cut in the corpus (`scripts/evidence/sentence_cuts.py`); wrong cuts gather in OCR-damaged early
  decades. A joined sentence takes the language and confidence of its longest part;
- **removed:** every non-space character lies inside the work's removed ranges, merged, so a
  sentence spanning two adjacent ranges counts as removed. A removed sentence is never joined;
- **cut_off:** the sentence holding the end of a work flagged truncated, whose text stops
  mid-sentence: its words are real uses up to the cut, so it stays in the view, but no usage is
  drawn from it;
- otherwise none. A partly covered sentence stays whole, so offsets never move; a mention inside a
  removed range is never a usage, which matching decides.

Sentences of undetermined language and of malformed works stay, with their language recorded.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

import polars as pl

from interpretant.catalog.sentence_view import (
    CUT_OFF,
    REMOVED,
    SENTENCE_VIEW_COLUMNS,
    JoinRule,
)
from interpretant.corpus.cleaning import TRUNCATED

FUNCTION_WORDS = frozenset(
    {
        "the", "a", "an", "of", "and", "or", "to", "in", "for", "with", "by", "on", "at", "from",
        "as", "that", "which", "its", "their", "his", "her", "our",
    }
)  # fmt: skip
_SPACED_DOTS = re.compile(r"\.\s+\.$")
_LIST_NUMBER = re.compile(r"(?<!\d)[:;]\s*\(?\d{1,2}[.)]$")
_FINAL_WORD = re.compile(r"(?<![A-Za-z])([A-Za-z]+)$")
# Abbreviations that never end a sentence, in their usual case (all-capital forms are acronyms).
_NEVER_FINAL = re.compile(
    r"(?<![A-Za-z])(?:ed|Ed|eds|Eds|trans|Trans|tr|Tr|cf|Cf|e\.g|i\.e|viz|Viz|vs|esp|univ|Univ"
    r"|dept|Dept|Dr|Prof|prof|Mr|Mrs)\.$"
)
# Reference abbreviations, which end a sentence only when no number follows them.
_REFERENCE = re.compile(
    r"(?<![A-Za-z])(?:p|pp|Pp|vol|Vol|VOL|vols|Vols|ch|Ch|chap|Chap|no|No|nos|Nos|fig|Fig|FIG"
    r"|figs|Figs|eq|Eq|eqs|Eqs|ref|Ref|refs|Refs|sec|Sec)\.$"
)
_NUMBER_START = re.compile(r"\d|[(\[]|(?:[IVXLC]+|[ivxlc]+)(?![A-Za-z])")
_SAINT = re.compile(r"(?<![A-Za-z])(?:St|ST)\.$")
_OPENERS, _CLOSERS = "([", ")]"


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


def join_rule(before: str, after: str) -> JoinRule | None:
    """The merge rule that joins ``after`` to the sentence ``before`` it, if one does."""
    if not _has_letter(after) or not _has_letter(before):
        return "no_letters"
    if after[:1] in ",;:)]}":
        return "closing_punctuation"
    if after[:1] == ".":
        return "period"
    if _SPACED_DOTS.search(before) and after[:1].islower():
        return "spaced_dots"
    if _LIST_NUMBER.search(before):
        return "list_number"
    final = _FINAL_WORD.search(before)
    if final and final.group(1).lower() in FUNCTION_WORDS:
        return "function_word"
    if _unclosed(before) > 0 and _closes_first(after):
        return "inside_brackets"
    if _NEVER_FINAL.search(before):
        return "abbreviation"
    if _REFERENCE.search(before) and _NUMBER_START.match(after):
        return "reference_number"
    if _SAINT.search(before) and after[:1].isupper():
        return "saint"
    return None


def build_sentence_view(abstracts: pl.DataFrame) -> pl.DataFrame:
    """One row per sentence of every kept work, in text order, joined, with its exclusion."""
    rows = []
    kept = abstracts.filter(pl.col("dropped_reason").is_null())
    for work_id, text, sentences, removals, flags in kept.select(
        "work_id", "text", "sentences", "removals", "flags"
    ).iter_rows():
        merged = merge_ranges((removal["start"], removal["end"]) for removal in removals)
        removed = [
            bool(merged) and is_inside_removal(text, s["start"], s["end"], merged)
            for s in sentences
        ]
        groups = _join(text, sentences, removed)
        last = len(sentences) - 1 if TRUNCATED in flags else None
        for number, (parts, rules) in enumerate(groups):
            start, end = sentences[parts[0]]["start"], sentences[parts[-1]]["end"]
            longest = sentences[
                max(parts, key=lambda i: sentences[i]["end"] - sentences[i]["start"])
            ]
            exclusion = None
            if len(parts) == 1 and removed[parts[0]]:
                exclusion = REMOVED
            elif last in parts:
                exclusion = CUT_OFF
            rows.append(
                (
                    work_id,
                    number,
                    start,
                    end,
                    longest["language"],
                    longest["confidence"],
                    word_count(text[start:end]),
                    exclusion,
                    rules,
                )
            )
    return pl.DataFrame(rows, schema=SENTENCE_VIEW_COLUMNS, orient="row")


def _join(
    text: str, sentences: Sequence[dict[str, Any]], removed: Sequence[bool]
) -> list[tuple[list[int], list[JoinRule]]]:
    """The sentences grouped by the merge rules: each group's sentence indexes and its rules."""
    groups: list[tuple[list[int], list[JoinRule]]] = []
    for index, sentence in enumerate(sentences):
        if groups and not removed[index] and not removed[groups[-1][0][-1]]:
            parts, rules = groups[-1]
            before = text[sentences[parts[0]]["start"] : sentences[parts[-1]]["end"]]
            rule = join_rule(before, text[sentence["start"] : sentence["end"]])
            if rule is not None:
                parts.append(index)
                rules.append(rule)
                continue
        groups.append(([index], []))
    return groups


def _has_letter(text: str) -> bool:
    return any(character.isalpha() for character in text)


def _unclosed(text: str) -> int:
    """How many brackets the text opens and leaves open."""
    return sum(
        text.count(opener) - text.count(closer)
        for opener, closer in zip(_OPENERS, _CLOSERS, strict=True)
    )


def _closes_first(text: str) -> bool:
    """Whether the text closes a bracket it never opened."""
    depth = 0
    for character in text:
        depth += (character in _OPENERS) - (character in _CLOSERS)
        if depth < 0:
            return True
    return False
