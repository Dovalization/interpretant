"""The contextual sentence view's rows (decision agenda 2.2; `docs/sentence_view.md`).

The contextual pipeline draws its usages from sentences. The spaCy pass already split each text
into sentences and the language stage read each one's language. This module joins the cuts the
merge rules find wrong and decides what each sentence of a kept work may be used for:

- **joined:** two consecutive sentences become one when a merge rule says the cut between them
  falls where no sentence ends (`join_rule`); a removed sentence joins only if it has no letters.
  The rules came from a scan of every cut in the corpus (`scripts/evidence/sentence_cuts.py`);
  wrong cuts gather in OCR-damaged early decades. A joined sentence takes the language and
  confidence of its longest part;
- **split:** a sentence is cut where the spaCy pass missed a break after a capital or short
  acronym with a period, before a common sentence opener (`missed_breaks`);
- **removed:** every non-space character lies inside the work's removed ranges, merged, so a
  sentence spanning two adjacent ranges counts as removed;
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
    r"(?<![A-Za-z])(?:ed|Ed|eds|Eds|trans|Trans|tr|Tr|cf|Cf|e\.g|eg|i\.e|viz|Viz|vs|esp|univ|Univ"
    r"|dept|Dept|Dr|Prof|prof|Mr|Mrs)\.$"
)
# Reference abbreviations, which end a sentence only when no number follows them.
_REFERENCE = re.compile(
    r"(?<![A-Za-z])(?:p|pp|Pp|vol|Vol|VOL|vols|Vols|ch|Ch|chap|Chap|no|No|nos|Nos|fig|Fig|FIG"
    r"|figs|Figs|eq|Eq|eqs|Eqs|ref|Ref|refs|Refs|sec|Sec)\.$"
)
_NUMBER_START = re.compile(r"\d|[(\[]|(?:[IVXLC]+|[ivxlc]+)(?![A-Za-z])")
_SAINT = re.compile(r"(?<![A-Za-z])St\.$")
# A word ending like a sentence: two lowercase letters, closing marks, then a stop.
_SENTENCE_END = re.compile(r"[a-z]{2}[)\]\"'”’]*[.!?]$")
# A break the spaCy pass missed: a capital or short acronym with a period, after a word that starts
# lowercase or a number (so not a name's initial, a reference's or a contents line's label), before
# a common sentence opener that is not itself an initial (*over a VP. Also*, *of X. This is*). *I.*
# is left alone: mostly a label.
OPENERS = (
    "The|This|These|That|Those|A|An|In|We|It|Its|Our|Also|However|Thus|Such|There|They|He|She"
    "|Finally|Moreover|Here|Furthermore|Although|When|Then|As|For|One|Two|Both|Each|No"
)
_MISSED_BREAK = re.compile(
    rf"(?:(?<=[a-z0-9] )|(?<=\d))(?!I\.)[A-Z]{{1,3}}\.(\s+)(?=(?:{OPENERS})\b)(?!A\.)"
)
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
    if after[:1] == "." and not after[1:2].isupper():
        return "period"
    if _SPACED_DOTS.search(before) and after[:1].islower():
        return "spaced_dots"
    if _LIST_NUMBER.search(before):
        return "list_number"
    final = _FINAL_WORD.search(before)
    if final and final.group(1).lower() in FUNCTION_WORDS:
        return "function_word"
    sentence_ends = bool(_SENTENCE_END.search(before)) and _starts_upper(after)
    if _unclosed(before) > 0 and _closes_first(after) and not sentence_ends:
        return "inside_brackets"
    if _NEVER_FINAL.search(before):
        return "abbreviation"
    if _REFERENCE.search(before) and _NUMBER_START.match(after):
        return "reference_number"
    if _SAINT.search(before) and after[:1].isupper():
        return "saint"
    return None


def missed_breaks(sentence: str) -> list[tuple[int, int]]:
    """Where the split rule cuts a sentence: the first piece's end and the next one's start."""
    cuts = []
    for match in _MISSED_BREAK.finditer(sentence):
        before = sentence[: match.start()]
        words = before.split()
        previous = words[-1].lstrip(".,;:([\"'“‘") if words else ""
        if before[-1:].isdigit() or previous[:1].islower() or previous[:1].isdigit():
            cuts.append((match.start(1), match.end(1)))
    return cuts


def build_sentence_view(abstracts: pl.DataFrame) -> pl.DataFrame:
    """One row per sentence of every kept work, in text order, joined and split, excluded or not."""
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
        last = len(sentences) - 1 if TRUNCATED in flags else None
        number = 0
        for parts, rules in _join(text, sentences, removed):
            start, end = sentences[parts[0]]["start"], sentences[parts[-1]]["end"]
            longest = sentences[
                max(parts, key=lambda i: sentences[i]["end"] - sentences[i]["start"])
            ]
            is_removed = len(parts) == 1 and removed[parts[0]]
            cuts = [] if is_removed else missed_breaks(text[start:end])
            pieces = list(
                zip(
                    [start] + [start + after for _, after in cuts],
                    [start + before for before, _ in cuts] + [end],
                    strict=True,
                )
            )
            joins = [sentences[part]["start"] for part in parts[1:]]
            for index, (low, high) in enumerate(pieces):
                exclusion = None
                if is_removed:
                    exclusion = REMOVED
                elif last in parts and index == len(pieces) - 1:
                    exclusion = CUT_OFF
                upper = pieces[index + 1][0] if index + 1 < len(pieces) else end + 1
                piece_rules = [
                    rule for rule, at in zip(rules, joins, strict=True) if low <= at < upper
                ]
                rows.append(
                    (
                        work_id,
                        number,
                        low,
                        high,
                        longest["language"],
                        longest["confidence"],
                        word_count(text[low:high]),
                        exclusion,
                        piece_rules,
                        index > 0,
                    )
                )
                number += 1
    return pl.DataFrame(rows, schema=SENTENCE_VIEW_COLUMNS, orient="row")


def _join(
    text: str, sentences: Sequence[dict[str, Any]], removed: Sequence[bool]
) -> list[tuple[list[int], list[JoinRule]]]:
    """The sentences grouped by the merge rules: each group's sentence indexes and its rules.

    A removed sentence is never joined, unless it has no letters: a stray mark such as *.* or *2*
    is removed as a template sentence for being common, not because it is a citation or a notice.
    A row of no letters that holds a digit, before a row starting with *.* or *)*, is a list number
    cut from its item, so it starts a group and the item joins it.
    """
    rows = [text[sentence["start"] : sentence["end"]] for sentence in sentences]
    joinable = [not removed[i] or not _has_letter(row) for i, row in enumerate(rows)]
    groups: list[tuple[list[int], list[JoinRule]]] = []
    for index, row in enumerate(rows):
        following = rows[index + 1] if index + 1 < len(rows) else ""
        list_number = not _has_letter(row) and _has_digit(row) and following[:1] in (".", ")")
        if groups and joinable[index] and joinable[groups[-1][0][-1]] and not list_number:
            parts, rules = groups[-1]
            before = text[sentences[parts[0]]["start"] : sentences[parts[-1]]["end"]]
            rule = join_rule(before, row)
            if rule is not None:
                parts.append(index)
                rules.append(rule)
                continue
        groups.append(([index], []))
    return groups


def _has_letter(text: str) -> bool:
    return any(character.isalpha() for character in text)


def _has_digit(text: str) -> bool:
    return any(character.isdigit() for character in text)


def _starts_upper(text: str) -> bool:
    """Whether the text, opening quotes and brackets aside, starts with a capital."""
    return text.lstrip("\"'“‘([")[:1].isupper()


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
