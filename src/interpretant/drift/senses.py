"""Occurrences of a word by field and decade, for the contextual measures and the sense labels.

The cased abstracts behind the cleaned cells, with abstracts found in more than one field's cell
removed on request, and seeded samples of a word's occurrences in them: up to ``CAP`` abstracts
per cell that contain the word, one occurrence each, drawn per seed. (The sense clustering that
also lived here was deleted in step 1b; rounds C, C′ and C″ are recorded in docs/experiments.md.)
"""

from __future__ import annotations

import gzip
import json
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from interpretant.corpus.openalex import OpenAlexSource, cell_path, strip_boilerplate

CAP = 500
CONTEXT_WORDS = 100

Abstracts = Mapping[tuple[str, int], Sequence[tuple[str, str]]]

_LABEL = re.compile(r"^\s*(?:abstract|summary)\b[\s:.\-–—]*", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"“])")
_URL = re.compile(r"https?://\S+|www\.\S+|\bdoi:\s*\S+|\b10\.\d{4,9}/\S+", re.IGNORECASE)
MAX_FOREIGN_SHARE = 0.2


@dataclass(frozen=True)
class Occurrence:
    """One use of a word: where it was found, and the context the encoder reads."""

    field: str
    decade: int
    work_id: str
    context: str
    start: int
    end: int


@dataclass(frozen=True)
class Sampling:
    """How occurrences are drawn: up to ``cap`` per cell, with or without replacement, the
    word alone or with its regular plural, and with or without round C′'s quality rules
    (abstracts mostly outside the Latin script skipped, URLs and DOIs removed)."""

    cap: int = CAP
    replace: bool = False
    plural: bool = False
    quality: bool = False


ROUND_C = Sampling()


def foreign_share(text: str) -> float:
    """The share of a text's letters outside the Latin script."""
    letters = [character for character in text if character.isalpha()]
    return sum(ord(character) > 0x024F for character in letters) / max(len(letters), 1)


def cased_abstracts(raw_dir: Path, field: str, decade: int) -> list[tuple[str, str]]:
    """``(work id, abstract)`` for every document of a cleaned cell, with case and punctuation
    kept, boilerplate stripped and a leading "Abstract" label removed."""
    kept = set(OpenAlexSource(raw_dir, field).cell(decade).ids)
    with gzip.open(cell_path(raw_dir, field, decade), "rt", encoding="utf-8") as cell:
        records = [json.loads(line) for line in cell]
    return [
        (str(record["id"]), _LABEL.sub("", strip_boilerplate(str(record["abstract"]))))
        for record in records
        if record["id"] in kept
    ]


def drop_shared(abstracts: Abstracts) -> dict[tuple[str, int], list[tuple[str, str]]]:
    """Remove every abstract that appears in more than one field's cell of the same decade."""
    cells: Counter[tuple[str, int]] = Counter(
        (work_id, decade)
        for (_, decade), documents in abstracts.items()
        for work_id, _ in documents
    )
    return {
        (field, decade): [
            (work_id, text) for work_id, text in documents if cells[(work_id, decade)] == 1
        ]
        for (field, decade), documents in abstracts.items()
    }


def word_pattern(word: str, plural: bool = False) -> re.Pattern[str]:
    """The word as a whole word, in any case, and with ``plural`` its regular plural too; a
    hyphenated compound (*self-attention*) counts."""
    suffix = "s?" if plural else ""
    return re.compile(rf"\b{re.escape(word)}{suffix}\b", re.IGNORECASE)


def context_around(
    text: str, start: int, end: int, max_words: int = CONTEXT_WORDS
) -> tuple[str, int, int]:
    """The sentence holding ``text[start:end]``, cut to ``max_words`` words around it when
    longer, with the occurrence's offsets in that context."""
    sentence_start = 0
    for boundary in _SENTENCE_END.finditer(text):
        if boundary.end() > start:
            break
        sentence_start = boundary.end()
    following = _SENTENCE_END.search(text, end)
    sentence_end = following.start() if following else len(text)
    words = list(re.finditer(r"\S+", text[sentence_start:sentence_end]))
    if len(words) > max_words:
        target = next(
            index for index, word in enumerate(words) if word.end() + sentence_start > start
        )
        first = min(max(target - max_words // 2, 0), len(words) - max_words)
        last = first + max_words - 1
        sentence_end = sentence_start + words[last].end()
        sentence_start += words[first].start()
    return text[sentence_start:sentence_end], start - sentence_start, end - sentence_start


def sample_occurrences(
    abstracts: Abstracts, word: str, seed: int, sampling: Sampling = ROUND_C
) -> list[Occurrence]:
    """The first occurrence of ``word`` in each of up to ``sampling.cap`` abstracts per cell
    that contain it: the first in a seeded random order, or, with ``sampling.replace``, as many
    as the cell has (up to the cap) drawn with replacement, so seeds differ in small cells."""
    pattern = word_pattern(word, sampling.plural)
    occurrences: list[Occurrence] = []
    for (field, decade), documents in sorted(abstracts.items()):
        order = list(documents)
        rng = random.Random(f"{seed}:{word}:{field}:{decade}")
        rng.shuffle(order)
        found: list[Occurrence] = []
        for work_id, raw in order:
            text = _URL.sub(" ", raw) if sampling.quality else raw
            if sampling.quality and foreign_share(text) > MAX_FOREIGN_SHARE:
                continue
            match = pattern.search(text)
            if match is None:
                continue
            context, start, end = context_around(text, match.start(), match.end())
            found.append(Occurrence(field, decade, work_id, context, start, end))
            if not sampling.replace and len(found) == sampling.cap:
                break
        size = min(len(found), sampling.cap)
        occurrences.extend(rng.choices(found, k=size) if sampling.replace else found)
    return occurrences


_WORD_TOKEN = re.compile(r"\w+")


def sample_many(
    abstracts: Abstracts, plan: Sequence[tuple[str, Sampling]], seed: int
) -> dict[str, list[Occurrence]]:
    """:func:`sample_occurrences` for many words in one pass over each cell, with the same
    result for every word: each abstract is cleaned and tokenised once, and each word's seeded
    order is walked only through the abstracts that contain it. Sampling with replacement is not
    supported, and each word must be a single run of word characters."""
    for word, sampling in plan:
        if sampling.replace or not _WORD_TOKEN.fullmatch(word):
            raise ValueError(f"sample_many cannot draw {word!r} with {sampling}")
    result: dict[str, list[Occurrence]] = {word: [] for word, _ in plan}
    for (field, decade), documents in sorted(abstracts.items()):
        prepared: dict[bool, list[tuple[str, set[str]] | None]] = {}
        for quality in {sampling.quality for _, sampling in plan}:
            rows: list[tuple[str, set[str]] | None] = []
            for _, raw in documents:
                text = _URL.sub(" ", raw) if quality else raw
                if quality and foreign_share(text) > MAX_FOREIGN_SHARE:
                    rows.append(None)
                else:
                    rows.append((text, set(_WORD_TOKEN.findall(text.lower()))))
            prepared[quality] = rows
        for word, sampling in plan:
            rows = prepared[sampling.quality]
            target = word.lower()
            order = list(range(len(documents)))
            random.Random(f"{seed}:{word}:{field}:{decade}").shuffle(order)
            pattern = word_pattern(word, sampling.plural)
            found = result[word]
            taken = 0
            for index in order:
                row = rows[index]
                if row is None:
                    continue
                text, tokens = row
                if target not in tokens and not (sampling.plural and target + "s" in tokens):
                    continue
                match = pattern.search(text)
                if match is None:
                    continue
                context, start, end = context_around(text, match.start(), match.end())
                found.append(Occurrence(field, decade, documents[index][0], context, start, end))
                taken += 1
                if taken == sampling.cap:
                    break
    return result
