"""Quality rules, version 2 (docs/corpus.md): records that are not English research abstracts.

A corpus-wide scan on 2026-11-11 found four problems the earlier rules let through: abstracts in
other languages written in the Latin script, records that are not research at all (digitised
student newspapers, agency newsletters) made of one template sentence repeated across hundreds of
records, notice sentences appended to real abstracts by publishers and indexers, and exam-dump
advertising. :func:`clean_cells` drops the first, second and fourth kind and strips the notices
from the rest. Earlier rounds did not apply it; every round from 2026-11-11 on says whether it
does.
"""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

MIN_WORDS = 20
MIN_ENGLISH_SHARE = 0.15
SHARED_MIN_ABSTRACTS = 20
TEMPLATE_SHARE = 0.5
SENTENCE_CHARS = (40, 300)

_WORD = re.compile(r"[a-z]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_SPAM = re.compile(
    r"\b(?:brain ?dumps?|exam dumps?|dumps (?:pdf|questions)|dumps4\w*|pass4sure|"
    r"certification exams?|exam questions and answers|actual (?:exam )?questions)\b",
    re.IGNORECASE,
)

Abstracts = Mapping[tuple[str, int], Sequence[tuple[str, str]]]


def english_share(text: str) -> float:
    """The share of a text's words that are English function words (*the*, *of*, *and*...)."""
    words = _WORD.findall(text.lower())
    return sum(word in ENGLISH_STOP_WORDS for word in words) / max(len(words), 1)


def sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE.split(text) if part.strip()]


def _key(sentence: str) -> str | None:
    low, high = SENTENCE_CHARS
    return sentence.lower() if low <= len(sentence) <= high else None


def shared_sentences(texts: Iterable[str], min_abstracts: int = SHARED_MIN_ABSTRACTS) -> set[str]:
    """Sentences (lowercased, 40 to 300 characters) found in ``min_abstracts`` or more texts."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update({key for s in sentences(text) if (key := _key(s)) is not None})
    return {sentence for sentence, count in counts.items() if count >= min_abstracts}


def clean_abstract(text: str, shared: set[str] | frozenset[str]) -> str | None:
    """The abstract with shared notice sentences removed, or None when the record is dropped:
    a text of ``MIN_WORDS`` or more words under ``MIN_ENGLISH_SHARE`` English function words, a
    template record (``TEMPLATE_SHARE`` or more of its characters in shared sentences), or
    exam-dump advertising."""
    if _SPAM.search(text):
        return None
    if len(_WORD.findall(text.lower())) >= MIN_WORDS and english_share(text) < MIN_ENGLISH_SHARE:
        return None
    parts = sentences(text)
    repeated = [part for part in parts if _key(part) in shared]
    if not repeated:
        return text
    if sum(len(part) for part in repeated) >= TEMPLATE_SHARE * max(len(text), 1):
        return None
    return " ".join(part for part in parts if _key(part) not in shared)


def clean_cells(
    abstracts: Abstracts, shared: set[str] | frozenset[str]
) -> dict[tuple[str, int], list[tuple[str, str]]]:
    """Every cell's ``(work id, abstract)`` pairs after :func:`clean_abstract`, dropped records
    left out."""
    cleaned: dict[tuple[str, int], list[tuple[str, str]]] = {}
    for cell, documents in abstracts.items():
        kept = []
        for work_id, text in documents:
            result = clean_abstract(text, shared)
            if result is not None:
                kept.append((work_id, result))
        cleaned[cell] = kept
    return cleaned


def grid_shared_sentences(raw_dir: Path, cache: Path) -> frozenset[str]:
    """The shared sentences of the whole grid under ``raw_dir``, computed once and cached."""
    if cache.exists():
        return frozenset(json.loads(cache.read_text(encoding="utf-8")))

    def texts() -> Iterable[str]:
        for path in sorted(raw_dir.glob("*/*.jsonl.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as cell:
                for line in cell:
                    yield str(json.loads(line).get("abstract") or "")

    found = shared_sentences(texts())
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(sorted(found), ensure_ascii=False), encoding="utf-8")
    return frozenset(found)
