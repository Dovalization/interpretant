"""Corpus audit: per-cell volume, target-word frequency and books-manifest risks.

A cell is one ``(source, decade)`` processed text file. Drift between two decades is
reliable only when a word occurs often enough in both: a vector trained on a handful of
occurrences stays close to its initialisation, and its nearest neighbours are noise.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from interpretant.corpus.books import BooksSource

_CELL_FILENAME = re.compile(r"^(?P<source>[a-z0-9]+)_(?P<decade>\d{4})\.txt$")
# "Zimmern translation", "English translation (Baskin, 1959)"; not "inscription, translation".
_TRANSLATION_NOTE = re.compile(r"\w+ translation\b", re.IGNORECASE)


@dataclass
class CellStats:
    """Volume and target-word counts for one (source, decade) cell."""

    source: str
    decade: int
    documents: int = 0
    tokens: int = 0
    target_counts: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class BookFlag:
    """A manifest entry whose dating misrepresents the English text it contributes."""

    author: str
    title: str
    decade: str
    year: int | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class AuthorConcentration:
    """The single largest author's share of a decade's book words."""

    decade: str
    books: int
    words: int
    top_author: str
    top_share: float


def load_target_words(path: Path) -> list[str]:
    """Read a words-of-interest file: one word per line, ``#`` starts a comment."""
    words: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        word = line.split("#", 1)[0].strip()
        if word:
            words.append(word)
    return words


def count_cell(path: Path, source: str, decade: int, targets: set[str]) -> CellStats:
    """Stream a processed text file (one document per line) and count its contents."""
    stats = CellStats(source=source, decade=decade)
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            tokens = line.split()
            if not tokens:
                continue
            stats.documents += 1
            stats.tokens += len(tokens)
            stats.target_counts.update(token for token in tokens if token in targets)
    return stats


def scan_processed(processed_dir: Path, targets: set[str]) -> list[CellStats]:
    """Count every ``<source>/texts/<source>_<decade>.txt`` file under ``processed_dir``."""
    cells: list[CellStats] = []
    for path in sorted(processed_dir.glob("*/texts/*.txt")):
        match = _CELL_FILENAME.match(path.name)
        if match is None:
            continue
        cells.append(count_cell(path, match["source"], int(match["decade"]), targets))
    return cells


def merge_by_decade(cells: Iterable[CellStats]) -> dict[int, CellStats]:
    """Sum cells across sources, giving one merged cell per decade."""
    merged: dict[int, CellStats] = {}
    for cell in cells:
        total = merged.setdefault(cell.decade, CellStats(source="merged", decade=cell.decade))
        total.documents += cell.documents
        total.tokens += cell.tokens
        total.target_counts.update(cell.target_counts)
    return dict(sorted(merged.items()))


def reliable_transitions(
    cells_by_decade: dict[int, CellStats], word: str, min_occurrences: int
) -> list[tuple[int, int]]:
    """Return consecutive decade pairs where ``word`` reaches ``min_occurrences`` in both."""
    decades = sorted(cells_by_decade)
    return [
        (earlier, later)
        for earlier, later in zip(decades, decades[1:], strict=False)
        if cells_by_decade[earlier].target_counts[word] >= min_occurrences
        and cells_by_decade[later].target_counts[word] >= min_occurrences
    ]


def format_decade_ranges(decades: Iterable[int], step: int = 10) -> str:
    """Compress decades into ranges, e.g. ``[1960, 1970, 1980, 2000]`` → ``1960–1980, 2000``."""
    ranges: list[list[int]] = []
    for decade in sorted(decades):
        if ranges and decade - ranges[-1][-1] == step:
            ranges[-1].append(decade)
        else:
            ranges.append([decade])
    return ", ".join(str(run[0]) if len(run) == 1 else f"{run[0]}–{run[-1]}" for run in ranges)


def _decade_start(label: str) -> int | None:
    try:
        return int(label.rstrip("s"))
    except ValueError:
        return None


def flag_manifest_entries(entries: Iterable[dict[str, object]]) -> list[BookFlag]:
    """Flag books filed outside their publication decade or dated by an untranslated original.

    A translation contributes the translator's English, so dating it by the original's
    year places that vocabulary in the wrong decade. An entry with a ``translator`` field
    is already dated by its English edition.
    """
    flags: list[BookFlag] = []
    for entry in entries:
        decade_label = str(entry.get("decade", ""))
        raw_year = entry.get("year")
        year = raw_year if isinstance(raw_year, int) else None
        reasons: list[str] = []

        decade = _decade_start(decade_label)
        if decade is None:
            continue
        if year is not None and year // 10 * 10 != decade:
            reasons.append(f"filed in {decade_label} but dated {year}")
        if "translator" not in entry and _TRANSLATION_NOTE.search(str(entry.get("notes", ""))):
            reasons.append("translation dated by the original, not the English edition")

        if reasons:
            flags.append(
                BookFlag(
                    author=str(entry.get("author", "")),
                    title=str(entry.get("title", "")),
                    decade=decade_label,
                    year=year,
                    reasons=tuple(reasons),
                )
            )
    return flags


def author_concentration(
    book_words: Iterable[tuple[str, str, int]],
) -> list[AuthorConcentration]:
    """Summarise each decade's largest author share from ``(decade, author, words)`` rows."""
    words_by_decade: dict[str, Counter[str]] = {}
    books_by_decade: Counter[str] = Counter()
    for decade, author, words in book_words:
        words_by_decade.setdefault(decade, Counter())[author] += words
        books_by_decade[decade] += 1

    summaries: list[AuthorConcentration] = []
    for decade, author_words in sorted(words_by_decade.items()):
        total = sum(author_words.values())
        top_author, top_words = author_words.most_common(1)[0]
        summaries.append(
            AuthorConcentration(
                decade=decade,
                books=books_by_decade[decade],
                words=total,
                top_author=top_author,
                top_share=top_words / total if total else 0.0,
            )
        )
    return summaries


@dataclass
class CorpusAudit:
    """Everything ``corpus audit`` reports, built by :func:`audit_corpus`."""

    min_occurrences: int
    targets: list[str]
    cells: list[CellStats]
    merged: dict[int, CellStats]
    reliable_transitions: dict[str, list[tuple[int, int]]]
    author_concentration: list[AuthorConcentration]
    book_flags: list[BookFlag]
    missing_books: list[Path]

    def to_json(self) -> dict[str, object]:
        """The audit as JSON-ready data (the file ``figures`` and ``export web`` read)."""
        return {
            "min_occurrences": self.min_occurrences,
            "cells": [
                {**asdict(cell), "target_counts": dict(cell.target_counts)} for cell in self.cells
            ],
            "reliable_transitions": self.reliable_transitions,
            "author_concentration": [asdict(summary) for summary in self.author_concentration],
            "book_flags": [asdict(flag) for flag in self.book_flags],
            "missing_books": [str(path) for path in self.missing_books],
        }


def audit_corpus(
    processed_dir: Path, targets: list[str], books: BooksSource, min_occurrences: int
) -> CorpusAudit:
    """Count every cell under ``processed_dir`` and check the books manifest."""
    cells = scan_processed(processed_dir, set(targets))
    merged = merge_by_decade(cells)
    present, missing = books.validate()
    book_words = [
        (
            str(entry.get("decade", "")),
            str(entry.get("author", "")),
            len(books.entry_path(entry).read_text(encoding="utf-8", errors="replace").split()),
        )
        for entry in present
        if str(entry.get("decade", "")).endswith("0s")
    ]
    return CorpusAudit(
        min_occurrences=min_occurrences,
        targets=targets,
        cells=cells,
        merged=merged,
        reliable_transitions={
            word: reliable_transitions(merged, word, min_occurrences) for word in targets
        },
        author_concentration=author_concentration(book_words),
        book_flags=sorted(flag_manifest_entries(books.manifest()), key=lambda flag: flag.decade),
        missing_books=[books.entry_path(entry) for entry in missing],
    )
