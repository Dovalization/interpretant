"""Books corpus source — reads extracted .txt files from data/external/books/.

Each decade's texts live in a subdirectory named after the decade label
(e.g. ``1960s/``, ``1970s/``). The directory structure is populated by
``interpretant pdf extract`` / ``interpretant pdf batch``.

A manifest file (``manifest.json``) tracks metadata for each book.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from interpretant.corpus.caps import cap_author_tokens, largest_author
from interpretant.corpus.preprocessing import preprocess_text


def _decade_label(decade: int) -> str:
    """Convert a decade integer to a directory label, e.g. 1960 → '1960s'."""
    return f"{decade}s"


class BooksSource:
    """Loads documents from book-length PDF extractions.

    Expected layout::

        books_dir/
        ├── 1960s/
        │   ├── merleau_ponty_phenomenology.txt
        │   └── ...
        ├── 1970s/
        │   └── ...
        └── manifest.json   (optional)

    Texts are expected to already be extracted plain text (produced by
    ``interpretant pdf extract``). Each file becomes one document after
    preprocessing.
    """

    def __init__(self, books_dir: Path) -> None:
        self.books_dir = books_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _file_to_text(self, txt_path: Path) -> str:
        """Read a .txt file and return preprocessed text."""
        raw = txt_path.read_text(encoding="utf-8", errors="replace")
        return preprocess_text(raw)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def manifest(self) -> list[dict[str, object]]:
        """Load and return entries from manifest.json, or [] if absent."""
        manifest_path = self.books_dir / "manifest.json"
        if not manifest_path.exists():
            return []
        raw: list[dict[str, object]] = json.loads(manifest_path.read_text(encoding="utf-8"))
        return raw

    def entry_path(self, entry: dict[str, object]) -> Path:
        """Resolve a manifest entry to its text file.

        Most entries name the file with ``filename`` inside the decade directory; a few
        older ones carry a ``file`` path that ends in the same decade directory.
        """
        filename = str(entry.get("filename") or Path(str(entry.get("file", ""))).name)
        return self.books_dir / str(entry.get("decade", "")) / filename

    def decade_documents(self, decade: int) -> list[tuple[str, str]]:
        """Return ``(author, preprocessed text)`` for every present book filed in ``decade``."""
        label = _decade_label(decade)
        documents: list[tuple[str, str]] = []
        for entry in self.manifest():
            path = self.entry_path(entry)
            if str(entry.get("decade", "")) != label or not path.is_file():
                continue
            text = self._file_to_text(path)
            if text:
                documents.append((str(entry.get("author", "")), text))
        return documents

    def validate(self) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """Return (present, missing) based on whether each manifest file exists on disk."""
        present: list[dict[str, object]] = []
        missing: list[dict[str, object]] = []
        for entry in self.manifest():
            (present if self.entry_path(entry).is_file() else missing).append(entry)
        return present, missing

    def stats(self) -> dict[str, object]:
        """Return a summary dict with per-decade book/word counts and per-field breakdown."""
        manifest_entries = self.manifest()
        present, missing = self.validate()

        # Per-decade: count present files and their word counts
        per_decade: dict[str, dict[str, int]] = {}
        for entry in present:
            label = str(entry.get("decade", ""))
            try:
                text = self.entry_path(entry).read_text(encoding="utf-8", errors="replace")
                word_count = len(text.split())
            except OSError:
                word_count = 0
            if label not in per_decade:
                per_decade[label] = {"books": 0, "words": 0}
            per_decade[label]["books"] += 1
            per_decade[label]["words"] += word_count

        # Per-field: count all manifest entries (present + missing)
        per_field: dict[str, int] = {}
        for entry in manifest_entries:
            fields = entry.get("fields", [])
            if isinstance(fields, list):
                for field in fields:
                    field_str = str(field)
                    per_field[field_str] = per_field.get(field_str, 0) + 1

        return {
            "total_books": len(present),
            "per_decade": per_decade,
            "per_field": per_field,
            "manifest_entries": len(manifest_entries),
            "missing": len(missing),
        }


@dataclass(frozen=True)
class IngestedDecade:
    """One decade written by :func:`ingest_books`."""

    decade: int
    books: int
    tokens: int
    largest_author: str


def ingest_books(
    source: BooksSource,
    texts_dir: Path,
    decades: Iterable[int],
    author_cap: int,
    leave_out_largest_author: bool = False,
) -> list[IngestedDecade]:
    """Write ``books_<decade>.txt`` for every decade with books, each author capped at
    ``author_cap`` tokens.

    With ``leave_out_largest_author`` each decade's largest author is dropped first (the
    leave-one-author-out check); ``largest_author`` then names the author left out.
    """
    texts_dir.mkdir(parents=True, exist_ok=True)
    ingested = []
    for decade in decades:
        books = source.decade_documents(decade)
        if not books:
            continue
        top = largest_author(books)
        if leave_out_largest_author:
            books = [(author, text) for author, text in books if author != top]
        documents = cap_author_tokens(books, author_cap)
        (texts_dir / f"books_{decade}.txt").write_text("\n".join(documents), encoding="utf-8")
        tokens = sum(len(document.split()) for document in documents)
        ingested.append(IngestedDecade(decade, len(books), tokens, top))
    return ingested
