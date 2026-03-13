"""Books corpus source — reads extracted .txt files from data/external/books/.

Each decade's texts live in a subdirectory named after the decade label
(e.g. ``1960s/``, ``1970s/``). The directory structure is populated by
``interpretant pdf extract`` / ``interpretant pdf batch``.

A manifest file (``manifest.json``) tracks metadata for each book.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterator
from pathlib import Path

from interpretant.corpus.base import CorpusSource
from interpretant.corpus.preprocessing import is_long_enough, preprocess_text


def _decade_label(decade: int) -> str:
    """Convert a decade integer to a directory label, e.g. 1960 → '1960s'."""
    return f"{decade}s"


def _label_to_decade(label: str) -> int | None:
    """Parse a decade directory label back to int, e.g. '1960s' → 1960."""
    try:
        return int(label.rstrip("s"))
    except ValueError:
        return None


class BooksSource(CorpusSource):
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

    def __init__(
        self,
        books_dir: Path,
        min_year: int = 1900,
        max_year: int = 2030,
    ) -> None:
        self.raw_dir = books_dir
        self.books_dir = books_dir
        self.min_year = min_year
        self.max_year = max_year

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decade_dirs(self) -> dict[int, Path]:
        """Return {decade_int: path} for all decade subdirectories."""
        result: dict[int, Path] = {}
        if not self.books_dir.exists():
            return result
        for child in sorted(self.books_dir.iterdir()):
            if child.is_dir():
                decade = _label_to_decade(child.name)
                if decade is not None:
                    result[decade] = child
        return result

    def _txt_files(self, decade_dir: Path) -> list[Path]:
        return sorted(decade_dir.glob("*.txt"))

    def _file_to_text(self, txt_path: Path) -> str:
        """Read a .txt file and return preprocessed text."""
        raw = txt_path.read_text(encoding="utf-8", errors="replace")
        return preprocess_text(raw)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def iter_documents(self) -> Iterator[str]:
        """Yield preprocessed text for every book across all decades."""
        for _decade, decade_dir in sorted(self._decade_dirs().items()):
            for txt_path in self._txt_files(decade_dir):
                text = self._file_to_text(txt_path)
                if text:
                    yield text

    def iter_decade_slices(
        self,
        start: int,
        end: int,
        step: int,
        min_tokens: int = 0,
        workers: int = 1,
    ) -> Iterator[tuple[int, list[str]]]:
        """Yield (decade_start, documents) for each decade in [start, end)."""
        decades = list(range(start, end, step))
        buckets: dict[int, list[str]] = {decade: [] for decade in decades}
        decade_dirs = self._decade_dirs()
        skipped = 0

        for decade in decades:
            decade_dir = decade_dirs.get(decade)
            if decade_dir is None:
                continue
            if decade < self.min_year or decade > self.max_year:
                skipped += 1
                continue
            for txt_path in self._txt_files(decade_dir):
                text = self._file_to_text(txt_path)
                if not text:
                    continue
                if min_tokens > 0 and not is_long_enough(text, min_tokens):
                    skipped += 1
                    continue
                buckets[decade].append(text)

        if skipped:
            warnings.warn(
                f"Skipped {skipped} books entries (out of range or below min_tokens).",
                stacklevel=2,
            )

        for decade in sorted(buckets):
            yield decade, buckets[decade]

    def document_count(self) -> int:
        """Return total number of book text files across all decades."""
        return sum(
            len(self._txt_files(d)) for d in self._decade_dirs().values()
        )

    def manifest(self) -> list[dict[str, object]]:
        """Load and return entries from manifest.json, or [] if absent."""
        manifest_path = self.books_dir / "manifest.json"
        if not manifest_path.exists():
            return []
        raw: list[dict[str, object]] = json.loads(manifest_path.read_text(encoding="utf-8"))
        return raw

    def validate(self) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """Return (present, missing) based on whether each manifest file exists on disk."""
        present: list[dict[str, object]] = []
        missing: list[dict[str, object]] = []
        for entry in self.manifest():
            filename = str(entry.get("filename", ""))
            decade_label = str(entry.get("decade", ""))
            path = self.books_dir / decade_label / filename
            (present if path.exists() else missing).append(entry)
        return present, missing

    def stats(self) -> dict[str, object]:
        """Return a summary dict with per-decade book/word counts and per-field breakdown."""
        manifest_entries = self.manifest()
        present, missing = self.validate()

        # Per-decade: count present files and their word counts
        per_decade: dict[str, dict[str, int]] = {}
        for entry in present:
            label = str(entry.get("decade", ""))
            filename = str(entry.get("filename", ""))
            path = self.books_dir / label / filename
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
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
