"""ACL Anthology corpus source — reads the bulk BibTeX download with abstracts."""

from __future__ import annotations

import gzip
import re
from collections.abc import Iterator
from pathlib import Path

import bibtexparser
from bibtexparser.bparser import BibTexParser
from tqdm import tqdm

from interpretant.corpus.base import CorpusSource
from interpretant.corpus.preprocessing import is_long_enough, preprocess_text

_BIB_FILENAME = "anthology+abstracts.bib.gz"
_DOWNLOAD_URL = "https://aclanthology.org/anthology+abstracts.bib.gz"

# BibTeX entry types that are venue/volume records, not research documents
_SKIP_ENTRY_TYPES = frozenset({"proceedings", "book", "collection", "inbook"})

# Matches BibTeX accent commands: {\'e}, {\`e}, {\^e}, {\"e}, {\~n}, etc.
# Captures just the base letter so accented chars survive as ASCII.
_ACCENT_RE = re.compile(r"\{\\[^a-zA-Z{}]([a-zA-Z])\}")
# Matches remaining brace groups after accent handling
_BRACE_RE = re.compile(r"\{[^{}]*\}")


class ACLAnthologySource(CorpusSource):
    """Loads documents from the ACL Anthology bulk BibTeX download.

    Download the file with::

        uv run interpretant corpus download acl

    or manually from https://aclanthology.org/anthology+abstracts.bib.gz
    and place it in ``raw_dir``.
    """

    def __init__(
        self,
        raw_dir: Path,
        min_year: int = 1965,
        max_year: int = 2023,
        bib_filename: str = _BIB_FILENAME,
    ) -> None:
        self.raw_dir = raw_dir
        self.min_year = min_year
        self.max_year = max_year
        self.bib_filename = bib_filename

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bib_path(self) -> Path:
        path = self.raw_dir / self.bib_filename
        if not path.exists():
            raise FileNotFoundError(
                f"ACL Anthology bib file not found at {path}. "
                f"Run: uv run interpretant corpus download acl\n"
                f"Or download manually from {_DOWNLOAD_URL}"
            )
        return path

    def _parse_year(self, entry: dict[str, str]) -> int | None:
        raw = entry.get("year", "").strip()
        if not raw:
            return None
        try:
            return int(raw[:4])
        except ValueError:
            return None

    def _strip_bibtex_braces(self, text: str) -> str:
        """Remove BibTeX brace notation, preserving readable text.

        Handles accent commands like {\'e} → e, then strips remaining braces.
        """
        # First pass: replace accent commands with their base letter
        text = _ACCENT_RE.sub(lambda m: m.group(1), text)
        # Iteratively unwrap remaining brace groups
        prev = None
        while prev != text:
            prev = text
            text = _BRACE_RE.sub(lambda m: m.group(0)[1:-1], text)
        return text

    def _entry_to_text(self, entry: dict[str, str]) -> str:
        # Skip venue/volume records — they have no research content
        if entry.get("ENTRYTYPE", "").lower() in _SKIP_ENTRY_TYPES:
            return ""
        title = self._strip_bibtex_braces(entry.get("title", "") or "")
        abstract = self._strip_bibtex_braces(entry.get("abstract", "") or "")
        return preprocess_text(f"{title} {abstract}")

    def _iter_entries(self) -> Iterator[dict[str, str]]:
        """Stream parsed BibTeX entries from the (possibly gzipped) file."""
        bib_path = self._bib_path()
        opener = gzip.open if bib_path.suffix == ".gz" else open

        with opener(bib_path, "rb") as raw_file:
            content = raw_file.read().decode("utf-8", errors="replace")

        parser = BibTexParser(common_strings=True)
        parser.ignore_nonstandard_types = False
        parser.homogenize_fields = False
        db = bibtexparser.loads(content, parser=parser)
        yield from db.entries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def iter_documents(self) -> Iterator[str]:
        """Yield preprocessed documents for all entries with at least a title."""
        for entry in self._iter_entries():
            text = self._entry_to_text(entry)
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
        """Yield (decade_start, documents) for each decade in [start, end).

        Includes all entries with at least a title. Entries without abstracts
        contribute title-only text, which is better than nothing for early decades
        where abstracts were not consistently recorded.
        """
        decades = list(range(start, end, step))
        buckets: dict[int, list[str]] = {decade: [] for decade in decades}
        skipped = 0

        entries = list(self._iter_entries())

        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            def _process_entry(entry: dict[str, str]) -> tuple[int | None, str]:
                return self._parse_year(entry), self._entry_to_text(entry)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(
                    tqdm(
                        pool.map(_process_entry, entries),
                        total=len(entries),
                        desc="Processing ACL Anthology",
                        unit=" entries",
                    )
                )
        else:
            results = []
            for entry in tqdm(entries, desc="Processing ACL Anthology", unit=" entries"):
                results.append((self._parse_year(entry), self._entry_to_text(entry)))

        for year, text in results:
            if year is None or year < self.min_year or year > self.max_year:
                skipped += 1
                continue

            decade = (year // step) * step
            if decade not in buckets:
                continue

            if not text:
                continue
            if min_tokens > 0 and not is_long_enough(text, min_tokens):
                skipped += 1
                continue
            buckets[decade].append(text)

        if skipped:
            import warnings
            warnings.warn(
                f"Skipped {skipped} ACL entries "
                "(missing/unparseable year, out of range, or below min_tokens).",
                stacklevel=2,
            )

        for decade in sorted(buckets):
            yield decade, buckets[decade]

    def document_count(self) -> int:
        """Return total number of entries in the bib file (O(N) parse)."""
        return sum(1 for _ in self._iter_entries())
