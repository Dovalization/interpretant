"""arXiv corpus source — reads the Kaggle metadata JSONL snapshot."""

from __future__ import annotations

import json
from collections.abc import Iterator
from email.utils import parsedate
from pathlib import Path

from tqdm import tqdm

from interpretant.corpus.base import CorpusSource
from interpretant.corpus.preprocessing import is_long_enough, preprocess_text


class ArxivSource(CorpusSource):
    """Loads documents from the arXiv bulk metadata JSONL snapshot.

    The snapshot is available from Kaggle:
    https://www.kaggle.com/datasets/Cornell-University/arxiv

    Place ``arxiv-metadata-oai-snapshot.json`` in ``raw_dir`` before use.
    Each line is one JSON record with fields: ``id``, ``title``,
    ``categories``, ``abstract``, ``versions``.
    """

    def __init__(
        self,
        raw_dir: Path,
        categories: list[str] | None = None,
        min_year: int = 1991,
        max_year: int = 2030,
        snapshot_filename: str = "arxiv-metadata-oai-snapshot.json",
    ) -> None:
        self.raw_dir = raw_dir
        self._category_set: set[str] | None = set(categories) if categories else None
        self.min_year = min_year
        self.max_year = max_year
        self.snapshot_filename = snapshot_filename

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _snapshot_path(self) -> Path:
        path = self.raw_dir / self.snapshot_filename
        if not path.exists():
            raise FileNotFoundError(
                f"arXiv snapshot not found at {path}. "
                "Download it from https://www.kaggle.com/datasets/Cornell-University/arxiv "
                f"and place it in {self.raw_dir}."
            )
        return path

    def _parse_year(self, record: dict[str, object]) -> int | None:
        """Extract the publication year from a record's first version date."""
        versions = record.get("versions")
        if not versions or not isinstance(versions, list):
            return None
        first = versions[0]
        if not isinstance(first, dict):
            return None
        created = first.get("created", "")
        if not isinstance(created, str):
            return None
        parsed = parsedate(created)
        if parsed is None:
            return None
        return parsed[0]

    def _matches_categories(self, record: dict[str, object]) -> bool:
        """Return True if the record's categories overlap with the configured filter."""
        if self._category_set is None:
            return True
        raw_cats = record.get("categories", "")
        if not isinstance(raw_cats, str) or not raw_cats.strip():
            return False
        paper_cats = set(raw_cats.split())
        return bool(paper_cats & self._category_set)

    def _record_to_text(self, record: dict[str, object]) -> str:
        """Concatenate title + abstract and apply preprocessing."""
        title = record.get("title", "") or ""
        abstract = record.get("abstract", "") or ""
        if not isinstance(title, str):
            title = ""
        if not isinstance(abstract, str):
            abstract = ""
        raw = f"{title} {abstract}"
        return preprocess_text(raw)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def iter_documents(self) -> Iterator[str]:
        """Yield preprocessed documents for all records matching the category filter."""
        snapshot = self._snapshot_path()
        with snapshot.open(encoding="utf-8") as file_handle:
            for line in file_handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record: dict[str, object] = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not self._matches_categories(record):
                    continue
                text = self._record_to_text(record)
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
        """Yield (decade_start, documents) tuples, one per decade in [start, end).

        Performs a single streaming pass over the snapshot, accumulating
        documents into decade buckets, then yields them in sorted order.

        Args:
            start: First decade to include (e.g. 1990).
            end: Exclusive upper bound decade (e.g. 2020 excludes the 2020s).
            step: Decade step size, typically 10.
            min_tokens: Skip documents with fewer tokens (0 = no filter).
            workers: Accepted but ignored; arXiv is single-file sequential.
        """
        snapshot = self._snapshot_path()
        decades = list(range(start, end, step))
        buckets: dict[int, list[str]] = {decade: [] for decade in decades}
        skipped = 0

        with snapshot.open(encoding="utf-8") as file_handle:
            for line in tqdm(file_handle, desc="Reading arXiv snapshot", unit=" records"):
                line = line.strip()
                if not line:
                    continue
                try:
                    record: dict[str, object] = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                if not self._matches_categories(record):
                    continue

                year = self._parse_year(record)
                if year is None or year < self.min_year or year > self.max_year:
                    skipped += 1
                    continue

                decade = (year // step) * step
                if decade not in buckets:
                    continue

                text = self._record_to_text(record)
                if not text:
                    continue
                if min_tokens > 0 and not is_long_enough(text, min_tokens):
                    skipped += 1
                    continue
                buckets[decade].append(text)

        if skipped:
            import warnings

            warnings.warn(
                f"Skipped {skipped} records (malformed JSON, missing date, or out of year range).",
                stacklevel=2,
            )

        for decade in sorted(buckets):
            yield decade, buckets[decade]

    def document_count(self) -> int:
        """Return total line count of the snapshot (O(N) line scan, no JSON parsing).

        This is an upper bound; malformed or filtered records will not appear
        in ``iter_documents`` but are counted here.
        """
        snapshot = self._snapshot_path()
        count = 0
        with snapshot.open(encoding="utf-8") as file_handle:
            for line in file_handle:
                if line.strip():
                    count += 1
        return count
