"""PubMed corpus source — reads NCBI bulk XML.gz baseline files."""

from __future__ import annotations

import gzip
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

from tqdm import tqdm

from interpretant.corpus.base import CorpusSource
from interpretant.corpus.preprocessing import is_long_enough, preprocess_text

_FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"

# Matches structured abstract labels at the start of a section text,
# e.g. "BACKGROUND: ", "METHODS AND RESULTS: ", "AIM/PURPOSE: "
_SECTION_LABEL_RE = re.compile(r"^[A-Z][A-Z\s/]{2,}:\s*")


def _process_xml_file(
    xml_path: Path,
    min_year: int,
    max_year: int,
    language_filter: frozenset[str],
    min_tokens: int,
    step: int,
) -> tuple[dict[int, list[str]], int]:
    """Process one XML.gz file; return (decade_buckets, skipped_count).

    Module-level so it is picklable by ProcessPoolExecutor.
    """
    import contextlib
    import gzip as _gzip
    import xml.etree.ElementTree as ET  # noqa: N814 (re-import in subprocess scope)

    from interpretant.corpus.preprocessing import is_long_enough as _is_long_enough
    from interpretant.corpus.preprocessing import preprocess_text as _preprocess_text

    buckets: dict[int, list[str]] = {}
    skipped = 0

    opener = _gzip.open if xml_path.suffix == ".gz" else open
    with opener(xml_path, "rb") as fh:
        for _event, elem in ET.iterparse(fh, events=("end",)):
            if elem.tag != "PubmedArticle":
                continue

            # --- year ---
            year_el = elem.find(".//Journal/JournalIssue/PubDate/Year")
            year: int | None = None
            if year_el is not None and year_el.text:
                with contextlib.suppress(ValueError):
                    year = int(year_el.text.strip()[:4])
            if year is None:
                medline_el = elem.find(".//Journal/JournalIssue/PubDate/MedlineDate")
                if medline_el is not None and medline_el.text:
                    with contextlib.suppress(ValueError):
                        year = int(medline_el.text.strip()[:4])
            if year is None or year < min_year or year > max_year:
                skipped += 1
                elem.clear()
                continue

            # --- language ---
            lang_els = elem.findall(".//MedlineCitation/Article/Language")
            lang: str | None = None
            if lang_els:
                raw_lang = lang_els[0].text
                if raw_lang:
                    lang = raw_lang.strip().lower()
            if lang is None or lang not in language_filter:
                skipped += 1
                elem.clear()
                continue

            decade = (year // step) * step

            # --- text ---
            title_el = elem.find(".//ArticleTitle")
            title = title_el.text or "" if title_el is not None else ""

            abstract_parts: list[str] = []
            for ab_el in elem.findall(".//Abstract/AbstractText"):
                raw = ab_el.text or ""
                label = ab_el.get("Label", "")
                if label:
                    prefix = f"{label}:"
                    if raw.upper().startswith(prefix.upper()):
                        raw = raw[len(prefix):].lstrip()
                raw = _SECTION_LABEL_RE.sub("", raw)
                abstract_parts.append(raw)

            text = _preprocess_text(f"{title} {' '.join(abstract_parts)}")

            if not text:
                elem.clear()
                continue
            if min_tokens > 0 and not _is_long_enough(text, min_tokens):
                skipped += 1
                elem.clear()
                continue

            if decade not in buckets:
                buckets[decade] = []
            buckets[decade].append(text)
            elem.clear()

    return buckets, skipped


class PubMedSource(CorpusSource):
    """Loads documents from PubMed NCBI bulk XML.gz baseline files.

    Download with::

        uv run interpretant corpus download pubmed

    Files are saved to ``raw_dir`` as ``pubmedXXnYYYY.xml.gz``.
    """

    def __init__(
        self,
        raw_dir: Path,
        min_year: int = 1960,
        max_year: int = 2023,
        language_filter: list[str] | None = None,
    ) -> None:
        self.raw_dir = raw_dir
        self.min_year = min_year
        self.max_year = max_year
        self.language_filter: frozenset[str] = frozenset(
            language_filter if language_filter is not None else ["eng"]
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _xml_files(self) -> list[Path]:
        """Return sorted list of .xml.gz files in raw_dir."""
        files = sorted(self.raw_dir.glob("pubmed*.xml.gz"))
        if not files:
            raise FileNotFoundError(
                f"No PubMed XML files found in {self.raw_dir}. "
                "Run: uv run interpretant corpus download pubmed"
            )
        return files

    def _parse_year(self, article: ET.Element) -> int | None:
        """Extract publication year from an Article element."""
        # Primary location: Journal/JournalIssue/PubDate/Year
        year_el = article.find(".//Journal/JournalIssue/PubDate/Year")
        if year_el is not None and year_el.text:
            try:
                return int(year_el.text.strip()[:4])
            except ValueError:
                pass
        # Fallback: MedlineDate (e.g. "2001 Jan-Feb") — take first 4 chars
        medline_el = article.find(".//Journal/JournalIssue/PubDate/MedlineDate")
        if medline_el is not None and medline_el.text:
            try:
                return int(medline_el.text.strip()[:4])
            except ValueError:
                pass
        return None

    def _article_language(self, article: ET.Element) -> str | None:
        """Return the first Language element text, lowercased, or None."""
        lang_els = article.findall(".//MedlineCitation/Article/Language")
        if not lang_els:
            return None
        raw = lang_els[0].text
        return raw.strip().lower() if raw else None

    def _article_to_text(self, article: ET.Element) -> str:
        """Extract and preprocess title + abstract from an Article element."""
        title_el = article.find(".//ArticleTitle")
        title = title_el.text or "" if title_el is not None else ""

        abstract_parts: list[str] = []
        for el in article.findall(".//Abstract/AbstractText"):
            raw = el.text or ""
            label = el.get("Label", "")
            if label:
                prefix = f"{label}:"
                if raw.upper().startswith(prefix.upper()):
                    raw = raw[len(prefix):].lstrip()
            raw = _SECTION_LABEL_RE.sub("", raw)
            abstract_parts.append(raw)

        return preprocess_text(f"{title} {' '.join(abstract_parts)}")

    def _iter_articles(self, xml_path: Path) -> Iterator[ET.Element]:
        """Stream PubmedArticle elements from a single XML.gz file."""
        opener = gzip.open if xml_path.suffix == ".gz" else open
        with opener(xml_path, "rb") as fh:
            for _event, elem in ET.iterparse(fh, events=("end",)):
                if elem.tag == "PubmedArticle":
                    yield elem
                    elem.clear()  # free memory after processing

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def iter_documents(self) -> Iterator[str]:
        """Yield preprocessed documents for all articles with a title."""
        for xml_path in self._xml_files():
            for article in self._iter_articles(xml_path):
                lang = self._article_language(article)
                if lang is None or lang not in self.language_filter:
                    continue
                text = self._article_to_text(article)
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

        Streams all XML files, accumulating documents into decade buckets.
        """
        decades = list(range(start, end, step))
        buckets: dict[int, list[str]] = {decade: [] for decade in decades}
        skipped = 0

        xml_files = self._xml_files()

        if workers > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        _process_xml_file,
                        path,
                        self.min_year,
                        self.max_year,
                        self.language_filter,
                        min_tokens,
                        step,
                    )
                    for path in xml_files
                ]
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Reading PubMed files",
                    unit=" files",
                ):
                    partial_buckets, n_skipped = future.result()
                    skipped += n_skipped
                    for decade, docs in partial_buckets.items():
                        if decade in buckets:
                            buckets[decade].extend(docs)
        else:
            for xml_path in tqdm(xml_files, desc="Reading PubMed files", unit=" files"):
                for article in self._iter_articles(xml_path):
                    year = self._parse_year(article)
                    if year is None or year < self.min_year or year > self.max_year:
                        skipped += 1
                        continue

                    lang = self._article_language(article)
                    if lang is None or lang not in self.language_filter:
                        skipped += 1
                        continue

                    decade = (year // step) * step
                    if decade not in buckets:
                        continue

                    text = self._article_to_text(article)
                    if not text:
                        continue
                    if min_tokens > 0 and not is_long_enough(text, min_tokens):
                        skipped += 1
                        continue
                    buckets[decade].append(text)

        if skipped:
            import warnings
            warnings.warn(
                f"Skipped {skipped} PubMed articles "
                "(missing/out-of-range year, language filter, or below min_tokens).",
                stacklevel=2,
            )

        for decade in sorted(buckets):
            yield decade, buckets[decade]

    def document_count(self) -> int:
        """Return total article count across all XML files (full parse, slow)."""
        return sum(1 for _ in self.iter_documents())
