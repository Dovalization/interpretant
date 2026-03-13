"""PDF text extraction for the interpretant books corpus.

Two backends:
- docling (default): ML-based layout understanding, no server required.
  Best for academic PDFs with complex layouts, multi-column text,
  headers/footers, references sections.
- pymupdf: fast, minimal, no ML models. Best for clean born-digital PDFs
  where speed matters more than layout intelligence.

Usage::

    extractor = PDFExtractor(backend="docling")
    text = extractor.extract(Path("path/to/book.pdf"))
    extractor.save(text, Path("data/external/books/1960s/merleau_ponty.txt"))
"""

from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod
from pathlib import Path


class PDFBackend(ABC):
    """Abstract base for PDF extraction backends."""

    @abstractmethod
    def extract(self, pdf_path: Path) -> str:
        """Extract raw text from a PDF file."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is ready to use."""
        ...


class DoclingBackend(PDFBackend):
    """Primary extraction backend using Docling (IBM Research, 2024).

    Docling uses ML models to understand document layout before extracting
    text. This means it correctly identifies and excludes headers, footers,
    page numbers, figure captions, and references sections — returning only
    the body text of the document.

    On first use, Docling downloads its ML models (~500MB) to a local cache.
    Subsequent runs are fully offline.
    """

    # Patterns to strip markdown formatting from Docling's export
    _MD_PATTERNS = [
        (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),   # headings
        (re.compile(r"\*{1,2}(.+?)\*{1,2}"), r"\1"),     # bold/italic
        (re.compile(r"^[-*_]{3,}\s*$", re.MULTILINE), ""),  # dividers
        (re.compile(r"^\|.*\|$", re.MULTILINE), ""),      # table rows
        (re.compile(r"^[-|: ]+$", re.MULTILINE), ""),     # table separators
        (re.compile(r"`{1,3}[^`]*`{1,3}"), ""),           # code spans/blocks
        (re.compile(r"!\[.*?\]\(.*?\)"), ""),              # images
        (re.compile(r"\[(.+?)\]\(.*?\)"), r"\1"),          # links → text
    ]

    def extract(self, pdf_path: Path) -> str:
        """Extract text using Docling's DocumentConverter."""
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        markdown = result.document.export_to_markdown()
        return self._strip_markdown(markdown)

    def is_available(self) -> bool:
        """Check if docling is importable."""
        try:
            import docling  # noqa: F401
            return True
        except ImportError:
            return False

    def _strip_markdown(self, markdown: str) -> str:
        """Convert Docling markdown output to clean plaintext."""
        text = markdown
        for pattern, replacement in self._MD_PATTERNS:
            text = pattern.sub(replacement, text)
        return text


class PyMuPDFBackend(PDFBackend):
    """Fast fallback backend using PyMuPDF (fitz).

    Best for clean born-digital PDFs with simple single-column layouts.
    Does not understand layout — extracts text naively page by page.
    Will include headers, footers, and references mixed with body text.
    """

    def extract(self, pdf_path: Path) -> str:
        """Extract text page-by-page using fitz."""
        import fitz  # type: ignore[import-untyped]

        doc = fitz.open(str(pdf_path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(pages)

    def is_available(self) -> bool:
        try:
            import fitz  # type: ignore[import-untyped]  # noqa: F401
            return True
        except ImportError:
            return False


class PDFExtractor:
    """Main interface for PDF extraction.

    Selects backend, runs extraction, applies postprocessing, and
    provides quality assessment of the extracted text.
    """

    def __init__(self, backend: str = "docling") -> None:
        """Initialise extractor with the chosen backend.

        Args:
            backend: "docling" | "pymupdf" | "auto".
                     "auto" uses docling if available, falls back to pymupdf.
        """
        self._backend = self._select_backend(backend)

    def _select_backend(self, name: str) -> PDFBackend:
        docling = DoclingBackend()
        pymupdf = PyMuPDFBackend()

        if name == "docling":
            if not docling.is_available():
                raise RuntimeError(
                    "Docling is not installed. Run: uv add docling"
                )
            return docling
        if name == "pymupdf":
            if not pymupdf.is_available():
                raise RuntimeError(
                    "PyMuPDF is not installed. Run: uv add pymupdf"
                )
            return pymupdf
        if name == "auto":
            if docling.is_available():
                return docling
            if pymupdf.is_available():
                return pymupdf
            raise RuntimeError(
                "No PDF backend available. Install docling or pymupdf."
            )
        raise ValueError(f"Unknown backend: {name!r}. Use 'docling', 'pymupdf', or 'auto'.")

    @property
    def backend_name(self) -> str:
        return type(self._backend).__name__.replace("Backend", "").lower()

    def extract(self, pdf_path: Path) -> str:
        """Extract and postprocess text from a PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Clean plaintext ready for the corpus pipeline.
        """
        raw = self._backend.extract(pdf_path)
        return self._postprocess(raw)

    def save(self, text: str, output_path: Path) -> None:
        """Write extracted text to output_path, creating parent dirs."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")

    def quality_check(self, text: str) -> dict[str, object]:
        """Heuristic quality assessment of extracted text.

        Returns:
            dict with keys: word_count, avg_line_length, short_token_ratio,
            suspected_ocr_errors, recommendation ("ok" | "try_docling" | "poor").
        """
        words = text.split()
        word_count = len(words)
        lines = [line for line in text.split("\n") if line.strip()]
        avg_line_length = sum(len(line) for line in lines) / max(len(lines), 1)
        short_token_ratio = sum(1 for w in words if len(w) <= 2) / max(word_count, 1)
        suspected_ocr_errors = short_token_ratio > 0.3

        if word_count < 1000:
            recommendation = "poor"
        elif suspected_ocr_errors or avg_line_length < 20:
            recommendation = "try_docling"
        else:
            recommendation = "ok"

        return {
            "word_count": word_count,
            "avg_line_length": round(avg_line_length, 1),
            "short_token_ratio": round(short_token_ratio, 3),
            "suspected_ocr_errors": suspected_ocr_errors,
            "recommendation": recommendation,
        }

    def _postprocess(self, text: str) -> str:
        """Clean raw extracted text.

        Steps (in order):
        1. Unicode NFKC normalisation
        2. Remove soft hyphens
        3. Rejoin hyphenated line breaks ("phenom-\\nenology" → "phenomenology")
        4. Strip lines that are only digits (page numbers)
        5. Collapse 3+ consecutive blank lines to 2
        6. Strip leading/trailing whitespace per line
        7. Final strip
        """
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\u00ad", "")
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines).strip()
