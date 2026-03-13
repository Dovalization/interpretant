"""Tests for PDF extraction — postprocessing and quality checks only, no real PDFs."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from interpretant.corpus.pdf_extractor import DoclingBackend, PDFExtractor, PyMuPDFBackend

# ---------------------------------------------------------------------------
# Helpers — bypass __init__ for unit-testing internal methods
# ---------------------------------------------------------------------------


def _bare_extractor() -> PDFExtractor:
    """PDFExtractor instance without invoking backend selection."""
    return PDFExtractor.__new__(PDFExtractor)


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------


def test_postprocess_removes_page_numbers() -> None:
    extractor = _bare_extractor()
    result = extractor._postprocess("Some text\n\n42\n\nMore text")
    lines = [line.strip() for line in result.split("\n") if line.strip()]
    assert "42" not in lines


def test_postprocess_rejoins_hyphenated_words() -> None:
    extractor = _bare_extractor()
    result = extractor._postprocess("phenom-\nenology of perception")
    assert "phenomenology" in result


def test_postprocess_collapses_blank_lines() -> None:
    extractor = _bare_extractor()
    result = extractor._postprocess("line one\n\n\n\n\nline two")
    assert "\n\n\n" not in result


def test_postprocess_normalizes_unicode() -> None:
    extractor = _bare_extractor()
    result = extractor._postprocess("caf\u00e9")
    assert "café" in result


def test_postprocess_removes_soft_hyphens() -> None:
    extractor = _bare_extractor()
    result = extractor._postprocess("con\u00adcept")
    assert "\u00ad" not in result


# ---------------------------------------------------------------------------
# Quality check
# ---------------------------------------------------------------------------


def test_quality_check_poor_on_short_text() -> None:
    extractor = _bare_extractor()
    result = extractor.quality_check("Too short")
    assert result["recommendation"] == "poor"
    assert result["word_count"] == 2


def test_quality_check_ok_on_normal_text() -> None:
    extractor = _bare_extractor()
    text = " ".join(["phenomenology"] * 2000)
    result = extractor.quality_check(text)
    assert result["recommendation"] == "ok"
    assert result["word_count"] == 2000


def test_quality_check_flags_ocr_errors() -> None:
    extractor = _bare_extractor()
    # >30% single-char tokens → suspected OCR errors
    text = " ".join(["a"] * 500 + ["phenomenology"] * 100)
    result = extractor.quality_check(text)
    assert result["suspected_ocr_errors"] is True


def test_quality_check_returns_all_keys() -> None:
    extractor = _bare_extractor()
    result = extractor.quality_check("word " * 1500)
    assert set(result.keys()) == {
        "word_count",
        "avg_line_length",
        "short_token_ratio",
        "suspected_ocr_errors",
        "recommendation",
    }


# ---------------------------------------------------------------------------
# Backend availability
# ---------------------------------------------------------------------------


def test_docling_backend_unavailable_when_not_installed() -> None:
    backend = DoclingBackend()
    with patch.dict("sys.modules", {"docling": None}):
        assert backend.is_available() is False


def test_pymupdf_backend_unavailable_when_not_installed() -> None:
    backend = PyMuPDFBackend()
    with patch.dict("sys.modules", {"fitz": None}):
        assert backend.is_available() is False


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_auto_selects_docling_when_available() -> None:
    with patch.object(DoclingBackend, "is_available", return_value=True):
        extractor = PDFExtractor(backend="auto")
        assert isinstance(extractor._backend, DoclingBackend)


def test_auto_falls_back_to_pymupdf() -> None:
    with (
        patch.object(DoclingBackend, "is_available", return_value=False),
        patch.object(PyMuPDFBackend, "is_available", return_value=True),
    ):
        extractor = PDFExtractor(backend="auto")
        assert isinstance(extractor._backend, PyMuPDFBackend)


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        PDFExtractor(backend="grobid")


def test_backend_name_property() -> None:
    with patch.object(DoclingBackend, "is_available", return_value=True):
        extractor = PDFExtractor(backend="auto")
        assert extractor.backend_name == "docling"


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def test_save_creates_parent_dirs(tmp_path: pytest.TempPathFactory) -> None:
    extractor = _bare_extractor()
    output = tmp_path / "books" / "1960s" / "test.txt"  # type: ignore[operator]
    extractor.save("some extracted text", output)
    assert output.exists()
    assert output.read_text() == "some extracted text"


def test_save_overwrites_existing(tmp_path: pytest.TempPathFactory) -> None:
    extractor = _bare_extractor()
    output = tmp_path / "out.txt"  # type: ignore[operator]
    extractor.save("first", output)
    extractor.save("second", output)
    assert output.read_text() == "second"
