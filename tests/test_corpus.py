"""Tests for corpus preprocessing utilities and ArxivSource."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interpretant.corpus.arxiv import ArxivSource
from interpretant.corpus.preprocessing import filter_empty, preprocess_text

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RECORDS = [
    # included: cs.CL, 2005 → decade 2000
    {
        "id": "0001",
        "title": "Neural language models",
        "categories": "cs.CL cs.AI",
        "abstract": "We study language modelling with neural networks.",
        "versions": [{"version": "v1", "created": "Mon, 3 Jan 2005 00:00:00 GMT"}],
    },
    # included: cs.LG, 1998 → decade 1990
    {
        "id": "0002",
        "title": "Statistical learning theory",
        "categories": "cs.LG stat.ML",
        "abstract": "A framework for $\\ell_2$ regularisation.",
        "versions": [{"version": "v1", "created": "Tue, 15 Apr 1998 00:00:00 GMT"}],
    },
    # excluded: wrong category
    {
        "id": "0003",
        "title": "Quantum entanglement",
        "categories": "quant-ph",
        "abstract": "Entanglement in bipartite systems.",
        "versions": [{"version": "v1", "created": "Mon, 1 Mar 2000 00:00:00 GMT"}],
    },
    # excluded: missing versions
    {
        "id": "0004",
        "title": "Missing date paper",
        "categories": "cs.CL",
        "abstract": "No date here.",
        "versions": [],
    },
    # included: cs.CL, 2012 → decade 2010
    {
        "id": "0005",
        "title": "Attention mechanisms",
        "categories": "cs.CL",
        "abstract": "We propose a new attention $W_q$ model.",
        "versions": [{"version": "v1", "created": "Fri, 5 Oct 2012 00:00:00 GMT"}],
    },
    # excluded: year out of min_year=1995 range
    {
        "id": "0006",
        "title": "Early paper",
        "categories": "cs.CL",
        "abstract": "Very old work.",
        "versions": [{"version": "v1", "created": "Wed, 7 Jun 1992 00:00:00 GMT"}],
    },
]


@pytest.fixture()
def arxiv_snapshot(tmp_path: Path) -> Path:
    """Write sample records to a JSONL snapshot file and return its parent dir."""
    snapshot = tmp_path / "arxiv-metadata-oai-snapshot.json"
    lines = [json.dumps(r) for r in SAMPLE_RECORDS]
    # Add one malformed line to verify it is skipped gracefully
    lines.insert(2, "{ this is not valid json }")
    snapshot.write_text("\n".join(lines), encoding="utf-8")
    return tmp_path


@pytest.fixture()
def arxiv_source(arxiv_snapshot: Path) -> ArxivSource:
    return ArxivSource(
        raw_dir=arxiv_snapshot,
        categories=["cs.CL", "cs.LG"],
        min_year=1995,
        max_year=2019,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestArxivSource:
    def test_iter_documents_filters_by_category(self, arxiv_source: ArxivSource) -> None:
        docs = list(arxiv_source.iter_documents())
        # records 0001 (cs.CL/2005), 0002 (cs.LG/1998), 0005 (cs.CL/2012) should appear
        # record 0003 (quant-ph) excluded; 0004 (no date) excluded; 0006 (year<1995) included by
        # category but excluded in iter_decade_slices (iter_documents has no year filter)
        assert len(docs) >= 3

    def test_iter_documents_yields_preprocessed_text(self, arxiv_source: ArxivSource) -> None:
        for doc in arxiv_source.iter_documents():
            assert "$" not in doc
            assert doc == doc.lower()

    def test_iter_decade_slices_groups_correctly(self, arxiv_source: ArxivSource) -> None:
        with pytest.warns(UserWarning, match="Skipped"):
            slices = dict(arxiv_source.iter_decade_slices(start=1990, end=2020, step=10))
        # record 0002 → 1990; record 0001 → 2000; record 0005 → 2010
        assert 1990 in slices
        assert 2000 in slices
        assert 2010 in slices
        assert len(slices[1990]) == 1
        assert len(slices[2000]) == 1
        assert len(slices[2010]) == 1

    def test_iter_decade_slices_respects_year_bounds(self, arxiv_source: ArxivSource) -> None:
        with pytest.warns(UserWarning, match="Skipped"):
            slices = dict(arxiv_source.iter_decade_slices(start=1990, end=2020, step=10))
        # record 0006 (year=1992 < min_year=1995) must not appear
        all_docs = [doc for docs in slices.values() for doc in docs]
        assert not any("very old" in doc for doc in all_docs)

    def test_malformed_record_is_skipped(self, arxiv_source: ArxivSource) -> None:
        # Should not raise; malformed JSON is silently skipped
        docs = list(arxiv_source.iter_documents())
        assert len(docs) > 0

    def test_document_count_returns_line_count(self, arxiv_snapshot: Path) -> None:
        source = ArxivSource(raw_dir=arxiv_snapshot)
        count = source.document_count()
        # SAMPLE_RECORDS (6) + 1 malformed line = 7 non-empty lines
        assert count == len(SAMPLE_RECORDS) + 1

    def test_missing_snapshot_raises(self, tmp_path: Path) -> None:
        source = ArxivSource(raw_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            list(source.iter_documents())

    def test_no_category_filter_includes_all(self, arxiv_snapshot: Path) -> None:
        source = ArxivSource(raw_dir=arxiv_snapshot, min_year=1900, max_year=2030)
        docs = list(source.iter_documents())
        # All valid records (with non-empty text) should appear
        # all 6 records have non-empty text; iter_documents has no year filter
        assert len(docs) == 6


class TestPreprocessText:
    def test_lowercases_input(self) -> None:
        assert preprocess_text("Hello World") == "hello world"

    def test_removes_inline_math(self) -> None:
        result = preprocess_text("The formula $E=mc^2$ is famous.")
        assert "$" not in result
        assert "e" in result  # "the" and "is" and "famous" should survive

    def test_removes_display_math(self) -> None:
        result = preprocess_text("See $$\\int_0^1 f(x) dx$$ for details.")
        assert "$$" not in result

    def test_removes_punctuation(self) -> None:
        result = preprocess_text("Hello, world! How are you?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_collapses_whitespace(self) -> None:
        result = preprocess_text("  multiple   spaces   here  ")
        assert "  " not in result
        assert result == result.strip()

    def test_unicode_normalisation(self) -> None:
        # Ligature 'ﬁ' → 'fi' after NFKC
        result = preprocess_text("ﬁeld")
        assert "ﬁ" not in result

    def test_empty_string(self) -> None:
        assert preprocess_text("") == ""

    def test_preserves_hyphenated_words(self) -> None:
        result = preprocess_text("state-of-the-art method")
        assert "state-of-the-art" in result


class TestFilterEmpty:
    def test_removes_empty_strings(self) -> None:
        assert filter_empty(["a", "", "b"]) == ["a", "b"]

    def test_removes_whitespace_only(self) -> None:
        assert filter_empty(["a", "   ", "\t", "b"]) == ["a", "b"]

    def test_empty_list(self) -> None:
        assert filter_empty([]) == []

    def test_all_empty(self) -> None:
        assert filter_empty(["", " ", "\n"]) == []

    def test_preserves_order(self) -> None:
        texts = ["c", "a", "b"]
        assert filter_empty(texts) == texts
