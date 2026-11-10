"""Tests for BooksSource."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interpretant.corpus.books import BooksSource

FAKE_TEXT_A = "the quick brown fox jumps over the lazy dog " * 20
FAKE_TEXT_B = "consciousness embodiment perception affordance representation " * 20


@pytest.fixture()
def books_dir(tmp_path: Path) -> Path:
    """Set up a minimal books directory with two present files and one missing."""
    manifest = [
        {
            "author": "Author A",
            "title": "Book A",
            "year": 1965,
            "decade": "1960s",
            "fields": ["philosophy_of_mind", "design"],
            "status": "pending",
            "source": "",
            "source_url": "",
            "filename": "author_a_book_a.txt",
            "notes": "",
        },
        {
            "author": "Author B",
            "title": "Book B",
            "year": 1972,
            "decade": "1970s",
            "fields": ["ai_cs"],
            "status": "pending",
            "source": "",
            "source_url": "",
            "filename": "author_b_book_b.txt",
            "notes": "",
        },
        {
            "author": "Author C",
            "title": "Missing Book",
            "year": 1985,
            "decade": "1980s",
            "fields": ["arts"],
            "status": "pending",
            "source": "",
            "source_url": "",
            "filename": "nonexistent.txt",
            "notes": "",
        },
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    (tmp_path / "1960s").mkdir()
    (tmp_path / "1960s" / "author_a_book_a.txt").write_text(FAKE_TEXT_A, encoding="utf-8")

    (tmp_path / "1970s").mkdir()
    (tmp_path / "1970s" / "author_b_book_b.txt").write_text(FAKE_TEXT_B, encoding="utf-8")

    # 1980s dir exists but the file does not
    (tmp_path / "1980s").mkdir()

    return tmp_path


def test_manifest_loads_required_keys(books_dir: Path) -> None:
    source = BooksSource(books_dir=books_dir)
    entries = source.manifest()
    assert len(entries) == 3
    required = {"author", "title", "year", "decade", "fields", "filename"}
    for entry in entries:
        assert required <= entry.keys()


def test_validate_identifies_missing(books_dir: Path) -> None:
    source = BooksSource(books_dir=books_dir)
    _present, missing = source.validate()
    assert len(missing) == 1
    assert missing[0]["filename"] == "nonexistent.txt"


def test_validate_identifies_present(books_dir: Path) -> None:
    source = BooksSource(books_dir=books_dir)
    present, _missing = source.validate()
    assert len(present) == 2
    filenames = {str(e["filename"]) for e in present}
    assert "author_a_book_a.txt" in filenames
    assert "author_b_book_b.txt" in filenames


def test_entry_path_accepts_legacy_file_key(books_dir: Path) -> None:
    source = BooksSource(books_dir=books_dir)
    entry: dict[str, object] = {
        "decade": "1960s",
        "file": "data/external/books/1960s/author_a_book_a.txt",
    }
    assert source.entry_path(entry).is_file()


def test_validate_treats_entry_without_file_name_as_missing(books_dir: Path) -> None:
    manifest_path = books_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.append({"author": "Author D", "title": "No file", "decade": "1960s"})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _present, missing = BooksSource(books_dir=books_dir).validate()

    assert {str(entry["author"]) for entry in missing} == {"Author C", "Author D"}


def test_stats_returns_counts(books_dir: Path) -> None:
    source = BooksSource(books_dir=books_dir)
    stats = source.stats()
    assert stats["total_books"] == 2
    assert stats["manifest_entries"] == 3
    assert stats["missing"] == 1


def test_stats_per_field(books_dir: Path) -> None:
    source = BooksSource(books_dir=books_dir)
    stats = source.stats()
    per_field = stats["per_field"]
    assert isinstance(per_field, dict)
    # philosophy_of_mind and design come from entry A; ai_cs from B; arts from C
    assert per_field.get("philosophy_of_mind") == 1
    assert per_field.get("ai_cs") == 1
    assert per_field.get("arts") == 1


def test_decade_documents_pairs_each_present_book_with_its_author(books_dir: Path) -> None:
    documents = BooksSource(books_dir=books_dir).decade_documents(1960)

    assert [author for author, _ in documents] == ["Author A"]
    assert documents[0][1].startswith("the quick brown fox")
