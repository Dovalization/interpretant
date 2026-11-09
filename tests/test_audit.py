"""Tests for the corpus audit helpers."""

from __future__ import annotations

from pathlib import Path

from interpretant.corpus.audit import (
    author_concentration,
    flag_manifest_entries,
    format_decade_ranges,
    load_target_words,
    merge_by_decade,
    reliable_transitions,
    scan_processed,
)

TARGETS = {"network", "affordance"}


def _write_cell(processed_dir: Path, source: str, decade: int, lines: list[str]) -> None:
    texts_dir = processed_dir / source / "texts"
    texts_dir.mkdir(parents=True, exist_ok=True)
    (texts_dir / f"{source}_{decade}.txt").write_text("\n".join(lines), encoding="utf-8")


def test_load_target_words_skips_comments_and_blanks(tmp_path: Path) -> None:
    words_file = tmp_path / "words.txt"
    words_file.write_text("# header\n\nnetwork\naffordance  # Gibson\n", encoding="utf-8")

    assert load_target_words(words_file) == ["network", "affordance"]


def test_scan_processed_counts_documents_tokens_and_targets(tmp_path: Path) -> None:
    _write_cell(tmp_path, "acl", 1990, ["neural network model", "", "network of networks"])

    [cell] = scan_processed(tmp_path, TARGETS)

    assert (cell.source, cell.decade) == ("acl", 1990)
    assert cell.documents == 2
    assert cell.tokens == 6
    assert cell.target_counts["network"] == 2
    assert cell.target_counts["affordance"] == 0


def test_scan_processed_ignores_files_outside_the_naming_scheme(tmp_path: Path) -> None:
    _write_cell(tmp_path, "acl", 1990, ["network"])
    (tmp_path / "acl" / "texts" / "notes.txt").write_text("network", encoding="utf-8")

    assert len(scan_processed(tmp_path, TARGETS)) == 1


def test_merge_by_decade_sums_sources(tmp_path: Path) -> None:
    _write_cell(tmp_path, "acl", 1990, ["network network"])
    _write_cell(tmp_path, "books", 1990, ["network affordance"])
    _write_cell(tmp_path, "books", 2000, ["affordance"])

    merged = merge_by_decade(scan_processed(tmp_path, TARGETS))

    assert list(merged) == [1990, 2000]
    assert merged[1990].tokens == 4
    assert merged[1990].target_counts["network"] == 3


def test_reliable_transitions_require_threshold_in_both_decades(tmp_path: Path) -> None:
    _write_cell(tmp_path, "books", 1950, ["network " * 2])
    _write_cell(tmp_path, "books", 1960, ["network " * 5])
    _write_cell(tmp_path, "books", 1970, ["network " * 6])
    merged = merge_by_decade(scan_processed(tmp_path, TARGETS))

    assert reliable_transitions(merged, "network", min_occurrences=5) == [(1960, 1970)]
    assert reliable_transitions(merged, "affordance", min_occurrences=1) == []


def test_format_decade_ranges_compresses_consecutive_runs() -> None:
    assert format_decade_ranges([2000, 1960, 1970, 1980]) == "1960–1980, 2000"
    assert format_decade_ranges([]) == ""


def test_flag_manifest_entries_catches_misfiled_and_translated_books() -> None:
    entries: list[dict[str, object]] = [
        {"author": "Kant", "title": "Critique", "year": 1781, "decade": "1890s", "notes": ""},
        {
            "author": "Heidegger",
            "title": "Being and Time",
            "year": 1927,
            "decade": "1920s",
            "notes": "Macquarrie & Robinson translation.",
        },
        {"author": "James", "title": "Principles", "year": 1890, "decade": "1890s", "notes": ""},
    ]

    flags = flag_manifest_entries(entries)

    assert [flag.author for flag in flags] == ["Kant", "Heidegger"]
    assert flags[0].reasons == ("filed in 1890s but dated 1781",)
    assert "translation" in flags[1].reasons[0]


def test_flag_manifest_entries_accepts_translations_dated_by_english_edition() -> None:
    entries: list[dict[str, object]] = [
        {
            "author": "Heidegger",
            "title": "Being and Time",
            "year": 1962,
            "original_year": 1927,
            "translator": "John Macquarrie and Edward Robinson",
            "decade": "1960s",
            "notes": "Macquarrie & Robinson translation.",
        },
        {"author": "Coleridge", "title": "Biographia", "year": 1817, "decade": "canon"},
        {
            "author": "Latour",
            "title": "Science in Action",
            "year": 1987,
            "decade": "1980s",
            "notes": "Black box, inscription, translation, obligatory passage points.",
        },
    ]

    assert flag_manifest_entries(entries) == []


def test_author_concentration_reports_top_share_per_decade() -> None:
    rows = [("1890s", "James", 900), ("1890s", "Peirce", 100), ("1900s", "Bergson", 50)]

    summaries = author_concentration(rows)

    assert [summary.decade for summary in summaries] == ["1890s", "1900s"]
    assert summaries[0].top_author == "James"
    assert summaries[0].top_share == 0.9
    assert summaries[0].books == 2
    assert summaries[1].top_share == 1.0
