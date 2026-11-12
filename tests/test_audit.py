"""Tests for the corpus audit helpers."""

from __future__ import annotations

from pathlib import Path

from interpretant.corpus.audit import (
    audit_corpus,
    format_decade_ranges,
    load_target_words,
    merge_by_decade,
    reliable_transitions,
    scan_processed,
)

TARGETS = {"network", "affordance"}


def _write_cell(
    processed_dir: Path, source: str, decade: int, lines: list[str], layer: str = ""
) -> None:
    texts_dir = processed_dir / (layer or source) / "texts"
    texts_dir.mkdir(parents=True, exist_ok=True)
    (texts_dir / f"{source}_{decade}.txt").write_text("\n".join(lines), encoding="utf-8")


def test_load_target_words_skips_comments_and_blanks(tmp_path: Path) -> None:
    words_file = tmp_path / "words.txt"
    words_file.write_text("# header\n\nnetwork\naffordance  # Gibson\n", encoding="utf-8")

    assert load_target_words(words_file) == ["network", "affordance"]


def test_scan_processed_counts_documents_tokens_and_targets(tmp_path: Path) -> None:
    _write_cell(tmp_path, "acl", 1990, ["neural network model", "", "network of networks"])

    [cell] = scan_processed(tmp_path, TARGETS)

    assert (cell.layer, cell.source, cell.decade) == ("acl", "acl", 1990)
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
    _write_cell(tmp_path, "hci", 1990, ["network affordance"])
    _write_cell(tmp_path, "hci", 2000, ["affordance"])

    merged = merge_by_decade(scan_processed(tmp_path, TARGETS))

    assert list(merged) == [1990, 2000]
    assert merged[1990].tokens == 4
    assert merged[1990].target_counts["network"] == 3


def test_reliable_transitions_require_threshold_in_both_decades(tmp_path: Path) -> None:
    _write_cell(tmp_path, "hci", 1950, ["network " * 2])
    _write_cell(tmp_path, "hci", 1960, ["network " * 5])
    _write_cell(tmp_path, "hci", 1970, ["network " * 6])
    merged = merge_by_decade(scan_processed(tmp_path, TARGETS))

    assert reliable_transitions(merged, "network", min_occurrences=5) == [(1960, 1970)]
    assert reliable_transitions(merged, "affordance", min_occurrences=1) == []


def test_format_decade_ranges_compresses_consecutive_runs() -> None:
    assert format_decade_ranges([2000, 1960, 1970, 1980]) == "1960–1980, 2000"
    assert format_decade_ranges([]) == ""


def test_audit_corpus_counts_cells_and_reports_json(tmp_path: Path) -> None:
    _write_cell(tmp_path, "hci", 1990, ["network network affordance"])
    _write_cell(tmp_path, "hci", 2000, ["network affordance"])

    audit = audit_corpus(tmp_path, ["network"], 1)
    report = audit.to_json()

    assert audit.reliable_transitions == {"network": [(1990, 2000)]}
    assert audit.merged[1990].target_counts["network"] == 2
    assert report["cells"] == [
        {
            "layer": "hci",
            "source": "hci",
            "decade": 1990,
            "documents": 1,
            "tokens": 3,
            "target_counts": {"network": 2},
        },
        {
            "layer": "hci",
            "source": "hci",
            "decade": 2000,
            "documents": 1,
            "tokens": 2,
            "target_counts": {"network": 1},
        },
    ]


def test_audit_keeps_a_sample_and_its_full_cell_apart_and_counts_the_decade_once(
    tmp_path: Path,
) -> None:
    _write_cell(tmp_path, "hci", 2010, ["network"] * 2, layer="openalex")
    _write_cell(tmp_path, "hci", 2010, ["network"] * 7, layer="openalex_full")

    audit = audit_corpus(tmp_path, ["network"], 1)

    documents = {(cell.layer, cell.source): cell.documents for cell in audit.cells}
    assert documents == {("openalex", "hci"): 2, ("openalex_full", "hci"): 7}
    assert audit.merged[2010].documents == 2
