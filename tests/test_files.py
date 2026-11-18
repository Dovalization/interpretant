"""Tests for finding a pipeline stage's cell files."""

from __future__ import annotations

from pathlib import Path

from interpretant.corpus.files import source_texts


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_source_texts_lists_only_the_requested_sources(tmp_path: Path) -> None:
    write(tmp_path / "openalex" / "texts" / "hci_1990.txt", "a")
    write(tmp_path / "acl" / "texts" / "acl_1990.txt", "b")
    write(tmp_path / "openalex_full" / "texts" / "hci_1990.txt", "c")

    paths = source_texts(tmp_path, ("openalex_full", "openalex"))

    assert [path.relative_to(tmp_path).as_posix() for path in paths] == [
        "openalex/texts/hci_1990.txt",
        "openalex_full/texts/hci_1990.txt",
    ]
