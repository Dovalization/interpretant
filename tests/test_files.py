"""Tests for finding and rewriting a pipeline stage's cell files."""

from __future__ import annotations

from pathlib import Path

from interpretant.corpus.files import rewrite_texts, source_texts


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


def test_rewrite_texts_mirrors_the_layout_and_transforms_every_line(tmp_path: Path) -> None:
    stage = tmp_path / "processed"
    write(stage / "openalex" / "texts" / "hci_1990.txt", "user interfaces\nmental models\n")

    def shout(line: str) -> str:
        return " ".join(line.split()).upper()

    written = rewrite_texts(source_texts(stage, ("openalex",)), stage, tmp_path / "next", shout)

    assert written == [tmp_path / "next" / "openalex" / "texts" / "hci_1990.txt"]
    assert written[0].read_text(encoding="utf-8") == "USER INTERFACES\nMENTAL MODELS\n"
