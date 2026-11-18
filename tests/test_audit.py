"""Tests for reading words-of-interest files."""

from __future__ import annotations

from pathlib import Path

from interpretant.corpus.audit import load_target_words


def test_load_target_words_skips_comments_and_blanks(tmp_path: Path) -> None:
    words_file = tmp_path / "words.txt"
    words_file.write_text("# header\n\nnetwork\naffordance  # Gibson\n", encoding="utf-8")

    assert load_target_words(words_file) == ["network", "affordance"]
