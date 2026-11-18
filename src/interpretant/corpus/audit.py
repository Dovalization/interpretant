"""Words-of-interest files: the target and control word lists the commands read."""

from __future__ import annotations

from pathlib import Path


def load_target_words(path: Path) -> list[str]:
    """Read a words-of-interest file: one word per line, ``#`` starts a comment."""
    words: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        word = line.split("#", 1)[0].strip()
        if word:
            words.append(word)
    return words
