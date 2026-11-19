"""Seal held-out words in the judges' notes of review files before they are committed.

`docs/held_out.md`: development outputs never name the held-out words. A judge's free-text note
may name one, so every ``note`` value in the given JSON files is rewritten with each held-out word
shown by its reported name (`Word.reported_name`). The files keep their layout. Run from the
repository root, after the adjudicator has read the notes:
`python scripts/review/seal_notes.py FILE.json [FILE.json...]`.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from interpretant.catalog.config import read_catalog
from interpretant.catalog.words import Word, load_words, seal_held_out

Layout = Callable[[Any], str]


def _dumps(indent: int | None, *, ensure_ascii: bool) -> Layout:
    return lambda data: json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)


def _one_item_per_line(margin: str, *, ensure_ascii: bool) -> Layout:
    """An object written one top-level entry per line after a margin, each entry compact."""

    def write(data: Any) -> str:  # noqa: ANN401 (any parsed JSON value)
        if not isinstance(data, dict):
            return ""
        entries = (
            f"{margin}{json.dumps(key, ensure_ascii=ensure_ascii)}: "
            f"{json.dumps(value, ensure_ascii=ensure_ascii)}"
            for key, value in data.items()
        )
        return "{\n" + ",\n".join(entries) + "\n}"

    return write


# The ways the review files were written: by json.dumps, or one item per line by the agents.
LAYOUTS: list[Layout] = [
    *(
        _dumps(indent, ensure_ascii=ensure_ascii)
        for indent in (1, 2, 0, None, 4)
        for ensure_ascii in (False, True)
    ),
    *(
        _one_item_per_line(margin, ensure_ascii=ensure_ascii)
        for margin in ("", "  ")
        for ensure_ascii in (False, True)
    ),
]


def sealed(value: Any, words: list[Word]) -> Any:  # noqa: ANN401 (any parsed JSON value)
    """The value with every ``note`` string sealed, at any depth."""
    if isinstance(value, dict):
        return {
            key: seal_held_out(inner, words)
            if key == "note" and isinstance(inner, str)
            else sealed(inner, words)
            for key, inner in value.items()
        }
    if isinstance(value, list):
        return [sealed(inner, words) for inner in value]
    return value


def layout(path: Path, text: str, data: Any) -> Layout:  # noqa: ANN401 (any parsed JSON value)
    """The layout that reproduces the file exactly.

    Raises:
        SystemExit: if no known layout reproduces it, so a rewrite would change more than notes.
    """
    for write in LAYOUTS:
        if write(data) == text.rstrip("\n"):
            return write
    raise SystemExit(f"no known JSON layout reproduces {path}; seal it by hand")


def main() -> None:
    """Seal the notes of every file given and say how many files changed."""
    words = load_words(read_catalog())
    changed = 0
    for name in sys.argv[1:]:
        path = Path(name)
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        result = sealed(data, words)
        if result == data:
            continue
        write = layout(path, text, data)
        ending = "\n" if text.endswith("\n") else ""
        path.write_text(write(result) + ending, encoding="utf-8")
        changed += 1
    sys.stdout.write(f"sealed notes in {changed} of {len(sys.argv) - 1} files\n")


if __name__ == "__main__":
    main()
