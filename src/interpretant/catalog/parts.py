"""Datasets saved in parts: one file per input cell or download, named safely."""

from __future__ import annotations

import re
from pathlib import Path

PART_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def part_path(folder: Path, part: str, suffix: str) -> Path:
    """The file of one part, such as ``ai_2010``: lowercase letters, digits, ``_``, ``.``, ``-``.

    Raises:
        ValueError: if the part name is not safe as a file name.
    """
    if not PART_NAME.match(part):
        raise ValueError(f"part name {part!r} must match {PART_NAME.pattern}")
    return folder / f"{part}{suffix}"
