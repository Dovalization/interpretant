"""Helpers shared by the prediction scorers in ``scripts/score_*.py``.

A scorer is frozen once its verdicts are recorded, so a change here must leave every
scorer's output byte-identical.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import cache
from pathlib import Path

from gensim.models import KeyedVectors


def report(name: str, verdict: bool | str | None, detail: str) -> None:
    """Print a prediction's verdict: truthy passed, falsy falsified, None untestable, or a
    verdict word such as ``"partial"``. NumPy booleans count as booleans."""
    if verdict is None:
        word = "untestable"
    elif isinstance(verdict, str):
        word = verdict
    else:
        word = "passed" if verdict else "falsified"
    print(f"\n{name}: {word.upper()}\n  {detail}")


@cache
def load_vectors(root: Path, layer: str, field: str, decade: int) -> KeyedVectors:
    """Load a cell model ``<root>/<layer>/<field>_<decade>.kv`` once per process."""
    return KeyedVectors.load(str(root / layer / f"{field}_{decade}.kv"))


def word_count(vectors: KeyedVectors, word: str) -> int:
    """A word's occurrences in the cell a model was trained on (0 if absent)."""
    return int(vectors.get_vecattr(word, "count")) if word in vectors.key_to_index else 0


def overlap(words: Iterable[str], wordset: set[str]) -> int:
    """How many distinct ``words`` belong to ``wordset``."""
    return len(set(words) & wordset)
