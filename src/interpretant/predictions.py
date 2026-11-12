"""Helpers shared by the prediction scorers in ``scripts/``."""

from __future__ import annotations


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
