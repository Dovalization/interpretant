"""How a pipeline reports its progress while it runs, so long commands never go silent.

A pipeline names each stage as it starts, with the number of items when it is known, and
advances as items finish; the command line draws it (``cli.progress``). Library callers and tests
pass nothing and get the silent reporter.
"""

from __future__ import annotations

from typing import Protocol


class Progress(Protocol):
    """Receives a pipeline's stages and its advances within the current stage."""

    def stage(self, name: str, total: int | None = None) -> None:
        """A new stage starts; ``total`` items when known."""

    def advance(self, amount: int = 1) -> None:
        """``amount`` more items of the current stage are done."""


class SilentProgress:
    """A reporter that ignores everything, for callers with nowhere to show progress."""

    def stage(self, name: str, total: int | None = None) -> None:
        """Ignore a stage."""

    def advance(self, amount: int = 1) -> None:
        """Ignore an advance."""


SILENT = SilentProgress()
