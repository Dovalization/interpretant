"""How a pipeline reports its progress while it runs, so long commands never go silent.

A pipeline names each stage as it starts, with the number of items when it is known, and
advances as items finish; the command line draws it (``cli.progress``). Library callers and tests
pass nothing and get the silent reporter.
"""

from __future__ import annotations

import time
from collections.abc import Callable
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


class StageTimer:
    """Passes a pipeline's progress on unchanged and times each stage, for the run log."""

    def __init__(self, progress: Progress, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.progress = progress
        self.clock = clock
        self.durations: list[tuple[str, float]] = []
        self._current: tuple[str, float] | None = None

    def stage(self, name: str, total: int | None = None) -> None:
        """Close the running stage's clock, start the next, and pass the stage on."""
        self._close()
        self._current = (name, self.clock())
        self.progress.stage(name, total)

    def advance(self, amount: int = 1) -> None:
        """Pass the advance on."""
        self.progress.advance(amount)

    def finish(self) -> list[tuple[str, float]]:
        """Close the last stage's clock and return every stage with its seconds."""
        self._close()
        return list(self.durations)

    def _close(self) -> None:
        if self._current is None:
            return
        name, started = self._current
        self.durations.append((name, self.clock() - started))
        self._current = None
