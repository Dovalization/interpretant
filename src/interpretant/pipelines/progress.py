"""How a pipeline reports its progress while it runs, so long commands never go silent.

A pipeline names each stage as it starts, with the number of items when it is known, and
advances as items finish; a resumed stage skips the items an earlier run did, so they move the
position but not the rate. The command line draws it (``cli.progress``), a Prefect run publishes it
as well (``orchestration``), and ``TeeProgress`` sends one pipeline's progress to both. Library
callers and tests pass nothing and get the silent reporter. The ``format_`` functions word the
figures the same way for every reporter.
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

    def skip(self, amount: int) -> None:
        """``amount`` items of the current stage were done by an earlier run."""


class SilentProgress:
    """A reporter that ignores everything, for callers with nowhere to show progress."""

    def stage(self, name: str, total: int | None = None) -> None:
        """Ignore a stage."""

    def advance(self, amount: int = 1) -> None:
        """Ignore an advance."""

    def skip(self, amount: int) -> None:
        """Ignore skipped items."""


SILENT = SilentProgress()


class TeeProgress:
    """Passes each stage, advance and skip on to every reporter it was given, in order."""

    def __init__(self, *reporters: Progress) -> None:
        self.reporters = reporters

    def stage(self, name: str, total: int | None = None) -> None:
        """Pass the stage on to every reporter."""
        for reporter in self.reporters:
            reporter.stage(name, total)

    def advance(self, amount: int = 1) -> None:
        """Pass the advance on to every reporter."""
        for reporter in self.reporters:
            reporter.advance(amount)

    def skip(self, amount: int) -> None:
        """Pass the skipped items on to every reporter."""
        for reporter in self.reporters:
            reporter.skip(amount)


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

    def skip(self, amount: int) -> None:
        """Pass the skipped items on."""
        self.progress.skip(amount)

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


def format_pace(done: float, seconds: float, remaining: float | None) -> str:
    """The rate of ``done`` items in ``seconds`` and, with ``remaining`` items, the time left."""
    if done <= 0:
        return ""
    rate = done / max(seconds, 1e-9)
    if remaining is None:
        return format_rate(rate)
    return f"{format_rate(rate)}, about {format_duration(remaining / rate)} left"


def format_rate(per_second: float) -> str:
    """The rate in a unit that keeps a figure: whole items a second, then decimals, then a minute.

    Stages that count parts finish fewer than one a second, which a whole-number rate shows as
    "0/s".
    """
    if per_second >= 10:
        return f"{per_second:,.0f}/s"
    if per_second >= 1:
        return f"{per_second:.1f}/s"
    return f"{per_second * 60:.1f}/min"


def format_duration(seconds: float) -> str:
    """Seconds as hours and minutes, minutes and seconds, or seconds alone."""
    minutes, rest = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {rest:02d} s"
    return f"{rest} s"
