"""Live progress for the long commands: a bar in a terminal, a line every half minute in a log.

rich draws a live bar only in an interactive terminal; when the output goes to a file (a run left
in the background), it prints the final state alone. So in a file the same figures are printed
as plain lines: when a stage starts, every ``interval`` seconds while it runs, and when it ends.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class LiveProgress:
    """A pipeline's progress reporter for the command line, used as a context manager."""

    def __init__(
        self,
        console: Console,
        *,
        interval: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.console = console
        self.interval = interval
        self.clock = clock
        self.bars: Progress | None = None
        self.task: TaskID | None = None
        self.name = ""
        self.total: int | None = None
        self.completed = 0
        self.started = 0.0
        self.last_line = 0.0

    def __enter__(self) -> LiveProgress:
        if self.console.is_terminal:
            self.bars = Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
            )
            self.bars.start()
        return self

    def __exit__(self, *exception: object) -> None:
        """Close the last stage and stop the bars, whether or not the command failed."""
        self._finish_stage()
        if self.bars is not None:
            self.bars.stop()

    def stage(self, name: str, total: int | None = None) -> None:
        """Close the current stage and start ``name``, with ``total`` items when known."""
        self._finish_stage()
        self.name, self.total, self.completed = name, total, 0
        self.started = self.last_line = self.clock()
        if self.bars is not None:
            self.task = self.bars.add_task(name, total=total)
            return
        items = f" ({total:,} items)" if total is not None else ""
        self.console.print(f"{name}: started{items}")

    def advance(self, amount: int = 1) -> None:
        """Count ``amount`` more items of the current stage as done."""
        self.completed += amount
        if self.bars is not None and self.task is not None:
            self.bars.advance(self.task, amount)
            return
        now = self.clock()
        if now - self.last_line >= self.interval:
            self.last_line = now
            self.console.print(f"{self.name}: {self._figures(now)}")

    def _finish_stage(self) -> None:
        if not self.name:
            return
        if self.bars is not None and self.task is not None:
            self.bars.update(
                self.task, total=self.total or self.completed, completed=self.completed
            )
        else:
            elapsed = self.clock() - self.started
            done = f"{self.completed:,} items, " if self.completed else ""
            self.console.print(f"{self.name}: done ({done}{_duration(elapsed)})")
        self.name = ""

    def _figures(self, now: float) -> str:
        elapsed = max(now - self.started, 1e-9)
        rate = self.completed / elapsed
        if not self.total:
            return f"{self.completed:,} items, {rate:,.0f}/s"
        share = self.completed / self.total
        left = (self.total - self.completed) / rate if rate else 0.0
        return (
            f"{self.completed:,}/{self.total:,} ({share:.1%}), {rate:,.0f}/s, "
            f"about {_duration(left)} left"
        )


def _duration(seconds: float) -> str:
    minutes, rest = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {rest:02d} s"
    return f"{rest} s"
