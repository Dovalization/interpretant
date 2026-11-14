"""Live progress for the long commands: a bar in a terminal, a line every half minute in a log.

rich draws a live bar only in an interactive terminal; when the output goes to a file (a run left
in the background), it prints the final state alone. So in a file the same figures are printed
as plain lines: when a stage starts, every ``interval`` seconds while it runs, and when it ends.

The rate and the time left come from the items this run did: a resumed stage's skipped items
move the position alone, or the rate would count an earlier run's work as done in no time.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text


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
        self.skipped = 0
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
                _PaceColumn(),
                console=self.console,
                get_time=self.clock,
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
        self.name, self.total, self.completed, self.skipped = name, total, 0, 0
        self.started = self.last_line = self.clock()
        if self.bars is not None:
            self.task = self.bars.add_task(name, total=total, skipped=0)
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

    def skip(self, amount: int) -> None:
        """Count ``amount`` items an earlier run did as done, outside this run's rate."""
        self.completed += amount
        self.skipped += amount
        if self.bars is not None and self.task is not None:
            self.bars.update(self.task, advance=amount, skipped=self.skipped)

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
            earlier = f"{self.skipped:,} of them from an earlier run, " if self.skipped else ""
            self.console.print(f"{self.name}: done ({done}{earlier}{_duration(elapsed)})")
        self.name = ""

    def _figures(self, now: float) -> str:
        remaining = self.total - self.completed if self.total else None
        pace = _pace(self.completed - self.skipped, now - self.started, remaining)
        if not self.total:
            position = f"{self.completed:,} items"
        else:
            share = self.completed / self.total
            position = f"{self.completed:,}/{self.total:,} ({share:.1%})"
        return ", ".join(part for part in (position, pace) if part)


class _PaceColumn(ProgressColumn):
    """This run's rate and the time left at it, from the items not skipped."""

    def render(self, task: Task) -> Text:
        """The task's pace, empty until this run has done an item."""
        remaining = None if task.finished or task.total is None else task.remaining
        done = task.completed - task.fields.get("skipped", 0)
        return Text(_pace(done, task.elapsed or 0.0, remaining), style="progress.remaining")


def _pace(done: float, seconds: float, remaining: float | None) -> str:
    """The rate of ``done`` items in ``seconds`` and, with ``remaining`` items, the time left."""
    if done <= 0:
        return ""
    rate = done / max(seconds, 1e-9)
    if remaining is None:
        return _rate(rate)
    return f"{_rate(rate)}, about {_duration(remaining / rate)} left"


def _rate(per_second: float) -> str:
    """The rate in a unit that keeps a figure: whole items a second, then decimals, then a minute.

    Stages that count parts finish fewer than one a second, which a whole-number rate shows as
    "0/s".
    """
    if per_second >= 10:
        return f"{per_second:,.0f}/s"
    if per_second >= 1:
        return f"{per_second:.1f}/s"
    return f"{per_second * 60:.1f}/min"


def _duration(seconds: float) -> str:
    minutes, rest = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {rest:02d} s"
    return f"{rest} s"
