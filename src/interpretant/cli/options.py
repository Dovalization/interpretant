"""Options the corpus commands share."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TypeVar

import click

_Command = TypeVar("_Command", bound=Callable[..., None])


def default_process_count() -> int:
    """The machine's CPU count minus two, leaving room for the main process and the system."""
    return max(1, (os.cpu_count() or 2) - 2)


def process_count_option(command: _Command) -> _Command:
    """The ``--process-count`` option: worker processes, by default the CPU count minus two."""
    return click.option(
        "--process-count",
        type=click.IntRange(min=1),
        default=None,
        help="Worker processes. Default: the machine's CPU count minus 2.",
    )(command)
