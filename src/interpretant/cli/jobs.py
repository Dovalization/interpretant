"""Running a job from the command line: its progress drawn live, then the run's record.

Each job's command module has a ``run_<job>`` function doing the work: reading its options,
calling its pipeline with a progress reporter, printing what it wrote, and returning the counts
for the run log. The command line runs it here; a Prefect run (``orchestration``) runs the same
function with its own reporters and records the run itself.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from rich.console import Console

from interpretant.catalog.config import CatalogConfig, read_catalog
from interpretant.cli.progress import LiveProgress
from interpretant.pipelines.progress import Progress, StageTimer
from interpretant.pipelines.runs import record_run

JobRun = Callable[[CatalogConfig, Progress, Console], Mapping[str, int]]


def run_on_command_line(command: str, catalog_file: Path, run: JobRun) -> Mapping[str, int]:
    """Run a job with live progress on the console and append its record; its counts."""
    console = Console()
    catalog = read_catalog(catalog_file)
    with LiveProgress(console) as live:
        timer = StageTimer(live)
        counts = run(catalog, timer, console)
    record_run(catalog, command=command, stages=timer.finish(), counts=counts)
    return counts
