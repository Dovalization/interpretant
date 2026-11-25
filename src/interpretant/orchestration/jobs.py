"""Running a job as a Prefect run: the steps around its work, and the work under its limits.

A job's work is its command's ``run_<job>`` function (``cli``), the same the command line runs.
Here its console writes to the run's log file and the Prefect run, and its progress goes both to
the log, as the command line draws it for a file, and to a progress artifact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import IO, cast

from prefect import runtime, task
from prefect.cache_policies import NO_CACHE
from rich.console import Console

from interpretant.catalog.config import CATALOG_FILE, CatalogConfig, read_catalog
from interpretant.catalog.reports import append_run_record
from interpretant.cli.jobs import JobRun
from interpretant.cli.progress import LiveProgress
from interpretant.orchestration.progress import PrefectProgress
from interpretant.orchestration.registry import Job
from interpretant.orchestration.steps import (
    DVC_LOCK,
    RunOutput,
    capture_checkout_state,
    commit_record,
    open_run_log,
    pull_data,
    push_data,
    record_outputs,
)
from interpretant.pipelines.progress import StageTimer, TeeProgress
from interpretant.pipelines.runs import build_run_record


@dataclass(frozen=True)
class WorkDone:
    """A job's counts and the record of its run, which the record step appends and commits."""

    counts: Mapping[str, int]
    record: dict[str, object]


def run_job(
    job: Job,
    run: JobRun,
    *,
    pull: list[str] | None,
    save_outputs: bool,
    record_as: str | None = None,
    outputs: Sequence[Path] = (),
) -> Mapping[str, int]:
    """Run the job in the runner's checkout, commit its record, and return its counts.

    With ``save_outputs`` its stage's outputs are recorded in ``dvc.lock`` and pushed too.
    ``record_as`` names the run in the run log (the job's name by default). ``outputs`` are the
    files, or folders, this run wrote that git keeps (an audit's key and counts), committed with
    the record, as nobody works in the runner's checkout to commit them.
    """
    if save_outputs and job.stage is None:
        raise ValueError(f"{job.name} is not a DVC stage, so it has no outputs to record")
    root = Path.cwd()
    catalog = read_catalog(root / CATALOG_FILE.name)  # the checkout's, whose paths are its own
    with open_run_log(job.name, root=root) as log:
        checkout = capture_checkout_state(root)
        if pull:
            pull_data(pull, root=root)
        done = do_work(
            job, run, catalog, checkout=checkout, log=log, record_as=record_as or job.name
        )
        paths = [*job.tracked_outputs, *(str(path) for path in outputs)]
        if save_outputs and job.stage is not None:
            record_outputs(job.stage, root=root)
            paths.append(str(DVC_LOCK))
        commit_record(
            job.name,
            str(runtime.flow_run.id),
            paths,
            root=root,
            write_record=partial(append_run_record, catalog, done.record),
        )
        if save_outputs and job.stage is not None:
            push_data([job.stage], root=root)
    return done.counts


@task(name="work", cache_policy=NO_CACHE)
def do_work(
    job: Job,
    run: JobRun,
    catalog: CatalogConfig,
    *,
    checkout: tuple[str | None, bool],
    log: IO[str],
    record_as: str,
) -> WorkDone:
    """Run the job's work; its counts and the record of the run.

    Its progress goes to the log file and to a progress artifact on the run.
    """
    root = Path.cwd()
    # Rich writes and flushes only; typeshed doesn't count a TextIOBase as an IO[str]
    output = cast(IO[str], RunOutput(log))
    console = Console(file=output, force_terminal=False, soft_wrap=True)
    artifact = PrefectProgress(job.name)
    with LiveProgress(console) as live:
        timer = StageTimer(TeeProgress(live, artifact))
        counts = run(catalog, timer, console)
    artifact.close()
    record = build_run_record(
        command=record_as, stages=timer.finish(), counts=counts, root=root, checkout=checkout
    )
    return WorkDone(counts, record)
