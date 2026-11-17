"""Running a stage's parts in worker processes, one part per task.

The workers are started with ``spawn``: Polars runs its own threads, and its documentation warns
that ``fork`` can then deadlock a worker ("Multiprocessing" in the Polars user guide). A worker
therefore receives only picklable arguments: the function must be importable, and each task
carries its part's own input. With one process the parts run in the calling process.

A spawned worker imports the command line again before its first task (2 to 3 s), so a stage that
maps its parts several times keeps one pool for all of them (``worker_pool``). Workers run Polars
on one thread each: they work on one part at a time, and each extra thread pool cost about 100 MB
of memory per worker in a test (Polars' ``thread_pool_size`` notes; docs/pipeline_review.md, D).
"""

from __future__ import annotations

import multiprocessing
import os
import queue
import threading
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeVar

from interpretant.pipelines.progress import Progress

_Input = TypeVar("_Input")
_Result = TypeVar("_Result")
# A worker's report of items done inside its part.
Report = Callable[[int], None]
# Polars reads its thread count from this variable when a process imports it.
_POLARS_THREADS = "POLARS_MAX_THREADS"


@contextmanager
def worker_pool(process_count: int) -> Iterator[ProcessPoolExecutor | None]:
    """Spawned worker processes for one or more ``map_parts`` calls; None with one process.

    The workers start with Polars limited to one thread; the calling process keeps its own.
    """
    if process_count == 1:
        yield None
        return
    with _spawned_pool(process_count) as pool:
        yield pool


@contextmanager
def _spawned_pool(process_count: int) -> Iterator[ProcessPoolExecutor]:
    previous = os.environ.get(_POLARS_THREADS)
    os.environ[_POLARS_THREADS] = "1"
    try:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=process_count, mp_context=context) as pool:
            yield pool
    finally:
        if previous is None:
            del os.environ[_POLARS_THREADS]
        else:
            os.environ[_POLARS_THREADS] = previous


def map_parts(
    function: Callable[[str, _Input], _Result],
    inputs: Mapping[str, _Input],
    *,
    process_count: int,
    progress: Progress,
    sizes: Mapping[str, int] | None = None,
    pool: ProcessPoolExecutor | None = None,
) -> dict[str, _Result]:
    """Run ``function(part, input)`` for every part, advancing as each part finishes.

    The progress advances by one per part, or by the part's size when ``sizes`` gives each
    part's, and the parts then start largest first, so that a long part does not start last.
    The parts run in ``pool`` when one is given, otherwise in a pool of their own. Returns each
    part's result, in part order. A worker's exception is raised here.
    """
    order = sorted(inputs, key=lambda part: -sizes[part]) if sizes is not None else list(inputs)

    def advance(part: str) -> None:
        progress.advance(1 if sizes is None else sizes[part])

    if process_count == 1:
        results: dict[str, _Result] = {}
        for part in order:
            results[part] = function(part, inputs[part])
            advance(part)
        return {part: results[part] for part in inputs}
    if pool is not None:
        finished = _run_in(pool, function, inputs, order, advance)
    else:
        with _spawned_pool(process_count) as own_pool:
            finished = _run_in(own_pool, function, inputs, order, advance)
    return {part: finished[part] for part in inputs}


def _run_in(
    pool: ProcessPoolExecutor,
    function: Callable[[str, _Input], _Result],
    inputs: Mapping[str, _Input],
    order: list[str],
    advance: Callable[[str], None],
) -> dict[str, _Result]:
    futures = {pool.submit(function, part, inputs[part]): part for part in order}
    finished = {}
    for future in as_completed(futures):
        part = futures[future]
        finished[part] = future.result()
        advance(part)
    return finished


def map_parts_reporting(
    function: Callable[[str, _Input, Report], _Result],
    inputs: Mapping[str, _Input],
    *,
    process_count: int,
    progress: Progress,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, _Result]:
    """Run ``function(part, input, report)`` for every part, advancing as the parts report.

    For parts that are few and long, where advancing per finished part would leave the progress
    still for minutes: each part calls ``report(amount)`` as it finishes items, and the progress
    advances by every report as it arrives, from the workers through a manager's queue (a proxy
    that spawned workers can receive, as the ``multiprocessing`` documentation describes) and a
    thread of this process. The parts start largest first when ``sizes`` gives each part's size.
    Returns each part's result, in part order. A worker's exception is raised here.
    """
    order = sorted(inputs, key=lambda part: -sizes[part]) if sizes is not None else list(inputs)
    if process_count == 1:
        results = {part: function(part, inputs[part], progress.advance) for part in order}
        return {part: results[part] for part in inputs}
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        reports: queue.Queue[int] = manager.Queue()
        stop = threading.Event()
        relay = threading.Thread(target=_relay_reports, args=(reports, progress, stop), daemon=True)
        relay.start()
        try:
            with _spawned_pool(process_count) as pool:
                futures = {
                    pool.submit(function, part, inputs[part], _QueueReport(reports)): part
                    for part in order
                }
                finished = {futures[future]: future.result() for future in as_completed(futures)}
        finally:
            stop.set()
            relay.join()
            _drain_reports(reports, progress)
    return {part: finished[part] for part in inputs}


@dataclass(frozen=True)
class _QueueReport:
    """A worker's ``report``: puts the amount on the manager's queue."""

    reports: queue.Queue[int]

    def __call__(self, amount: int = 1) -> None:
        self.reports.put(amount)


def _relay_reports(reports: queue.Queue[int], progress: Progress, stop: threading.Event) -> None:
    """Advance the progress by each report until the parts are done."""
    while not stop.is_set():
        try:
            progress.advance(reports.get(timeout=0.2))
        except queue.Empty:
            continue


def _drain_reports(reports: queue.Queue[int], progress: Progress) -> None:
    """Advance by the reports still queued when the parts finished."""
    while True:
        try:
            progress.advance(reports.get_nowait())
        except queue.Empty:
            return
