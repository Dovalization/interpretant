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
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from typing import TypeVar

from interpretant.pipelines.progress import Progress

_Input = TypeVar("_Input")
_Result = TypeVar("_Result")
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
