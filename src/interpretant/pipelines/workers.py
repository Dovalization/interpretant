"""Running a stage's parts in worker processes, one part per task.

The workers are started with ``spawn``: Polars runs its own threads, and its documentation warns
that ``fork`` can then deadlock a worker ("Multiprocessing" in the Polars user guide). A worker
therefore receives only picklable arguments: the function must be importable, and each task
carries its part's own input. With one process the parts run in the calling process.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TypeVar

from interpretant.pipelines.progress import Progress

_Input = TypeVar("_Input")
_Result = TypeVar("_Result")


def map_parts(
    function: Callable[[str, _Input], _Result],
    inputs: Mapping[str, _Input],
    *,
    process_count: int,
    progress: Progress,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, _Result]:
    """Run ``function(part, input)`` for every part, advancing as each part finishes.

    The progress advances by one per part, or by the part's size when ``sizes`` gives each
    part's, and the parts then start largest first, so that a long part does not start last.
    Returns each part's result, in part order. A worker's exception is raised here.
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
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=process_count, mp_context=context) as pool:
        futures = {pool.submit(function, part, inputs[part]): part for part in order}
        finished = {}
        for future in as_completed(futures):
            part = futures[future]
            finished[part] = future.result()
            advance(part)
    return {part: finished[part] for part in inputs}
