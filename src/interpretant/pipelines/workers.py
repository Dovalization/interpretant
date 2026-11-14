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
) -> dict[str, _Result]:
    """Run ``function(part, input)`` for every part, advancing once per finished part.

    Returns each part's result, in part order. A worker's exception is raised here.
    """
    if process_count == 1:
        results: dict[str, _Result] = {}
        for part, part_input in inputs.items():
            results[part] = function(part, part_input)
            progress.advance()
        return results
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=process_count, mp_context=context) as pool:
        futures = {
            pool.submit(function, part, part_input): part for part, part_input in inputs.items()
        }
        finished = {}
        for future in as_completed(futures):
            finished[futures[future]] = future.result()
            progress.advance()
    return {part: finished[part] for part in inputs}
