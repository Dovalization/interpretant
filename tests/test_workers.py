"""Running a stage's parts one per task: the order, the progress and the shared worker pool."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from interpretant.pipelines.progress import SILENT
from interpretant.pipelines.workers import Report, map_parts, map_parts_reporting, worker_pool
from tests.test_progress import RecordingProgress


def test_map_parts_starts_the_largest_parts_first_and_advances_by_their_sizes() -> None:
    started: list[str] = []

    def record(part: str, value: int) -> int:
        started.append(part)
        return value * 2

    progress = RecordingProgress()
    progress.stage("parts", total=9)
    inputs = {"small": 1, "large": 2, "middle": 3}
    sizes = {"small": 1, "large": 5, "middle": 3}
    results = map_parts(record, inputs, process_count=1, progress=progress, sizes=sizes)
    assert started == ["large", "middle", "small"]
    assert list(results.items()) == [("small", 2), ("large", 4), ("middle", 6)]
    assert progress.stages == [("parts", 9, 9)]


def test_map_parts_advances_once_per_part_without_sizes() -> None:
    progress = RecordingProgress()
    progress.stage("parts", total=2)
    results = map_parts(
        lambda part, value: value, {"a": 1, "b": 2}, process_count=1, progress=progress
    )
    assert results == {"a": 1, "b": 2}
    assert progress.stages == [("parts", 2, 2)]


def worker_view(part: str, _value: int) -> tuple[int, str | None]:
    """The worker's process id and its Polars thread setting (importable, for spawned workers)."""
    return os.getpid(), os.environ.get("POLARS_MAX_THREADS")


def test_worker_pool_keeps_its_workers_for_several_rounds_with_one_polars_thread() -> None:
    inputs = {"a": 1, "b": 2, "c": 3}
    with worker_pool(2) as pool:
        first = map_parts(worker_view, inputs, process_count=2, progress=SILENT, pool=pool)
        second = map_parts(worker_view, inputs, process_count=2, progress=SILENT, pool=pool)
    workers = {pid for pid, _ in first.values()}
    assert {pid for pid, _ in second.values()} <= workers
    assert len(workers) <= 2
    assert os.getpid() not in workers
    assert {setting for _, setting in [*first.values(), *second.values()]} == {"1"}
    assert "POLARS_MAX_THREADS" not in os.environ


def test_worker_pool_runs_in_this_process_with_one_process() -> None:
    with worker_pool(1) as pool:
        assert pool is None


def report_each_item(part: str, items: int, report: Report) -> str:
    """Reports every item of a part as it goes (importable, for spawned workers)."""
    for _ in range(items):
        report(1)
    return part.upper()


def test_map_parts_reporting_advances_by_every_report_in_this_process() -> None:
    progress = RecordingProgress()
    progress.stage("items", total=6)
    results = map_parts_reporting(
        report_each_item, {"a": 2, "b": 4}, process_count=1, progress=progress
    )
    assert results == {"a": "A", "b": "B"}
    assert progress.stages == [("items", 6, 6)]


def test_map_parts_reporting_relays_the_workers_reports() -> None:
    progress = RecordingProgress()
    progress.stage("items", total=9)
    results = map_parts_reporting(
        report_each_item, {"a": 2, "b": 3, "c": 4}, process_count=2, progress=progress
    )
    assert results == {"a": "A", "b": "B", "c": "C"}
    assert progress.stages == [("items", 9, 9)]


def fail_first_or_mark(part: str, folder: str) -> str:
    """Fail on part p00; on any other part wait a little and leave a marker (importable)."""
    if part == "p00":
        raise ValueError("this part fails")
    time.sleep(0.3)
    Path(folder, part).touch()
    return part


def test_map_parts_cancels_the_queued_parts_when_one_fails(tmp_path: Path) -> None:
    parts = {f"p{index:02d}": str(tmp_path) for index in range(40)}
    with pytest.raises(ValueError, match="this part fails"):
        map_parts(fail_first_or_mark, parts, process_count=2, progress=SILENT)
    assert len(list(tmp_path.iterdir())) < 20
