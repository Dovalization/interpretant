"""Running a stage's parts one per task: the order the parts start in and the progress."""

from __future__ import annotations

from interpretant.pipelines.workers import map_parts
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
