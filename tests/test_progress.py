"""Live progress: the command line's display, and the stages the corpus pipelines report."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from interpretant.catalog import read_catalog
from interpretant.cli.progress import LiveProgress
from interpretant.pipelines.annotate_corpus import annotate_corpus
from interpretant.pipelines.build_corpus import BuildCorpusSettings, build_corpus
from interpretant.pipelines.build_linked_tables import build_linked_tables
from interpretant.pipelines.clean_corpus import clean_corpus
from interpretant.pipelines.detect_languages import detect_languages
from interpretant.pipelines.progress import StageTimer
from tests.corpus_catalog import (
    ANNOTATE_SETTINGS,
    CLEAN_SETTINGS,
    LANGUAGE_SETTINGS,
    SELECTION,
)


class Clock:
    """A clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class RecordingProgress:
    """Keeps every stage a pipeline reports, with how many items it advanced in each."""

    def __init__(self) -> None:
        self.stages: list[tuple[str, int | None, int]] = []

    def stage(self, name: str, total: int | None = None) -> None:
        self.stages.append((name, total, 0))

    def advance(self, amount: int = 1) -> None:
        name, total, done = self.stages[-1]
        self.stages[-1] = (name, total, done + amount)

    def skip(self, amount: int) -> None:
        self.advance(amount)


def log_console() -> tuple[Console, StringIO]:
    output = StringIO()
    return Console(file=output, force_terminal=False, width=120), output


def test_live_progress_prints_lines_in_a_log_at_each_interval() -> None:
    console, output = log_console()
    clock = Clock()
    with LiveProgress(console, interval=30, clock=clock) as progress:
        progress.stage("spaCy pass", total=1000)
        clock.now = 10
        progress.advance(100)
        clock.now = 40
        progress.advance(300)
        clock.now = 50
    assert output.getvalue().splitlines() == [
        "spaCy pass: started (1,000 items)",
        "spaCy pass: 400/1,000 (40.0%), 10/s, about 1 min 00 s left",
        "spaCy pass: done (400 items, 50 s)",
    ]


@pytest.mark.parametrize(
    ("completed", "rate"),
    [(400, "10/s"), (100, "2.5/s"), (12, "18.0/min")],
    ids=["whole items a second", "decimals under ten a second", "a minute under one a second"],
)
def test_live_progress_keeps_a_figure_in_slow_rates(completed: int, rate: str) -> None:
    console, output = log_console()
    clock = Clock()
    with LiveProgress(console, interval=30, clock=clock) as progress:
        progress.stage("cleaning the parts", total=1000)
        clock.now = 40
        progress.advance(completed)
    assert f", {rate}, about " in output.getvalue().splitlines()[1]


def test_live_progress_leaves_an_earlier_runs_items_out_of_the_rate_in_a_log() -> None:
    console, output = log_console()
    clock = Clock()
    with LiveProgress(console, interval=30, clock=clock) as progress:
        progress.stage("looking the works up in crossref", total=1000)
        progress.skip(600)
        clock.now = 40
        progress.advance(200)
        clock.now = 50
    assert output.getvalue().splitlines()[1:] == [
        "looking the works up in crossref: 800/1,000 (80.0%), 5.0/s, about 40 s left",
        "looking the works up in crossref: done (800 items, 600 of them from an earlier run, 50 s)",
    ]


def test_live_progress_leaves_an_earlier_runs_items_out_of_the_rate_in_a_terminal() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=160)
    clock = Clock()
    with LiveProgress(console, clock=clock) as progress:
        progress.stage("looking the works up", total=1000)
        progress.skip(600)
        clock.now = 40
        progress.advance(200)
    assert "800/1000" in output.getvalue()
    assert "5.0/s, about 40 s left" in output.getvalue()


def test_live_progress_closes_each_stage_when_the_next_starts() -> None:
    console, output = log_console()
    clock = Clock()
    with LiveProgress(console, clock=clock) as progress:
        progress.stage("merging the raw cells")
        clock.now = 3725
        progress.stage("checking the work metadata")
    assert output.getvalue().splitlines() == [
        "merging the raw cells: started",
        "merging the raw cells: done (1 h 02 min)",
        "checking the work metadata: started",
        "checking the work metadata: done (0 s)",
    ]


def test_live_progress_draws_bars_in_a_terminal() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=120)
    with LiveProgress(console) as progress:
        progress.stage("near-duplicates", total=10)
        progress.advance(10)
    assert "near-duplicates" in output.getvalue()


def test_stage_timer_times_each_stage_and_passes_the_progress_on() -> None:
    clock = Clock()
    recorded = RecordingProgress()
    timer = StageTimer(recorded, clock=clock)
    timer.stage("loading")
    clock.now = 2.5
    timer.stage("near-duplicates", total=3)
    timer.advance(3)
    clock.now = 10
    assert timer.finish() == [("loading", 2.5), ("near-duplicates", 7.5)]
    assert recorded.stages == [("loading", None, 0), ("near-duplicates", 3, 3)]


def test_stage_timer_without_stages_returns_no_stages() -> None:
    assert StageTimer(RecordingProgress(), clock=Clock()).finish() == []


def test_the_corpus_stages_report_every_step_and_every_abstract(
    corpus_catalog_file: Path,
) -> None:
    catalog = read_catalog(corpus_catalog_file)
    build_linked_tables(catalog)
    annotated = RecordingProgress()
    annotate_corpus(catalog, SELECTION, ANNOTATE_SETTINGS, progress=annotated)
    assert annotated.stages[-1] == ("spaCy pass", 4, 4)
    detected = RecordingProgress()
    detect_languages(catalog, SELECTION, LANGUAGE_SETTINGS, progress=detected)
    assert detected.stages == [("detecting sentence languages", 2, 2)]
    cleaned = RecordingProgress()
    clean_corpus(catalog, SELECTION, CLEAN_SETTINGS, progress=cleaned)
    assert cleaned.stages[-1] == ("cleaning the parts", 2, 2)
    built = RecordingProgress()
    build_corpus(catalog, SELECTION, BuildCorpusSettings(), progress=built)
    assert [name for name, _, _ in built.stages] == [
        "loading the annotated parts",
        "loading the staging tables",
        "record types",
        "template sentences",
        "repeated sentences",
        "near-duplicates",
        "author cap",
        "review flags",
        "saving the parts",
    ]
    counts = {name: (total, done) for name, total, done in built.stages}
    assert counts["near-duplicates"] == (3, 3)
