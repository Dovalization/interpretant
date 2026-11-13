"""Live progress: the command line's display, and the stages the corpus pipelines report."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console

from interpretant.catalog import read_catalog
from interpretant.cli.progress import LiveProgress
from interpretant.pipelines.annotate_corpus import annotate_corpus
from interpretant.pipelines.build_corpus import BuildCorpusSettings, build_corpus
from tests.corpus_catalog import ANNOTATE_SETTINGS, SELECTION


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


def test_the_corpus_stages_report_every_step_and_every_abstract(
    corpus_catalog_file: Path,
) -> None:
    catalog = read_catalog(corpus_catalog_file)
    annotated = RecordingProgress()
    annotate_corpus(catalog, SELECTION, ANNOTATE_SETTINGS, progress=annotated)
    assert annotated.stages[-1] == ("spaCy pass", 4, 4)
    built = RecordingProgress()
    build_corpus(catalog, SELECTION, BuildCorpusSettings(), progress=built)
    assert [name for name, _, _ in built.stages] == [
        "loading the annotated parts",
        "checking the work metadata",
        "template sentences",
        "repeated sentences",
        "near-duplicates",
        "author cap",
        "review flags",
        "saving the parts",
    ]
    counts = {name: (total, done) for name, total, done in built.stages}
    assert counts["near-duplicates"] == (3, 3)
