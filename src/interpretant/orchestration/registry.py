"""What Prefect needs to know about each job: its timeout, retries and DVC stage.

Plain data, with no Prefect import, so tests and the command line can read it. The flows
(``flows.py``) take their decorators' settings from here, and ``prefect.yaml`` deploys each job
named here on the runner (``tests/test_orchestration_jobs.py`` checks both, and each flow's
parameters against its command's options).
"""

from __future__ import annotations

from dataclasses import dataclass

HOUR = 3600
# Where DVC records each stage's outputs, committed with a reproduced stage's record
DVC_LOCK = "dvc.lock"


@dataclass(frozen=True)
class Job:
    """One job: a command whose work a flow runs in-process, with its own reporters.

    ``stage`` is its DVC stage, whose outputs a run can record (``dvc commit``) and push.
    ``tracked_outputs`` are the files it rewrites that git tracks, committed with the run's record
    so the runner's checkout is left clean.
    """

    name: str
    timeout_seconds: int
    stage: str | None = None
    tracked_outputs: tuple[str, ...] = ()
    retries: int = 0
    retry_delay_seconds: int = 0


JOBS: dict[str, Job] = {
    job.name: job
    for job in (
        Job("build-linked-tables", 1 * HOUR, stage="build_linked_tables"),
        Job("count-word-repairs", 1 * HOUR, stage="count_word_repairs"),
        Job("annotate-corpus", 3 * HOUR, stage="annotate_corpus"),
        Job("detect-languages", 2 * HOUR, stage="detect_languages"),
        Job("clean-corpus", 1 * HOUR, stage="clean_corpus"),
        Job("build-corpus", 1 * HOUR, stage="build_corpus"),
        Job(
            "report-corpus",
            1 * HOUR,
            stage="report_corpus",
            tracked_outputs=("metrics/corpus.json",),
        ),
        Job("build-static-view", 1 * HOUR, stage="build_static_view"),
        Job("build-sentence-view", 1 * HOUR, stage="build_sentence_view"),
        # No retries: a stopped run is finished with --resume, which skips the parts done
        Job("build-usages", 4 * HOUR, stage="build_usages"),
        Job("embed-records", 2 * HOUR, stage="embed_records"),
        # The fetchers resume where they stopped, so a failed run (an API outage, a rate limit)
        # is retried twice, ten minutes apart
        Job(
            "fetch-metadata",
            12 * HOUR,
            retries=2,
            retry_delay_seconds=600,
        ),
        Job(
            "fetch-linked-sources",
            12 * HOUR,
            retries=2,
            retry_delay_seconds=600,
        ),
        Job("draw-usages-audit", 1 * HOUR),
        Job("draw-junk-gate", 1 * HOUR),
        # judge-api is never retried, as its cap counts one run's spending only
        Job("judge-api", 6 * HOUR),
        # judge-claude-code resumes by part, so a run with failed parts is retried once
        Job("judge-claude-code", 6 * HOUR, retries=1, retry_delay_seconds=60),
        # judge-ollama resumes by part too, so a run with failed parts is retried once
        Job("judge-ollama", 12 * HOUR, retries=1, retry_delay_seconds=60),
    )
}
