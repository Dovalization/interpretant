"""The Prefect flows that run interpretant's jobs, one per job (``prefect.yaml`` deploys each).

Each job's flow takes its command's options as parameters (but ``--catalog``: a run uses its
checkout's catalog) and runs the command's work in-process (``jobs.run_job``). ``run_command`` runs
any other command: one-off scripts, the scorers, and ``dvc repro`` when a stage's outputs are to be
recorded in ``dvc.lock``. Its output is copied into the run's log file and the Prefect run.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from prefect import flow, runtime, task

from interpretant.catalog.linked_records import LinkedSource
from interpretant.cli.annotate_corpus import run_annotate_corpus
from interpretant.cli.build_corpus import run_build_corpus
from interpretant.cli.build_junk_gate import run_build_junk_gate
from interpretant.cli.build_linked_tables import run_build_linked_tables
from interpretant.cli.build_sentence_view import run_build_sentence_view
from interpretant.cli.build_static_view import run_build_static_view
from interpretant.cli.build_usages import run_build_usages
from interpretant.cli.clean_corpus import run_clean_corpus
from interpretant.cli.count_word_repairs import run_count_word_repairs
from interpretant.cli.detect_languages import run_detect_languages
from interpretant.cli.draw_junk_gate import run_draw_junk_gate
from interpretant.cli.draw_usages_audit import run_draw_usages_audit
from interpretant.cli.embed_records import run_embed_records
from interpretant.cli.fetch_linked_sources import run_fetch_linked_sources
from interpretant.cli.fetch_metadata import run_fetch_metadata
from interpretant.cli.judge_api import build_record_name, run_judge_api
from interpretant.cli.judge_claude_code import run_judge_claude_code
from interpretant.cli.judge_ollama import run_judge_ollama
from interpretant.cli.options import default_process_count
from interpretant.cli.report_corpus import run_report_corpus
from interpretant.cli.score_records import DEFAULT_HOST, run_score_records
from interpretant.orchestration.jobs import run_job
from interpretant.orchestration.registry import JOBS
from interpretant.orchestration.steps import (
    build_checkout_environment,
    capture_checkout_state,
    commit_record,
    open_run_log,
    pull_data,
    push_data,
    send_output_line,
)
from interpretant.pipelines.judge_api import ApiJudging, Thinking
from interpretant.pipelines.judge_claude_code import CommandLineJudging
from interpretant.pipelines.judge_ollama import JudgeKind, OllamaJudging
from interpretant.review.judging import Answers, Effort
from interpretant.review.junk_gate import DrawSet
from interpretant.review.usages_audit import AUDIT_SEED


class CommandFailedError(RuntimeError):
    """The command exited with a status other than 0."""


class JudgingIncompleteError(RuntimeError):
    """Some parts were not judged; a retry, or the next run, judges only those."""


@flow(name="command", flow_run_name="{name}", timeout_seconds=12 * 3600)
def run_command(
    command: list[str],
    name: str,
    pull: list[str] | None = None,
    outputs: list[str] | None = None,
    *,
    push_outputs: bool = False,
) -> None:
    """Run ``command`` in the checkout, then commit the run's record (and push its outputs).

    ``outputs`` are the files, or folders, the command writes that git keeps (a scorer's results),
    committed with the record.
    """
    root = Path.cwd()
    with open_run_log(name, root=root):
        capture_checkout_state(root)
        if pull:
            pull_data(pull, root=root)
        run_and_copy_output(command, root=root)
        commit_record(name, str(runtime.flow_run.id), outputs or [], root=root)
        if push_outputs:
            push_data(root=root)


@task(name="run")
def run_and_copy_output(command: Sequence[str], *, root: Path) -> None:
    """Run the command, copying each line of its output to the log file and the Prefect run."""
    with subprocess.Popen(
        list(command),
        cwd=root,
        env=build_checkout_environment(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            # To the file descriptor, which open_run_log points at the log file whatever object
            # sys.stdout has been replaced with
            os.write(1, line.encode())
            send_output_line(line)
    if process.returncode != 0:
        raise CommandFailedError(f"{command[0]} exited with status {process.returncode}")


# The jobs' flows: each decorator's settings come from the job's registry entry


@flow(
    name="build-linked-tables",
    timeout_seconds=JOBS["build-linked-tables"].timeout_seconds,
)
def build_linked_tables(*, pull: list[str] | None = None) -> None:
    """Write the staging tables of OpenAlex, Crossref and PubMed works from the raw files."""
    run_job(
        JOBS["build-linked-tables"],
        lambda catalog, progress, console: run_build_linked_tables(
            catalog, progress=progress, console=console
        ),
        pull=pull,
    )


@flow(name="count-word-repairs", timeout_seconds=JOBS["count-word-repairs"].timeout_seconds)
def count_word_repairs(
    process_count: int | None = None,
    *,
    pull: list[str] | None = None,
) -> None:
    """Count the word repairs and write the repaired texts the spaCy pass reads."""
    processes = process_count or default_process_count()
    run_job(
        JOBS["count-word-repairs"],
        lambda catalog, progress, console: run_count_word_repairs(
            catalog, processes, progress=progress, console=console
        ),
        pull=pull,
    )


@flow(name="annotate-corpus", timeout_seconds=JOBS["annotate-corpus"].timeout_seconds)
def annotate_corpus(
    fields: tuple[str, ...] = (),
    decades: tuple[int, ...] = (),
    batch_size: int = 256,
    process_count: int | None = None,
    *,
    pull: list[str] | None = None,
) -> None:
    """Annotate the selected cells with spaCy (stage 2), part by part."""
    processes = process_count or default_process_count()
    run_job(
        JOBS["annotate-corpus"],
        lambda catalog, progress, console: run_annotate_corpus(
            catalog, fields, decades, batch_size, processes, progress=progress, console=console
        ),
        pull=pull,
    )


@flow(name="detect-languages", timeout_seconds=JOBS["detect-languages"].timeout_seconds)
def detect_languages(
    process_count: int | None = None,
    *,
    pull: list[str] | None = None,
) -> None:
    """Give each sentence its language with lingua."""
    processes = process_count or default_process_count()
    run_job(
        JOBS["detect-languages"],
        lambda catalog, progress, console: run_detect_languages(
            catalog, processes, progress=progress, console=console
        ),
        pull=pull,
    )


@flow(name="clean-corpus", timeout_seconds=JOBS["clean-corpus"].timeout_seconds)
def clean_corpus(
    process_count: int | None = None,
    *,
    pull: list[str] | None = None,
) -> None:
    """Apply the cleaning rules to the annotated abstracts."""
    processes = process_count or default_process_count()
    run_job(
        JOBS["clean-corpus"],
        lambda catalog, progress, console: run_clean_corpus(
            catalog, processes, progress=progress, console=console
        ),
        pull=pull,
    )


@flow(name="build-corpus", timeout_seconds=JOBS["build-corpus"].timeout_seconds)
def build_corpus(
    process_count: int | None = None,
    *,
    pull: list[str] | None = None,
) -> None:
    """Apply the corpus-level rules to every part and write the abstracts table (stage 3)."""
    processes = process_count or default_process_count()
    run_job(
        JOBS["build-corpus"],
        lambda catalog, progress, console: run_build_corpus(
            catalog, processes, progress=progress, console=console
        ),
        pull=pull,
    )


@flow(name="report-corpus", timeout_seconds=JOBS["report-corpus"].timeout_seconds)
def report_corpus(*, pull: list[str] | None = None) -> None:
    """Write the corpus report and its metrics, read before the review (4b)."""
    run_job(
        JOBS["report-corpus"],
        lambda catalog, progress, console: run_report_corpus(
            catalog, progress=progress, console=console
        ),
        pull=pull,
    )


@flow(name="build-static-view", timeout_seconds=JOBS["build-static-view"].timeout_seconds)
def build_static_view(
    process_count: int | None = None,
    *,
    pull: list[str] | None = None,
) -> None:
    """Write the static view: lemma exceptions, then the spelling and compound maps applied."""
    processes = process_count or default_process_count()
    run_job(
        JOBS["build-static-view"],
        lambda catalog, progress, console: run_build_static_view(
            catalog, processes, progress=progress, console=console
        ),
        pull=pull,
    )


@flow(name="build-sentence-view", timeout_seconds=JOBS["build-sentence-view"].timeout_seconds)
def build_sentence_view(
    process_count: int | None = None,
    *,
    pull: list[str] | None = None,
) -> None:
    """Write the sentence view, with the sentence cuts repaired."""
    processes = process_count or default_process_count()
    run_job(
        JOBS["build-sentence-view"],
        lambda catalog, progress, console: run_build_sentence_view(
            catalog, processes, progress=progress, console=console
        ),
        pull=pull,
    )


@flow(name="build-usages", timeout_seconds=JOBS["build-usages"].timeout_seconds)
def build_usages(
    fields: tuple[str, ...] = (),
    decades: tuple[int, ...] = (),
    gpu_process_count: int = 7,
    batch_tokens: int = 1000,
    process_count: int | None = None,
    *,
    resume: bool = False,
    reuse_cache: bool = False,
    pull: list[str] | None = None,
) -> None:
    """Find the targets' mentions and write their usages, on the GPU (decision agenda 4.1-4.6)."""
    processes = process_count or default_process_count()
    run_job(
        JOBS["build-usages"],
        lambda catalog, progress, console: run_build_usages(
            catalog,
            fields,
            decades,
            gpu_process_count,
            batch_tokens,
            processes,
            resume=resume,
            reuse_cache=reuse_cache,
            progress=progress,
            console=console,
        ),
        pull=pull,
    )


@flow(name="embed-records", timeout_seconds=JOBS["embed-records"].timeout_seconds)
def embed_records(*, pull: list[str] | None = None) -> None:
    """Encode every record behind a drawn usage with the frozen text encoder, on the GPU."""
    run_job(
        JOBS["embed-records"],
        lambda catalog, progress, console: run_embed_records(
            catalog, progress=progress, console=console
        ),
        pull=pull,
    )


@flow(name="score-records", timeout_seconds=JOBS["score-records"].timeout_seconds)
def score_records(
    host: str = DEFAULT_HOST,
    limit: int | None = None,
    out: Path | None = None,
    *,
    pull: list[str] | None = None,
) -> None:
    """Score every record behind a drawn usage with the junk gate's two decision models."""
    run_job(
        JOBS["score-records"],
        lambda catalog, progress, console: run_score_records(
            catalog, host, limit, out, progress=progress, console=console
        ),
        pull=pull,
    )


@flow(name="build-junk-gate", timeout_seconds=JOBS["build-junk-gate"].timeout_seconds)
def build_junk_gate(
    process_count: int | None = None,
    *,
    pull: list[str] | None = None,
) -> None:
    """Score every record behind a drawn usage for junk and screen the cells."""
    run_job(
        JOBS["build-junk-gate"],
        lambda catalog, progress, console: run_build_junk_gate(
            catalog, process_count, progress=progress, console=console
        ),
        pull=pull,
    )


@flow(
    name="fetch-metadata",
    timeout_seconds=JOBS["fetch-metadata"].timeout_seconds,
    retries=JOBS["fetch-metadata"].retries,
    retry_delay_seconds=JOBS["fetch-metadata"].retry_delay_seconds,
)
def fetch_metadata(
    fields: tuple[str, ...] = (),
    decades: tuple[int, ...] = (),
    workers: int = 8,
    *,
    pull: list[str] | None = None,
) -> None:
    """Fetch OpenAlex's record of every raw work, resuming where it stopped."""
    run_job(
        JOBS["fetch-metadata"],
        lambda catalog, progress, console: run_fetch_metadata(
            catalog, fields, decades, workers, progress=progress, console=console
        ),
        pull=pull,
    )


@flow(
    name="fetch-linked-sources",
    timeout_seconds=JOBS["fetch-linked-sources"].timeout_seconds,
    retries=JOBS["fetch-linked-sources"].retries,
    retry_delay_seconds=JOBS["fetch-linked-sources"].retry_delay_seconds,
)
def fetch_linked_sources(
    sources: tuple[LinkedSource, ...] = (),
    fields: tuple[str, ...] = (),
    decades: tuple[int, ...] = (),
    *,
    pull: list[str] | None = None,
) -> None:
    """Look each raw work up in Crossref and PubMed, resuming where it stopped."""
    run_job(
        JOBS["fetch-linked-sources"],
        lambda catalog, progress, console: run_fetch_linked_sources(
            catalog, sources, fields, decades, progress=progress, console=console
        ),
        pull=pull,
    )


@flow(name="draw-usages-audit", timeout_seconds=JOBS["draw-usages-audit"].timeout_seconds)
def draw_usages_audit(
    out: Path,
    seed: int = AUDIT_SEED,
    process_count: int | None = None,
    *,
    skip_calibration: bool = False,
    pull: list[str] | None = None,
) -> None:
    """Draw the matching audit's kept, left-out and missed usages as review items."""
    processes = process_count or default_process_count()
    run_job(
        JOBS["draw-usages-audit"],
        lambda catalog, progress, console: run_draw_usages_audit(
            catalog,
            out,
            seed,
            processes,
            skip_calibration=skip_calibration,
            progress=progress,
            console=console,
        ),
        pull=pull,
        record_as=f"draw-usages-audit {seed}",
        # The items quote corpus sentences and stay out of git
        outputs=[out / "key.json", out / "calibration.json", out / "counts.json"],
    )


@flow(name="draw-junk-gate", timeout_seconds=JOBS["draw-junk-gate"].timeout_seconds)
def draw_junk_gate(
    out: Path,
    draw_set: DrawSet = "sample",
    process_count: int | None = None,
    *,
    pull: list[str] | None = None,
) -> None:
    """Draw one of the junk gate's sets of records as review items."""
    processes = process_count or default_process_count()
    run_job(
        JOBS["draw-junk-gate"],
        lambda catalog, progress, console: run_draw_junk_gate(
            catalog, out, draw_set, processes, progress=progress, console=console
        ),
        pull=pull,
        record_as=f"draw-junk-gate {draw_set}",
        # The items quote the corpus and stay out of git
        outputs=[out / "key.json", out / "counts.json"],
    )


# The judges' verdicts stay where they were written, out of git: their reasons can quote corpus
# sentences and name held-out words. A scorer run there brings back only its results, which seal
# those names.


@flow(name="judge-api", timeout_seconds=JOBS["judge-api"].timeout_seconds)
def judge_api(
    items: Path,
    out: Path,
    codebook: Path,
    answers: Answers,
    model: str,
    effort: Effort,
    max_dollars: float,
    chunk: int = 500,
    max_tokens: int = 4096,
    thinking: Thinking = "adaptive",
    second_out: Path | None = None,
    *,
    pull: list[str] | None = None,
) -> None:
    """Judge review items through the Message Batches API, within a cap of this run's spending.

    With ``second_out``, a second judge judges the same items beside the first.
    """
    judging = ApiJudging(
        codebook=codebook,
        answers=answers,
        model=model,
        effort=effort,
        max_dollars=max_dollars,
        chunk=chunk,
        max_tokens=max_tokens,
        thinking=thinking,
    )
    run_job(
        JOBS["judge-api"],
        lambda catalog, progress, console: run_judge_api(
            catalog, items, out, judging, progress=progress, console=console, second_out=second_out
        ),
        pull=pull,
        record_as=build_record_name(model, second_out),
    )


@flow(
    name="judge-claude-code",
    timeout_seconds=JOBS["judge-claude-code"].timeout_seconds,
    retries=JOBS["judge-claude-code"].retries,
    retry_delay_seconds=JOBS["judge-claude-code"].retry_delay_seconds,
)
def judge_claude_code(
    items: Path,
    verdicts: Path,
    name: str,
    codebook: Path,
    answers: Answers,
    model: str,
    effort: Effort,
    items_per_call: int = 1,
    parallel: int = 4,
    claude: str = "claude",
    *,
    pull: list[str] | None = None,
) -> None:
    """Judge review items with Claude Code on the subscription; fails if a part failed."""
    judging = CommandLineJudging(
        name=name,
        codebook=codebook,
        answers=answers,
        model=model,
        effort=effort,
        items_per_call=items_per_call,
        parallel=parallel,
        claude=claude,
    )
    counts = run_job(
        JOBS["judge-claude-code"],
        lambda catalog, progress, console: run_judge_claude_code(
            catalog, items, verdicts, judging, progress=progress, console=console
        ),
        pull=pull,
        record_as=f"judge-claude-code {model}",
    )
    if counts["failed_parts"]:
        raise JudgingIncompleteError(f"{counts['failed_parts']} parts failed")


@flow(
    name="judge-ollama",
    timeout_seconds=JOBS["judge-ollama"].timeout_seconds,
    retries=JOBS["judge-ollama"].retries,
    retry_delay_seconds=JOBS["judge-ollama"].retry_delay_seconds,
)
def judge_ollama(
    items: Path,
    verdicts: Path,
    name: str,
    codebook: Path,
    answers: Answers,
    model: str,
    kind: JudgeKind,
    host: str,
    host_label: str,
    parallel: int = 1,
    keep_alive: str = "2m",
    context_length: int = 8192,
    *,
    pull: list[str] | None = None,
) -> None:
    """Judge review items with a local model on an Ollama server; fails if a part failed."""
    judging = OllamaJudging(
        name=name,
        codebook=codebook,
        answers=answers,
        model=model,
        kind=kind,
        host_label=host_label,
        parallel=parallel,
        keep_alive=keep_alive,
        context_length=context_length,
    )
    counts = run_job(
        JOBS["judge-ollama"],
        lambda catalog, progress, console: run_judge_ollama(
            catalog, items, verdicts, judging, host, progress=progress, console=console
        ),
        pull=pull,
        record_as=f"judge-ollama {model} on {host_label}",
    )
    if counts["failed_parts"]:
        raise JudgingIncompleteError(f"{counts['failed_parts']} parts failed")
