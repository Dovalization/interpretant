"""The Prefect flows that run interpretant's jobs, one per job (``prefect.yaml`` deploys each).

Each job's flow takes its command's options as parameters (but ``--catalog``: a run uses its
checkout's catalog) and runs the command's work in-process (``jobs.run_job``). A stage's flow can
also record its outputs in ``dvc.lock`` and push them. ``run_command`` is the escape hatch: any
command, for one-off scripts and the scorers, whose output is copied into the run's log file and
the Prefect run.
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
from interpretant.cli.build_linked_tables import run_build_linked_tables
from interpretant.cli.build_sentence_view import run_build_sentence_view
from interpretant.cli.build_static_view import run_build_static_view
from interpretant.cli.build_usages import run_build_usages
from interpretant.cli.clean_corpus import run_clean_corpus
from interpretant.cli.count_word_repairs import run_count_word_repairs
from interpretant.cli.detect_languages import run_detect_languages
from interpretant.cli.fetch_linked_sources import run_fetch_linked_sources
from interpretant.cli.fetch_metadata import run_fetch_metadata
from interpretant.cli.options import default_process_count
from interpretant.cli.report_corpus import run_report_corpus
from interpretant.orchestration.jobs import run_job
from interpretant.orchestration.registry import JOBS
from interpretant.orchestration.steps import (
    Role,
    capture_checkout_state,
    commit_record,
    open_run_log,
    pull_data,
    push_data,
    send_output_line,
)


class CommandFailedError(RuntimeError):
    """The command exited with a status other than 0."""


@flow(name="command", flow_run_name="{name}", timeout_seconds=12 * 3600)
def run_command(
    command: list[str],
    name: str,
    pull: list[str] | None = None,
    *,
    push_outputs: bool = False,
    role: Role = "local",
) -> None:
    """Run ``command`` in the checkout, then commit the run's record (and push its outputs)."""
    root = Path.cwd()
    with open_run_log(name, root=root):
        capture_checkout_state(root)
        if pull:
            pull_data(pull, root=root)
        run_and_copy_output(command, root=root)
        commit_record(role, name, str(runtime.flow_run.id), root=root)
        if push_outputs:
            push_data(role, root=root)


@task(name="run")
def run_and_copy_output(command: Sequence[str], *, root: Path) -> None:
    """Run the command, copying each line of its output to the log file and the Prefect run."""
    # The checkout's environment first, as on its command line: `interpretant`, `dvc`, `python`
    path = f"{root / '.venv' / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
    with subprocess.Popen(
        list(command),
        cwd=root,
        env={**os.environ, "PATH": path},
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
def build_linked_tables(
    *, role: Role = "local", pull: list[str] | None = None, save_outputs: bool = False
) -> None:
    """Write the staging tables of OpenAlex, Crossref and PubMed works from the raw files."""
    run_job(
        JOBS["build-linked-tables"],
        lambda catalog, progress, console: run_build_linked_tables(
            catalog, progress=progress, console=console
        ),
        role=role,
        process_count=1,
        pull=pull,
        save_outputs=save_outputs,
    )


@flow(name="count-word-repairs", timeout_seconds=JOBS["count-word-repairs"].timeout_seconds)
def count_word_repairs(
    process_count: int | None = None,
    *,
    role: Role = "local",
    pull: list[str] | None = None,
    save_outputs: bool = False,
) -> None:
    """Count the word repairs and write the repaired texts the spaCy pass reads."""
    processes = process_count or default_process_count()
    run_job(
        JOBS["count-word-repairs"],
        lambda catalog, progress, console: run_count_word_repairs(
            catalog, processes, progress=progress, console=console
        ),
        role=role,
        process_count=processes,
        pull=pull,
        save_outputs=save_outputs,
    )


@flow(name="annotate-corpus", timeout_seconds=JOBS["annotate-corpus"].timeout_seconds)
def annotate_corpus(
    fields: tuple[str, ...] = (),
    decades: tuple[int, ...] = (),
    batch_size: int = 256,
    process_count: int | None = None,
    *,
    role: Role = "local",
    pull: list[str] | None = None,
    save_outputs: bool = False,
) -> None:
    """Annotate the selected cells with spaCy (stage 2), part by part."""
    processes = process_count or default_process_count()
    run_job(
        JOBS["annotate-corpus"],
        lambda catalog, progress, console: run_annotate_corpus(
            catalog, fields, decades, batch_size, processes, progress=progress, console=console
        ),
        role=role,
        process_count=processes,
        pull=pull,
        save_outputs=save_outputs,
    )


@flow(name="detect-languages", timeout_seconds=JOBS["detect-languages"].timeout_seconds)
def detect_languages(
    process_count: int | None = None,
    *,
    role: Role = "local",
    pull: list[str] | None = None,
    save_outputs: bool = False,
) -> None:
    """Give each sentence its language with lingua."""
    processes = process_count or default_process_count()
    run_job(
        JOBS["detect-languages"],
        lambda catalog, progress, console: run_detect_languages(
            catalog, processes, progress=progress, console=console
        ),
        role=role,
        process_count=processes,
        pull=pull,
        save_outputs=save_outputs,
    )


@flow(name="clean-corpus", timeout_seconds=JOBS["clean-corpus"].timeout_seconds)
def clean_corpus(
    process_count: int | None = None,
    *,
    role: Role = "local",
    pull: list[str] | None = None,
    save_outputs: bool = False,
) -> None:
    """Apply the cleaning rules to the annotated abstracts."""
    processes = process_count or default_process_count()
    run_job(
        JOBS["clean-corpus"],
        lambda catalog, progress, console: run_clean_corpus(
            catalog, processes, progress=progress, console=console
        ),
        role=role,
        process_count=processes,
        pull=pull,
        save_outputs=save_outputs,
    )


@flow(name="build-corpus", timeout_seconds=JOBS["build-corpus"].timeout_seconds)
def build_corpus(
    process_count: int | None = None,
    *,
    role: Role = "local",
    pull: list[str] | None = None,
    save_outputs: bool = False,
) -> None:
    """Apply the corpus-level rules to every part and write the abstracts table (stage 3)."""
    processes = process_count or default_process_count()
    run_job(
        JOBS["build-corpus"],
        lambda catalog, progress, console: run_build_corpus(
            catalog, processes, progress=progress, console=console
        ),
        role=role,
        process_count=processes,
        pull=pull,
        save_outputs=save_outputs,
    )


@flow(name="report-corpus", timeout_seconds=JOBS["report-corpus"].timeout_seconds)
def report_corpus(
    *, role: Role = "local", pull: list[str] | None = None, save_outputs: bool = False
) -> None:
    """Write the corpus report and its metrics, read before the review (4b)."""
    run_job(
        JOBS["report-corpus"],
        lambda catalog, progress, console: run_report_corpus(
            catalog, progress=progress, console=console
        ),
        role=role,
        process_count=1,
        pull=pull,
        save_outputs=save_outputs,
    )


@flow(name="build-static-view", timeout_seconds=JOBS["build-static-view"].timeout_seconds)
def build_static_view(
    process_count: int | None = None,
    *,
    role: Role = "local",
    pull: list[str] | None = None,
    save_outputs: bool = False,
) -> None:
    """Write the static view: lemma exceptions, then the spelling and compound maps applied."""
    processes = process_count or default_process_count()
    run_job(
        JOBS["build-static-view"],
        lambda catalog, progress, console: run_build_static_view(
            catalog, processes, progress=progress, console=console
        ),
        role=role,
        process_count=processes,
        pull=pull,
        save_outputs=save_outputs,
    )


@flow(name="build-sentence-view", timeout_seconds=JOBS["build-sentence-view"].timeout_seconds)
def build_sentence_view(
    process_count: int | None = None,
    *,
    role: Role = "local",
    pull: list[str] | None = None,
    save_outputs: bool = False,
) -> None:
    """Write the sentence view, with the sentence cuts repaired."""
    processes = process_count or default_process_count()
    run_job(
        JOBS["build-sentence-view"],
        lambda catalog, progress, console: run_build_sentence_view(
            catalog, processes, progress=progress, console=console
        ),
        role=role,
        process_count=processes,
        pull=pull,
        save_outputs=save_outputs,
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
    role: Role = "local",
    pull: list[str] | None = None,
    save_outputs: bool = False,
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
        role=role,
        process_count=processes,
        pull=pull,
        save_outputs=save_outputs,
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
    role: Role = "local",
    pull: list[str] | None = None,
) -> None:
    """Fetch OpenAlex's record of every raw work, resuming where it stopped."""
    run_job(
        JOBS["fetch-metadata"],
        lambda catalog, progress, console: run_fetch_metadata(
            catalog, fields, decades, workers, progress=progress, console=console
        ),
        role=role,
        process_count=1,
        pull=pull,
        save_outputs=False,
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
    role: Role = "local",
    pull: list[str] | None = None,
) -> None:
    """Look each raw work up in Crossref and PubMed, resuming where it stopped."""
    run_job(
        JOBS["fetch-linked-sources"],
        lambda catalog, progress, console: run_fetch_linked_sources(
            catalog, sources, fields, decades, progress=progress, console=console
        ),
        role=role,
        process_count=1,
        pull=pull,
        save_outputs=False,
    )
