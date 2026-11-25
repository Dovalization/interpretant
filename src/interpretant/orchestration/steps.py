"""The steps every interpretant flow shares.

They are the run's log file, the checkout it started from, DVC's pull and push, and committing the
run's record. A run's output goes to ``logs/<UTC time>-<name>.log`` as it always has, so the
dashboard that reads those files keeps working. The runner takes one run at a time (its work queue's
limit), so nothing here takes a lock. The record's commit is pushed to GitHub on a branch of its
own, ``runs/<flow run id>``, for ``interpretant submit`` or ``interpretant runs collect`` to bring
back.
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

from prefect import task
from prefect.cache_policies import NO_CACHE
from prefect.logging.handlers import APILogHandler

from interpretant.pipelines.runs import checkout_state

RUN_LOG = Path("metrics/runs.jsonl")
DVC_LOCK = Path("dvc.lock")
# Where the code comes from and the records go: the public repository
ORIGIN = "origin"
LOGS = Path("logs")
# git leaves its index lock for a moment while another git command runs; a record waits for it
INDEX_LOCK_ATTEMPTS = 5
INDEX_LOCK_WAIT_SECONDS = 2.0


# A run's output lines go to the Prefect run through this logger alone: Prefect's console handler
# would write each to the log file a second time, as standard error points there too
LINE_LIMIT = 2048
_output = logging.getLogger("interpretant.orchestration.output")
_output.propagate = False
_output.setLevel(logging.INFO)
if not _output.handlers:
    _output.addHandler(APILogHandler())


class RecordError(RuntimeError):
    """The run finished but its record could not be committed, so the run is not done."""


def send_output_line(line: str) -> None:
    """Send one line of the run's output to the Prefect run, cut to ``LINE_LIMIT`` characters."""
    _output.info(line.rstrip("\n")[:LINE_LIMIT])


class RunOutput(io.TextIOBase):
    """A text stream writing to the run's log file and sending each whole line to the Prefect run.

    A job's console prints here, so its tables and progress lines reach both.
    """

    def __init__(self, log: IO[str]) -> None:
        self.log = log
        self._partial = ""

    def writable(self) -> bool:
        """Always: it is an output stream."""
        return True

    def write(self, text: str) -> int:
        """Write ``text`` to the log file, and send any lines it completes to the Prefect run."""
        self.log.write(text)
        *lines, self._partial = (self._partial + text).split("\n")
        for line in lines:
            send_output_line(line)
        return len(text)

    def flush(self) -> None:
        """Flush the log file."""
        self.log.flush()


@contextmanager
def open_run_log(name: str, *, root: Path) -> Iterator[IO[str]]:
    """Point this process's standard output and error at the run's log file while the run lasts.

    The file descriptors are redirected, not just ``sys.stdout``, so child processes, C extensions
    and Prefect's console log all land in the file too; they are restored afterwards.
    """
    started = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    path = root / LOGS / f"{started}-{name}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with path.open("a", buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            yield log
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


@task(name="checkout-state")
def capture_checkout_state(root: Path) -> tuple[str | None, bool]:
    """The commit the run starts from and whether tracked files differ from it."""
    return checkout_state(root)


@task(name="dvc-pull", retries=2, retry_delay_seconds=[30, 120])
def pull_data(paths: Sequence[str], *, root: Path) -> None:
    """Fetch the given DVC-tracked paths before the run reads them."""
    _run(["dvc", "pull", *paths], root)


@task(name="dvc-push", retries=2, retry_delay_seconds=[30, 120])
def push_data(targets: Sequence[str] = (), *, root: Path) -> None:
    """Upload the outputs the checkout records (``targets``' or all) to the DVC remote."""
    _run(["dvc", "push", *targets], root)


@task(name="dvc-commit")
def record_outputs(stage: str, *, root: Path) -> None:
    """Record in ``dvc.lock`` the stage's outputs as the run just made them from the checkout."""
    _run(["dvc", "commit", "--force", stage], root)


@task(name="record", cache_policy=NO_CACHE)
def commit_record(
    name: str,
    flow_run_id: str,
    paths: Sequence[str] = (),
    *,
    root: Path,
    write_record: Callable[[], object] | None = None,
) -> str | None:
    """Commit the run's record and the files it wrote that git keeps; the commit, if any.

    ``paths`` are those files, or folders of them: new ones are added, and files git ignores
    inside a folder are left out (a path naming an ignored file fails the run); a path the run did
    not write is passed over. ``write_record`` appends the run's record to the run log just before
    the commit. A command run through the ``command`` flow writes its record itself, while it
    runs. The commit is pushed to GitHub as ``runs/<flow run id>``. Nothing is committed when the
    run left the files unchanged. Any failure is the run's: a run whose record is lost is not done,
    and the record it wrote is taken out of the run log again, so the next run's commit does not
    carry it under that run's name.
    """
    files = [str(RUN_LOG), *(path for path in paths if (root / path).exists())]
    run_log = root / RUN_LOG
    before = run_log.read_bytes() if run_log.exists() else b""
    if write_record is not None:
        write_record()
    written = run_log.read_bytes()[len(before) :] if run_log.exists() else b""
    if _git(root, "status", "--porcelain", "--", *files) == "":
        return None
    try:
        _add_with_index_lock_retries(root, files)
        _commit_with_index_lock_retries(root, name, files)
    except RecordError:
        _remove_written(run_log, written)
        raise
    commit = _git(root, "rev-parse", "HEAD")
    _run(["git", "push", "--quiet", ORIGIN, f"HEAD:refs/heads/runs/{flow_run_id}"], root)
    return commit


def _remove_written(run_log: Path, written: bytes) -> None:
    """Take the lines this run wrote back out of the run log, wherever they now are."""
    if not written:
        return
    text = run_log.read_bytes()
    start = text.rfind(written)
    if start != -1:
        run_log.write_bytes(text[:start] + text[start + len(written) :])


def _add_with_index_lock_retries(root: Path, files: Sequence[str]) -> None:
    _git_with_index_lock_retries(root, ["git", "add", "--", *files])


def _commit_with_index_lock_retries(root: Path, name: str, files: Sequence[str]) -> None:
    message = f"chore(metrics): record {name}"
    _git_with_index_lock_retries(
        root, ["git", "commit", "--quiet", "--no-verify", "-m", message, "--", *files]
    )


def _git_with_index_lock_retries(root: Path, command: Sequence[str]) -> None:
    for attempt in range(INDEX_LOCK_ATTEMPTS):
        completed = subprocess.run(
            list(command), cwd=root, capture_output=True, text=True, check=False
        )
        if completed.returncode == 0:
            return
        if "index.lock" not in completed.stderr or attempt == INDEX_LOCK_ATTEMPTS - 1:
            raise RecordError(f"couldn't commit the run's record: {completed.stderr.strip()}")
        time.sleep(INDEX_LOCK_WAIT_SECONDS)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def _run(command: Sequence[str], root: Path) -> None:
    """Run a command with its output in this process's output (the run's log file)."""
    subprocess.run(command, cwd=root, check=True)
