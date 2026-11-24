"""The steps every interpretant flow shares.

They are the run's log file, the checkout it started from, DVC's pull and push, and committing the
run's record. A run's output goes to ``logs/<UTC time>-<name>.log`` as it always has, so the
dashboard that reads those files keeps working. Its record is committed under a lock per role, so
two runs ending at once never collide on git's index; on the runner the commit is pushed to the
relay on a branch of its own, ``runs/<flow run id>``, for ``interpretant runs collect`` to bring
back.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Literal

from prefect import get_run_logger, task
from prefect.concurrency.sync import concurrency
from prefect.exceptions import MissingContextError

from interpretant.pipelines.runs import checkout_state

Role = Literal["runner", "local"]

RUN_LOG = Path("metrics/runs.jsonl")
RELAY = "relay"
LOGS = Path("logs")
# git leaves its index lock for a moment while another git command runs; a record waits for it
INDEX_LOCK_ATTEMPTS = 5
INDEX_LOCK_WAIT_SECONDS = 2.0


class RecordError(RuntimeError):
    """The run finished but its record could not be committed, so the run is not done."""


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
def push_data(role: Role, *, root: Path) -> None:
    """Upload the outputs the run's checkout records, one DVC command per role at a time."""
    with concurrency(f"interpretant-dvc-{role}", strict=True):
        _run(["dvc", "push"], root)


@task(name="record")
def commit_record(role: Role, name: str, flow_run_id: str, *, root: Path) -> str | None:
    """Commit the run's record and, on the runner, push it to the relay; the commit, if any.

    Nothing is committed when the run left the run log unchanged. On the local machine a rebase or
    merge in progress there is left alone and the record stays uncommitted. Any other failure is
    the run's: a run whose record is lost is not done.
    """
    with concurrency(f"interpretant-git-{role}", strict=True):
        if _git(root, "status", "--porcelain", "--", str(RUN_LOG)) == "":
            return None
        if role == "local" and _is_mid_operation(root):
            _logger().warning("left %s uncommitted: a rebase or merge is in progress here", RUN_LOG)
            return None
        _commit_with_index_lock_retries(root, name)
        commit = _git(root, "rev-parse", "HEAD")
        if role == "runner":
            _run(["git", "push", "--quiet", RELAY, f"HEAD:refs/heads/runs/{flow_run_id}"], root)
        return commit


def _commit_with_index_lock_retries(root: Path, name: str) -> None:
    command = ["git", "commit", "--quiet", "--no-verify", "-m", f"chore(metrics): record {name}"]
    for attempt in range(INDEX_LOCK_ATTEMPTS):
        completed = subprocess.run(
            [*command, "--", str(RUN_LOG)], cwd=root, capture_output=True, text=True, check=False
        )
        if completed.returncode == 0:
            return
        if "index.lock" not in completed.stderr or attempt == INDEX_LOCK_ATTEMPTS - 1:
            raise RecordError(f"couldn't commit the run's record: {completed.stderr.strip()}")
        time.sleep(INDEX_LOCK_WAIT_SECONDS)


def _logger() -> logging.Logger | logging.LoggerAdapter[logging.Logger]:
    """The run's logger inside a Prefect run, this module's outside one (tests, scripts)."""
    try:
        return get_run_logger()
    except MissingContextError:
        return logging.getLogger(__name__)


def _is_mid_operation(root: Path) -> bool:
    git_dir = Path(_git(root, "rev-parse", "--absolute-git-dir"))
    markers = ("rebase-merge", "rebase-apply", "MERGE_HEAD", "CHERRY_PICK_HEAD")
    return any((git_dir / marker).exists() for marker in markers)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def _run(command: Sequence[str], root: Path) -> None:
    """Run a command with its output in this process's output (the run's log file)."""
    subprocess.run(command, cwd=root, check=True)
