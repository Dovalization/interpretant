"""Submitting a run to Prefect from this checkout, following it, and collecting its record.

A run starts from a commit, not from this working tree: the commit must already be on GitHub, and
the runner checks it out when the run starts. Its record comes back on the GitHub branch
``runs/<flow run id>`` and is cherry-picked here.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

from prefect.client.orchestration import get_client
from prefect.client.schemas.filters import LogFilter, LogFilterFlowRunId
from prefect.client.schemas.sorting import LogSort
from prefect.deployments import run_deployment

from interpretant.orchestration.steps import ORIGIN

POLL_SECONDS = 2.0
LOG_PAGE = 200
# How long the lines a run sent before ending may take to arrive: FlowRunSubscriber's default
STRAGGLER_SECONDS = 3.0


class SubmissionError(RuntimeError):
    """The run could not be submitted; nothing was started."""


@dataclass(frozen=True)
class Submission:
    """What starting a run of a job's deployment sends to Prefect."""

    deployment: str
    parameters: dict[str, Any]
    job_variables: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


def build_submission(
    flow_name: str,
    parameters: Mapping[str, Any],
    *,
    commit: str,
    stay_awake: bool,
) -> Submission:
    """The runner deployment run for ``flow_name``, starting from ``commit``."""
    tags = ["interpretant", f"commit:{commit[:8]}", *(["stay-awake"] if stay_awake else [])]
    return Submission(
        f"{flow_name}/runner", dict(parameters), {"env": {"INTERPRETANT_COMMIT": commit}}, tags
    )


def get_published_commit(root: Path) -> str:
    """The commit this checkout is on, which must be on GitHub; refuses uncommitted changes."""
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise SubmissionError("commit or stash the changes to tracked files first")
    _git(root, "fetch", "--quiet", ORIGIN)
    branches = _git(root, "branch", "--remotes", "--contains", "HEAD", "--format=%(refname)")
    if not any(name.startswith(f"refs/remotes/{ORIGIN}/") for name in branches.splitlines()):
        raise SubmissionError("push this commit to GitHub first: the runner runs published code")
    return _git(root, "rev-parse", "HEAD")


def start_run(submission: Submission) -> UUID:
    """Create the deployment run and return its id, without waiting for it."""
    with get_client(sync_client=True) as client:
        try:
            client.api_healthcheck()
        except Exception as exc:
            raise SubmissionError(f"Prefect isn't reachable: {exc}") from exc
    flow_run = run_deployment(
        submission.deployment,
        parameters=submission.parameters,
        job_variables=submission.job_variables,
        tags=submission.tags,
        timeout=0,
        as_subflow=False,
    )
    flow_run_id: UUID = flow_run.id  # type: ignore[union-attr]
    return flow_run_id


def follow_run(flow_run_id: UUID, *, echo: Callable[[str], None]) -> str:
    """Print the run's states and log lines until it ends; its final state's name.

    The run sends its log lines in batches, so its last lines can reach the server after its
    final state does. Once the state is final, the lines are read until ``STRAGGLER_SECONDS``
    pass without a new one, as Prefect's own ``FlowRunSubscriber`` does, and the final state
    is printed after them.
    """
    shown = 0
    last_state = ""
    last_line_at = time.monotonic()
    final_seen_at: float | None = None
    with get_client(sync_client=True) as client:
        while True:
            flow_run = client.read_flow_run(flow_run_id)
            state = flow_run.state
            logs = client.read_logs(
                log_filter=LogFilter(flow_run_id=LogFilterFlowRunId(any_=[flow_run_id])),
                offset=shown,
                limit=LOG_PAGE,
                sort=LogSort.TIMESTAMP_ASC,
            )
            now = time.monotonic()
            for log in logs:
                echo(log.message)
            shown += len(logs)
            if logs:
                last_line_at = now
            name = state.name if state is not None and state.name else "Unknown"
            if state is not None and state.is_final():
                final_seen_at = now if final_seen_at is None else final_seen_at
                if not logs and now - max(final_seen_at, last_line_at) >= STRAGGLER_SECONDS:
                    echo(f"[{name}]")
                    return name
            elif name != last_state:
                echo(f"[{name}]")
                last_state = name
            if len(logs) < LOG_PAGE:
                time.sleep(POLL_SECONDS)


def collect_record(flow_run_id: UUID, root: Path) -> bool:
    """Cherry-pick the run's record commit from GitHub; False if it couldn't be applied."""
    branch = f"runs/{flow_run_id}"
    if not _git(root, "ls-remote", ORIGIN, f"refs/heads/{branch}"):
        return True
    _git(root, "fetch", "--quiet", ORIGIN, branch)
    if not _apply(root, "FETCH_HEAD"):
        return False
    _git(root, "push", "--quiet", ORIGIN, f":refs/heads/{branch}")
    return True


def collect_records(root: Path) -> list[str]:
    """Cherry-pick every run record still on GitHub; the branches it couldn't apply."""
    left: list[str] = []
    for line in _git(root, "ls-remote", ORIGIN, "refs/heads/runs/*").splitlines():
        commit, reference = line.split()
        branch = reference.removeprefix("refs/heads/")
        _git(root, "fetch", "--quiet", ORIGIN, branch)
        is_new = _git(root, "cherry", "HEAD", commit).startswith("+")
        if is_new and not _apply(root, commit):
            left.append(branch)
            continue
        _git(root, "push", "--quiet", ORIGIN, f":{reference}")
    return left


def _apply(root: Path, commit: str) -> bool:
    picked = subprocess.run(
        ["git", "cherry-pick", commit], cwd=root, capture_output=True, text=True, check=False
    )
    if picked.returncode == 0:
        return True
    subprocess.run(["git", "cherry-pick", "--abort"], cwd=root, capture_output=True, check=False)
    return False


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise SubmissionError(f"git {arguments[0]} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()
