"""The Prefect flows that run interpretant's jobs, one per job (``prefect.yaml`` deploys each).

``run_command`` is the escape hatch: any command, for one-off scripts and the scorers, whose
output is copied into the run's log file and the Prefect run. The jobs' own flows follow.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from prefect import flow, runtime, task
from prefect.logging.handlers import APILogHandler

from interpretant.orchestration.steps import (
    Role,
    capture_checkout_state,
    commit_record,
    open_run_log,
    pull_data,
    push_data,
)

# A command's lines go to the log file through standard output and to the Prefect run through this
# logger alone: Prefect's console handler would write each line to the file a second time.
LINE_LIMIT = 2048
_output = logging.getLogger("interpretant.orchestration.output")
_output.propagate = False
_output.setLevel(logging.INFO)
if not _output.handlers:
    _output.addHandler(APILogHandler())


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
            _output.info(line.rstrip("\n")[:LINE_LIMIT])
    if process.returncode != 0:
        raise CommandFailedError(f"{command[0]} exited with status {process.returncode}")
