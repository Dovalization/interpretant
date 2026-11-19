"""Run a command on the remote runner, or locally (`docs/design_compute_hosts.md`).

`python scripts/run.py [--where auto|local|remote] [--gpu [--allow-cpu]] [--pull PATH]...
[--push-outputs] [--name NAME] -- COMMAND...`

The remote runner is described in `hosts.local.yaml` (gitignored; see `hosts.example.yaml`). With
`auto` the command runs there if ssh answers, after the configured wake command if it does not, and
locally otherwise; a command marked `--gpu` refuses to fall back to the local machine unless
`--allow-cpu` is given. For a remote run the committed code is pushed to the runner's checkout
first, so the tree must have no uncommitted changes to tracked files.

On the chosen machine the command starts detached from the repository directory, its output going
to `logs/<time>-<name>.log` (where a run collector can follow it), after `dvc pull` of any `--pull`
paths. When it succeeds, the same machine commits `metrics/runs.jsonl` if the run added to it and,
with `--push-outputs`, runs `dvc push`; so a dropped connection loses nothing. Meanwhile this
script prints the log's new lines, and at the end pulls a remote run's commit back.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / "hosts.local.yaml"
POLL_SECONDS = 15
START_SECONDS = 10
Where = Literal["auto", "local", "remote"]


class RemoteRunner(BaseModel):
    """How to reach the remote runner and its checkout."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ssh: str = Field(min_length=1)
    path: str = Field(min_length=1)
    git_remote: str = Field(min_length=1)
    wake: str | None = None
    wake_timeout: int = Field(default=180, ge=0)
    gpu: bool = False


class Hosts(BaseModel):
    """The local configuration: a remote runner, or none."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    remote: RemoteRunner | None = None


class LaunchError(RuntimeError):
    """The command cannot run where it was asked to."""


def load_hosts(path: Path = CONFIG) -> Hosts:
    """The local configuration; no remote runner if the file does not exist."""
    if not path.exists():
        return Hosts()
    return Hosts.model_validate(yaml.safe_load(path.read_text()) or {})


def ssh_answers(alias: str) -> bool:
    """Whether the host answers ssh within a few seconds, without asking for anything."""
    probe = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", alias, "true"]
    return subprocess.run(probe, capture_output=True, check=False).returncode == 0


def run_wake(command: str) -> None:
    """Run the configured wake command."""
    subprocess.run(shlex.split(command), capture_output=True, check=False)


def choose(
    where: Where,
    hosts: Hosts,
    *,
    gpu: bool,
    allow_cpu: bool,
    answers: Callable[[str], bool] = ssh_answers,
    wake: Callable[[str], None] = run_wake,
    sleep: Callable[[float], None] = time.sleep,
) -> RemoteRunner | None:
    """The remote runner to use, or None for the local machine.

    Raises:
        LaunchError: if the remote runner is required but unavailable, or a GPU command would
            fall back to the local machine without ``allow_cpu``.
    """
    remote = hosts.remote
    if where != "local" and remote is not None and (remote.gpu or not gpu):
        if _reachable(remote, answers, wake, sleep):
            return remote
        if where == "remote":
            raise LaunchError(f"the remote runner {remote.ssh!r} does not answer")
    elif where == "remote":
        raise LaunchError("no remote runner with what this command needs in hosts.local.yaml")
    if gpu and not allow_cpu:
        raise LaunchError("a GPU command would run on the local CPU: pass --allow-cpu to do so")
    return None


def _reachable(
    remote: RemoteRunner,
    answers: Callable[[str], bool],
    wake: Callable[[str], None],
    sleep: Callable[[float], None],
) -> bool:
    if answers(remote.ssh):
        return True
    if remote.wake is None:
        return False
    wake(remote.wake)
    waited = 0
    while waited < remote.wake_timeout:
        sleep(5)
        waited += 5
        if answers(remote.ssh):
            return True
    return False


def run_script(command: list[str], log: str, *, pull: list[str], push_outputs: bool) -> str:
    """The shell script that runs the command on the chosen machine and records it there."""
    name = Path(log).stem
    steps = ['export PATH="$PWD/.venv/bin:$PATH"', "mkdir -p logs"]
    if pull:
        steps.append(f"dvc pull {shlex.join(pull)} >> {shlex.quote(log)} 2>&1")
    steps.append(f"{shlex.join(command)} > {shlex.quote(log)} 2>&1")
    steps.append("status=$?")
    recorded = (
        "git diff --quiet -- metrics/runs.jsonl || "
        f"git commit -q -m {shlex.quote(f'chore(metrics): record {name}')} -- metrics/runs.jsonl"
    )
    after = recorded + (" && dvc push" if push_outputs else "")
    steps.append(f'if [ "$status" -eq 0 ]; then {after}; fi >> {shlex.quote(log)} 2>&1')
    steps.append(f'echo "$status" > {shlex.quote(log)}.exit')
    return "; ".join(steps)


def _git(*arguments: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(REPOSITORY), *arguments], capture_output=True, text=True
    )
    if done.returncode != 0:
        raise LaunchError(f"git {' '.join(arguments)} failed: {done.stderr.strip()}")
    return done.stdout.strip()


def _on(remote: RemoteRunner | None, script: str) -> subprocess.CompletedProcess[str]:
    """Run a shell script in the chosen machine's checkout."""
    if remote is None:
        return subprocess.run(
            ["bash", "-c", script], cwd=REPOSITORY, capture_output=True, text=True
        )
    remote_script = f"cd {remote.path} && {script}"
    return subprocess.run(["ssh", remote.ssh, remote_script], capture_output=True, text=True)


def start(remote: RemoteRunner | None, script: str, root: Path = REPOSITORY) -> None:
    """Start the run script detached, in its own session, on the chosen machine.

    Locally Python starts the new session, as macOS has no ``setsid``; the remote runner is Linux.

    Raises:
        LaunchError: if ssh cannot start it on the remote runner.
    """
    if remote is None:
        subprocess.Popen(
            ["bash", "-c", script],
            cwd=root,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    detached = f"setsid nohup bash -c {shlex.quote(script)} > /dev/null 2>&1 < /dev/null &"
    started = _on(remote, detached)
    if started.returncode != 0:
        raise LaunchError(f"could not start the run: {started.stderr.strip()}")


def _log_appears(remote: RemoteRunner | None, log: str) -> bool:
    """Whether the run's log appears within a few seconds of the start."""
    for _ in range(START_SECONDS):
        if _on(remote, f"test -e {shlex.quote(log)}").returncode == 0:
            return True
        time.sleep(1)
    return False


def launch(arguments: argparse.Namespace) -> int:
    """Choose the machine, start the command there, follow it and bring its record back."""
    remote = choose(arguments.where, load_hosts(), gpu=arguments.gpu, allow_cpu=arguments.allow_cpu)
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if remote is not None:
        if _git("status", "--porcelain", "--untracked-files=no"):
            raise LaunchError("commit or stash the changes to tracked files before a remote run")
        _git("push", "-q", remote.git_remote, branch)
    name = arguments.name or Path(arguments.command[0]).stem
    log = f"logs/{datetime.now(UTC):%Y%m%dT%H%M%S}-{name}.log"
    script = run_script(
        arguments.command, log, pull=arguments.pull, push_outputs=arguments.push_outputs
    )
    start(remote, script)
    if not _log_appears(remote, log):
        raise LaunchError(f"the run did not start: no {log} after {START_SECONDS} s")
    sys.stdout.write(f"running on {'the remote runner' if remote else 'this machine'}: {log}\n")
    shown = 0
    while True:
        time.sleep(POLL_SECONDS)
        lines = _on(remote, f"cat {shlex.quote(log)} 2>/dev/null").stdout.splitlines()
        for line in lines[shown:]:
            sys.stdout.write(line + "\n")
        shown = len(lines)
        status = _on(remote, f"cat {shlex.quote(log)}.exit 2>/dev/null").stdout.strip()
        if status:
            break
    if remote is not None and status == "0":
        _git("pull", "-q", "--rebase", "--autostash", remote.git_remote, branch)
    sys.stdout.write(f"finished with exit status {status}\n")
    return int(status)


def main() -> None:
    """Parse the arguments and launch."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--where", choices=["auto", "local", "remote"], default="auto")
    parser.add_argument("--gpu", action="store_true", help="the command needs a GPU")
    parser.add_argument("--allow-cpu", action="store_true", help="let a GPU command run locally")
    parser.add_argument("--pull", action="append", default=[], help="dvc pull this path first")
    parser.add_argument("--push-outputs", action="store_true", help="dvc push after success")
    parser.add_argument("--name", help="a short name for the log file")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    if arguments.command[:1] == ["--"]:
        arguments.command = arguments.command[1:]
    if not arguments.command:
        parser.error("give the command after --")
    try:
        sys.exit(launch(arguments))
    except LaunchError as error:
        sys.exit(f"run.py: {error}")


if __name__ == "__main__":
    main()
