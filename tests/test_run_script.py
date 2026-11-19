"""The run launcher (`scripts/run.py`): where a command runs, and how a run records itself."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]


def load_launcher() -> ModuleType:
    """The launcher script as a module."""
    spec = importlib.util.spec_from_file_location("run_launcher", REPOSITORY / "scripts" / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_launcher"] = module
    spec.loader.exec_module(module)
    return module


run = load_launcher()
REMOTE = run.RemoteRunner(ssh="box", path="~/repo", git_remote="box", wake="wake box", gpu=True)


def never_sleep(_seconds: float) -> None:
    """A sleep that returns at once."""


def test_load_hosts_reads_the_example_and_allows_no_file(tmp_path: Path) -> None:
    hosts = run.load_hosts(REPOSITORY / "hosts.example.yaml")
    assert hosts.remote is not None and hosts.remote.gpu
    assert run.load_hosts(tmp_path / "missing.yaml").remote is None


def test_choose_uses_the_remote_runner_when_it_answers() -> None:
    hosts = run.Hosts(remote=REMOTE)
    chosen = run.choose("auto", hosts, gpu=True, allow_cpu=False, answers=lambda _: True)
    assert chosen == REMOTE


def test_choose_wakes_the_runner_and_waits_for_it() -> None:
    replies = iter([False, False, True])
    woken: list[str] = []
    chosen = run.choose(
        "auto",
        run.Hosts(remote=REMOTE),
        gpu=False,
        allow_cpu=False,
        answers=lambda _: next(replies),
        wake=woken.append,
        sleep=never_sleep,
    )
    assert chosen == REMOTE
    assert woken == ["wake box"]


def test_choose_falls_back_to_the_local_machine_but_not_for_a_gpu_command() -> None:
    hosts = run.Hosts(remote=REMOTE.model_copy(update={"wake": None}))
    down = {"answers": lambda _: False, "sleep": never_sleep}
    assert run.choose("auto", hosts, gpu=False, allow_cpu=False, **down) is None
    with pytest.raises(run.LaunchError, match="--allow-cpu"):
        run.choose("auto", hosts, gpu=True, allow_cpu=False, **down)
    assert run.choose("auto", hosts, gpu=True, allow_cpu=True, **down) is None
    with pytest.raises(run.LaunchError, match="does not answer"):
        run.choose("remote", hosts, gpu=False, allow_cpu=False, **down)


def test_choose_never_sends_a_gpu_command_to_a_runner_without_one() -> None:
    hosts = run.Hosts(remote=REMOTE.model_copy(update={"gpu": False}))
    with pytest.raises(run.LaunchError):
        run.choose("remote", hosts, gpu=True, allow_cpu=False, answers=lambda _: True)
    assert run.choose("local", hosts, gpu=False, allow_cpu=False) is None


def git(folder: Path, *arguments: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(folder), *arguments], capture_output=True, text=True, check=True
    )
    return done.stdout


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A git checkout with a tracked, empty run record."""
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "metrics").mkdir()
    (tmp_path / "metrics" / "runs.jsonl").write_text("")
    git(tmp_path, "add", "metrics/runs.jsonl")
    git(tmp_path, "commit", "-q", "-m", "start")
    return tmp_path


def record_and_exit(code: int) -> list[str]:
    """A command that adds a line to the run record and exits with ``code``."""
    program = f"open('metrics/runs.jsonl', 'a').write('{{}}\\n'); raise SystemExit({code})"
    return [sys.executable, "-c", program]


def test_a_successful_run_commits_its_record_and_writes_its_exit_status(checkout: Path) -> None:
    script = run.run_script(record_and_exit(0), "logs/t-ok.log", pull=[], push_outputs=False)
    subprocess.run(["bash", "-c", script], cwd=checkout, check=False)
    assert (checkout / "logs" / "t-ok.log.exit").read_text().strip() == "0"
    assert "chore(metrics): record t-ok" in git(checkout, "log", "--format=%s", "-1")
    assert git(checkout, "status", "--porcelain", "--untracked-files=no") == ""


def test_a_failed_run_records_nothing(checkout: Path) -> None:
    script = run.run_script(record_and_exit(3), "logs/t-bad.log", pull=[], push_outputs=False)
    subprocess.run(["bash", "-c", script], cwd=checkout, check=False)
    assert (checkout / "logs" / "t-bad.log.exit").read_text().strip() == "3"
    assert git(checkout, "log", "--format=%s", "-1").strip() == "start"


def test_start_detaches_a_local_run_that_finishes_and_records_itself(checkout: Path) -> None:
    script = run.run_script(record_and_exit(0), "logs/t-start.log", pull=[], push_outputs=False)
    run.start(None, script, root=checkout)
    exit_file = checkout / "logs" / "t-start.log.exit"
    for _ in range(100):
        if exit_file.exists() and exit_file.read_text().strip():
            break
        time.sleep(0.1)
    assert exit_file.read_text().strip() == "0"
    assert "chore(metrics): record t-start" in git(checkout, "log", "--format=%s", "-1")
