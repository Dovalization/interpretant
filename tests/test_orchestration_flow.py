"""The ``command`` flow end to end on Prefect's throwaway test server (a loopback, no network)."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from prefect.testing.utilities import prefect_test_harness

from interpretant.orchestration.flows import run_command

RECORDING = (
    "import pathlib; print('working'); "
    "pathlib.Path('metrics/runs.jsonl').open('a').write('{\"command\": \"probe\"}\\n')"
)


@pytest.fixture(scope="module", autouse=True)
def _prefect_server() -> Iterator[None]:
    with prefect_test_harness():
        yield


def git(root: Path, *arguments: str) -> str:
    environment = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    completed = subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, text=True, check=True, env=environment
    )
    return completed.stdout.strip()


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The runner's checkout, with a bare repository standing in for GitHub as its origin."""
    github, root = tmp_path / "github.git", tmp_path / "checkout"
    git(tmp_path, "init", "--quiet", "--bare", str(github))
    root.mkdir()
    git(root, "init", "--quiet")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test")
    git(root, "remote", "add", "origin", str(github))
    (root / "metrics").mkdir()
    (root / "metrics" / "runs.jsonl").write_text('{"command": "first"}\n')
    git(root, "add", ".")
    git(root, "commit", "--quiet", "-m", "first")
    monkeypatch.chdir(root)
    return root


def test_a_command_runs_logs_and_commits_its_record(checkout: Path) -> None:
    state = run_command([sys.executable, "-c", RECORDING], "probe", return_state=True)

    assert state.is_completed()
    assert git(checkout, "log", "-1", "--format=%s") == "chore(metrics): record probe"
    [branch] = git(checkout, "ls-remote", "origin", "refs/heads/runs/*").splitlines()
    assert branch.split()[0] == git(checkout, "rev-parse", "HEAD")
    [log] = (checkout / "logs").glob("*-probe.log")
    assert "working" in log.read_text()


def test_a_failing_command_fails_the_run_and_records_nothing(checkout: Path) -> None:
    state = run_command([sys.executable, "-c", "raise SystemExit(3)"], "broken", return_state=True)

    assert state.is_failed()
    assert git(checkout, "log", "-1", "--format=%s") == "first"


def test_a_command_finds_the_checkouts_own_tools(checkout: Path) -> None:
    tools = checkout / ".venv" / "bin"
    tools.mkdir(parents=True)
    (tools / "probe-tool").write_text("#!/bin/sh\necho found the checkout tool\n")
    (tools / "probe-tool").chmod(0o755)

    state = run_command(["probe-tool"], "tool", return_state=True)

    assert state.is_completed()
    [log] = (checkout / "logs").glob("*-tool.log")
    assert "found the checkout tool" in log.read_text()
