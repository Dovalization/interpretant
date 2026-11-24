"""The ``command`` flow end to end on Prefect's throwaway test server (a loopback, no network)."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from prefect.client.orchestration import get_client
from prefect.client.schemas.actions import GlobalConcurrencyLimitCreate
from prefect.testing.utilities import prefect_test_harness

from interpretant.orchestration.flows import run_command

RECORDING = (
    "import pathlib; print('working'); "
    "pathlib.Path('metrics/runs.jsonl').open('a').write('{\"command\": \"probe\"}\\n')"
)


@pytest.fixture(scope="module", autouse=True)
def _prefect_server() -> Iterator[None]:
    with prefect_test_harness():
        with get_client(sync_client=True) as client:
            for name in ("interpretant-git-local", "interpretant-dvc-local"):
                client.create_global_concurrency_limit(
                    GlobalConcurrencyLimitCreate(name=name, limit=1)
                )
        yield


def git(root: Path, *arguments: str) -> str:
    environment = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    completed = subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, text=True, check=True, env=environment
    )
    return completed.stdout.strip()


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    git(tmp_path, "init", "--quiet")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "metrics").mkdir()
    (tmp_path / "metrics" / "runs.jsonl").write_text('{"command": "first"}\n')
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "--quiet", "-m", "first")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_a_command_runs_logs_and_commits_its_record(checkout: Path) -> None:
    state = run_command([sys.executable, "-c", RECORDING], "probe", return_state=True)

    assert state.is_completed()
    assert git(checkout, "log", "-1", "--format=%s") == "chore(metrics): record probe"
    [log] = (checkout / "logs").glob("*-probe.log")
    assert "working" in log.read_text()


def test_a_failing_command_fails_the_run_and_records_nothing(checkout: Path) -> None:
    state = run_command([sys.executable, "-c", "raise SystemExit(3)"], "broken", return_state=True)

    assert state.is_failed()
    assert git(checkout, "log", "-1", "--format=%s") == "first"


def test_a_record_without_its_lock_fails_rather_than_racing(checkout: Path) -> None:
    """The runner's git limit doesn't exist on this server; the commit refuses to go ahead."""
    state = run_command(
        [sys.executable, "-c", RECORDING], "unlocked", role="runner", return_state=True
    )

    assert state.is_failed()
    assert git(checkout, "log", "-1", "--format=%s") == "first"
