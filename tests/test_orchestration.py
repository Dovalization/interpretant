"""Running jobs in Prefect: the runner's checkout, the record commit, progress and submission.

Real git in temporary folders, with a bare repository standing in for the relay; Prefect's own calls
are replaced, so no server is needed (``test_orchestration_flow.py`` runs a flow on a test server).
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
from collections.abc import Iterator
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
import yaml
from prefect.states import Completed, Running

from interpretant.orchestration import progress as progress_module
from interpretant.orchestration import steps, submit
from interpretant.orchestration.progress import PrefectProgress, format_artifact_key

REPOSITORY = Path(__file__).resolve().parents[1]


def git(root: Path, *arguments: str) -> str:
    environment = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    completed = subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, text=True, check=True, env=environment
    )
    return completed.stdout.strip()


def commit_file(root: Path, name: str, text: str) -> str:
    (root / name).parent.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(text)
    git(root, "add", name)
    git(root, "commit", "--quiet", "-m", f"write {name}")
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def relay(tmp_path: Path) -> Path:
    """A bare relay holding the repository's .gitattributes and a run log."""
    bare = tmp_path / "relay.git"
    git(tmp_path, "init", "--quiet", "--bare", "--initial-branch=main", str(bare))
    seed = tmp_path / "seed"
    git(tmp_path, "clone", "--quiet", str(bare), str(seed))
    git(seed, "config", "user.email", "test@example.com")
    git(seed, "config", "user.name", "Test")
    commit_file(seed, ".gitattributes", (REPOSITORY / ".gitattributes").read_text())
    commit_file(seed, "metrics/runs.jsonl", '{"command": "first"}\n')
    git(seed, "push", "--quiet", "origin", "HEAD:main")
    return bare


def clone(relay: Path, path: Path) -> Path:
    git(path.parent, "clone", "--quiet", "--origin", "relay", str(relay), str(path))
    git(path, "config", "user.email", "test@example.com")
    git(path, "config", "user.name", "Test")
    return path


# The runner's checkout, as prefect.yaml's pull step runs it: each line in its own shell


def runner_script() -> list[str]:
    deployments = yaml.safe_load((REPOSITORY / "prefect.yaml").read_text())
    pull = deployments["definitions"]["roles"]["runner"]["pull"]
    [step] = [step for step in pull if "prefect.deployments.steps.run_shell_script" in step]
    script: str = step["prefect.deployments.steps.run_shell_script"]["script"]
    return [line for line in script.splitlines() if line.strip()]


def run_script(checkout: Path, *, commit: str, uv_status: int = 0) -> tuple[int, str]:
    fake_bin = checkout.parent / "fake-bin"
    fake_bin.mkdir(exist_ok=True)
    (fake_bin / "uv").write_text(f"#!/bin/sh\nexit {uv_status}\n")
    (fake_bin / "uv").chmod(0o755)
    environment = {
        **{name: value for name, value in os.environ.items() if not name.startswith("GIT_")},
        "INTERPRETANT_COMMIT": commit,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    for line in runner_script():
        completed = subprocess.run(
            line, shell=True, cwd=checkout, capture_output=True, text=True, env=environment
        )
        if completed.returncode != 0:
            return completed.returncode, completed.stderr
    return 0, ""


def test_the_runner_checks_out_the_submitted_commit(relay: Path, tmp_path: Path) -> None:
    runner = clone(relay, tmp_path / "runner")
    mac = clone(relay, tmp_path / "mac")
    submitted = commit_file(mac, "code.py", "new\n")
    git(mac, "push", "--quiet", "relay", "HEAD:refs/heads/submitted")

    assert run_script(runner, commit=submitted) == (0, "")
    assert git(runner, "rev-parse", "HEAD") == submitted


def test_the_runner_refuses_a_checkout_with_changes(relay: Path, tmp_path: Path) -> None:
    runner = clone(relay, tmp_path / "runner")
    (runner / "metrics" / "runs.jsonl").write_text("someone's edit\n")

    status, message = run_script(runner, commit="HEAD")

    assert status != 0 and "changes to tracked files" in message


def test_the_runner_refuses_a_commit_no_branch_holds(relay: Path, tmp_path: Path) -> None:
    runner = clone(relay, tmp_path / "runner")
    git(runner, "checkout", "--quiet", "--detach")
    commit_file(runner, "notes.md", "work done by hand on the runner\n")

    status, message = run_script(runner, commit="relay/main")

    assert status != 0 and "no branch holds" in message


def test_the_runner_stops_when_its_environment_is_out_of_date(relay: Path, tmp_path: Path) -> None:
    runner = clone(relay, tmp_path / "runner")

    status, message = run_script(runner, commit="relay/main", uv_status=1)

    assert status != 0 and "uv sync --locked" in message


# The record commit


@pytest.fixture
def no_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(steps, "concurrency", lambda *_, **__: nullcontext())
    monkeypatch.setattr(steps, "INDEX_LOCK_WAIT_SECONDS", 0)


def append_record(checkout: Path, command: str) -> None:
    with (checkout / "metrics" / "runs.jsonl").open("a") as log:
        log.write(f'{{"command": "{command}"}}\n')


@pytest.mark.usefixtures("no_limits")
def test_a_runner_record_is_committed_and_pushed_to_its_own_branch(
    relay: Path, tmp_path: Path
) -> None:
    runner = clone(relay, tmp_path / "runner")
    append_record(runner, "probe")

    commit = steps.commit_record.fn("runner", "probe", "run-1", root=runner)

    assert git(runner, "log", "-1", "--format=%s") == "chore(metrics): record probe"
    assert git(runner, "ls-remote", "relay", "refs/heads/runs/run-1").split()[0] == commit


@pytest.mark.usefixtures("no_limits")
def test_each_run_sharing_a_checkout_commits_its_own_record(relay: Path, tmp_path: Path) -> None:
    local = clone(relay, tmp_path / "local")

    for command in ("first-run", "second-run"):
        steps.commit_record.fn(
            "local",
            command,
            command,
            root=local,
            write_record=partial(append_record, local, command),
        )

    added = git(local, "log", "-2", "--format=%s", "-p", "--", "metrics/runs.jsonl")
    assert re.findall(r"record (\S+)|\+\{\"command\": \"(\S+)\"\}", added) == [
        ("second-run", ""),
        ("", "second-run"),
        ("first-run", ""),
        ("", "first-run"),
    ]


@pytest.mark.usefixtures("no_limits")
def test_a_run_that_recorded_nothing_commits_nothing(relay: Path, tmp_path: Path) -> None:
    runner = clone(relay, tmp_path / "runner")

    assert steps.commit_record.fn("runner", "probe", "run-1", root=runner) is None


@pytest.mark.usefixtures("no_limits")
def test_a_record_that_cant_be_committed_fails_the_run(relay: Path, tmp_path: Path) -> None:
    local = clone(relay, tmp_path / "local")
    append_record(local, "probe")
    (local / ".git" / "index.lock").write_text("")

    with pytest.raises(steps.RecordError, match=r"index\.lock"):
        steps.commit_record.fn("local", "probe", "run-1", root=local)


@pytest.mark.usefixtures("no_limits")
def test_a_local_record_waits_out_a_merge_in_progress(relay: Path, tmp_path: Path) -> None:
    local = clone(relay, tmp_path / "local")
    append_record(local, "probe")
    (local / ".git" / "MERGE_HEAD").write_text(git(local, "rev-parse", "HEAD"))

    assert steps.commit_record.fn("local", "probe", "run-1", root=local) is None
    assert git(local, "status", "--porcelain") == "M metrics/runs.jsonl"


# The run's log file


def test_the_run_log_takes_this_process_and_its_children(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    with steps.open_run_log("probe", root=tmp_path):
        os.write(1, b"from this process\n")
        subprocess.run(["echo", "from a child"], check=True)
    os.write(1, b"after the run\n")

    [log] = (tmp_path / "logs").glob("*-probe.log")
    assert log.read_text().splitlines() == ["from this process", "from a child"]
    assert "after the run" in capfd.readouterr().out


# Progress published as an artifact


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def artifacts(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        progress_module,
        "create_progress_artifact",
        lambda **arguments: calls.append({"create": arguments}) or uuid4(),
    )
    monkeypatch.setattr(
        progress_module,
        "update_progress_artifact",
        lambda **arguments: calls.append({"update": arguments}),
    )
    yield calls


def test_progress_is_published_at_most_every_interval(artifacts: list[dict[str, Any]]) -> None:
    clock = FakeClock()
    progress = PrefectProgress("build-usages", interval=30, clock=clock)

    progress.stage("annotating", total=100)
    clock.now = 10
    progress.advance(10)
    clock.now = 31
    progress.advance(20)
    progress.close()

    updates = [call["update"] for call in artifacts if "update" in call]
    assert [update["progress"] for update in updates] == [30.0, 100.0]
    assert artifacts[0]["create"]["key"] == "build-usages-annotating"


def test_a_stage_without_a_total_shows_done_when_the_next_starts(
    artifacts: list[dict[str, Any]],
) -> None:
    clock = FakeClock()
    progress = PrefectProgress("report-corpus", interval=30, clock=clock)

    progress.stage("loading the abstracts table")
    clock.now = 4
    progress.stage("writing the report")

    [update] = [call["update"] for call in artifacts if "update" in call]
    assert update["progress"] == 100.0
    assert update["description"] == "loading the abstracts table: done in 4 s"


def test_progress_from_a_worker_thread_is_counted(artifacts: list[dict[str, Any]]) -> None:
    clock = FakeClock()
    progress = PrefectProgress("encode", interval=30, clock=clock)
    progress.stage("encoding", total=4)

    workers = [threading.Thread(target=progress.advance) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    progress.close()

    assert artifacts[-1]["update"]["progress"] == 100.0


def test_an_artifact_key_keeps_to_what_prefect_accepts() -> None:
    assert format_artifact_key("judge-api", "Judging Batch #2") == "judge-api-judging-batch-2"


# Submitting and collecting


def test_a_runner_submission_carries_its_commit_and_tags() -> None:
    submission = submit.build_submission(
        "command", "runner", {"command": ["true"]}, commit="abcdef0123456789", stay_awake=True
    )

    assert submission.deployment == "command/runner"
    assert submission.job_variables == {"env": {"INTERPRETANT_COMMIT": "abcdef0123456789"}}
    assert submission.tags == ["interpretant", "commit:abcdef01", "stay-awake"]


def test_a_local_submission_needs_no_commit_in_its_environment() -> None:
    submission = submit.build_submission(
        "command", "local", {"command": ["true"]}, commit="abc", stay_awake=False
    )

    assert (submission.deployment, submission.job_variables) == ("command/local", {})


class LateLinesClient:
    """A Prefect client whose run turns final before its last log lines reach the server."""

    def __init__(self) -> None:
        self.polls = 0
        # Each poll: the run's state and every line the server holds by then
        self.timeline = [
            (Running(), ["judging: started"]),
            (Completed(), ["judging: started"]),
            (Completed(), ["judging: started", "the report", "judging: done"]),
        ]

    def __enter__(self) -> LateLinesClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read_flow_run(self, flow_run_id: object) -> SimpleNamespace:
        self.polls += 1
        return SimpleNamespace(state=self.timeline[min(self.polls, 3) - 1][0])

    def read_logs(self, *, offset: int, **query: object) -> list[SimpleNamespace]:
        lines = self.timeline[min(self.polls, 3) - 1][1]
        return [SimpleNamespace(message=line) for line in lines[offset:]]


def test_following_a_run_waits_for_lines_sent_before_it_ended(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    monkeypatch.setattr(submit, "get_client", lambda **options: LateLinesClient())
    monkeypatch.setattr(submit, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep))
    shown: list[str] = []

    final = submit.follow_run(uuid4(), echo=shown.append)

    assert final == "Completed"
    assert shown == ["judging: started", "[Running]", "the report", "judging: done", "[Completed]"]


def test_publishing_refuses_uncommitted_changes(relay: Path, tmp_path: Path) -> None:
    mac = clone(relay, tmp_path / "mac")
    (mac / "metrics" / "runs.jsonl").write_text("edited\n")

    with pytest.raises(submit.SubmissionError, match="commit or stash"):
        submit.publish_commit(mac)


def test_publishing_puts_the_commit_on_the_relay(relay: Path, tmp_path: Path) -> None:
    mac = clone(relay, tmp_path / "mac")
    commit = commit_file(mac, "code.py", "new\n")

    assert submit.publish_commit(mac) == commit
    assert git(mac, "ls-remote", "relay", "refs/heads/submitted").split()[0] == commit


@pytest.mark.usefixtures("no_limits")
def test_a_run_record_comes_back_and_its_branch_goes(relay: Path, tmp_path: Path) -> None:
    runner = clone(relay, tmp_path / "runner")
    mac = clone(relay, tmp_path / "mac")
    append_record(runner, "on-the-runner")
    run_id = uuid4()
    steps.commit_record.fn("runner", "on-the-runner", str(run_id), root=runner)

    assert submit.collect_record(run_id, mac)
    assert '{"command": "on-the-runner"}' in (mac / "metrics" / "runs.jsonl").read_text()
    assert git(mac, "ls-remote", "relay", f"refs/heads/runs/{run_id}") == ""


@pytest.mark.usefixtures("no_limits")
def test_collecting_applies_every_record_left_on_the_relay(relay: Path, tmp_path: Path) -> None:
    runner = clone(relay, tmp_path / "runner")
    mac = clone(relay, tmp_path / "mac")
    for command in ("first-run", "second-run"):
        git(runner, "checkout", "--quiet", "--detach", "relay/main")
        append_record(runner, command)
        steps.commit_record.fn("runner", command, command, root=runner)

    assert submit.collect_records(mac) == []
    log = (mac / "metrics" / "runs.jsonl").read_text()
    assert '"first-run"' in log and '"second-run"' in log
    assert git(mac, "ls-remote", "relay", "refs/heads/runs/*") == ""
