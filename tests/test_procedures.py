"""The matching audit's procedure: its steps in order, and the gate before scoring."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from click.testing import CliRunner
from prefect.states import Completed, Failed
from prefect.testing.utilities import prefect_test_harness

from interpretant.orchestration import commands, procedures, submit
from interpretant.orchestration.procedures import SCORER, ScoringGate, check_scoring

AUDIT = "data/review/audit"


def git(root: Path, *arguments: str) -> str:
    environment = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    completed = subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, text=True, check=True, env=environment
    )
    return completed.stdout.strip()


def commit_files(root: Path, files: dict[str, str]) -> str:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
        git(root, "add", name)
    git(root, "commit", "--quiet", "-m", f"write {', '.join(files)}")
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def history(tmp_path: Path) -> SimpleNamespace:
    """A relay's clone: the plan and scorer, the draw's record, an amendment, the scorer's commit.

    Everything but ``local`` is pushed to the relay's main branch.
    """
    relay = tmp_path / "relay.git"
    git(tmp_path, "init", "--quiet", "--bare", "--initial-branch=main", str(relay))
    root = tmp_path / "procedures"
    git(tmp_path, "clone", "--quiet", "--origin", "relay", str(relay), str(root))
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test")
    base = commit_files(root, {"docs/usages_audit.md": "the plan\n", SCORER: "score\n"})
    drawn = commit_files(root, {f"{AUDIT}/key.json": "{}\n", f"{AUDIT}/counts.json": "{}\n"})
    amendment = commit_files(root, {"docs/usages_audit.md": "the plan, amended\n"})
    calibrated = commit_files(root, {f"{AUDIT}/author.json": "{}\n"})
    scorer = commit_files(root, {SCORER: "score, as committed before the run\n"})
    git(root, "push", "--quiet", "relay", "HEAD:main")
    local = commit_files(root, {"notes.md": "not pushed\n"})
    git(root, "fetch", "--quiet", "relay")
    return SimpleNamespace(
        root=root,
        base=base,
        drawn=drawn,
        amendment=amendment,
        calibrated=calibrated,
        scorer=scorer,
        local=local,
    )


def check(history: SimpleNamespace, amendment: str, scorer: str, **options: Any) -> str:
    gate = ScoringGate(amendment_commit=amendment, scorer_commit=scorer)
    return check_scoring(history.root, gate, AUDIT, history.drawn, options.get("calibration"))


def test_the_scorer_runs_from_commits_that_keep_the_pre_registration(
    history: SimpleNamespace,
) -> None:
    assert check(history, history.amendment, history.scorer) == ""
    assert check(history, history.amendment, history.scorer, calibration=AUDIT) == ""


@pytest.mark.parametrize(
    ("amendment", "scorer", "reason"),
    [
        ("missing", "scorer", "is not on the relay"),
        ("amendment", "local", "on no branch of the relay"),
        ("amendment", "base", "doesn't hold the draw's key.json"),
        ("drawn", "scorer", "touches neither the audit's plan nor the predictions"),
        ("scorer", "amendment", "touches neither"),
    ],
)
def test_the_gate_refuses_commits_that_break_it(
    history: SimpleNamespace, amendment: str, scorer: str, reason: str
) -> None:
    named = vars(history) | {"missing": "0" * 40}

    assert reason in check(history, named[amendment], named[scorer])


def test_the_gate_refuses_an_amendment_made_after_the_scorer(history: SimpleNamespace) -> None:
    git(history.root, "checkout", "--quiet", "--detach", history.scorer)
    later = commit_files(history.root, {"docs/usages_audit.md": "amended after the scorer\n"})
    git(history.root, "push", "--quiet", "relay", "HEAD:refs/heads/later")
    git(history.root, "fetch", "--quiet", "relay")

    assert "doesn't come before" in check(history, later, history.scorer)


def test_the_gate_refuses_an_audit_already_scored_or_uncalibrated(
    history: SimpleNamespace,
) -> None:
    git(history.root, "checkout", "--quiet", "--detach", history.scorer)
    scored = commit_files(history.root, {f"{AUDIT}/results.json": "{}\n"})
    git(history.root, "push", "--quiet", "relay", "HEAD:refs/heads/scored")
    git(history.root, "fetch", "--quiet", "relay")

    assert "already holds" in check(history, history.amendment, scored)
    assert f"has no {AUDIT}_of_record/author.json" in check(
        history, history.amendment, history.scorer, calibration=f"{AUDIT}_of_record"
    )


# The steps, with each job run faked: what each starts, from which commit, under which key


@pytest.fixture(scope="module", autouse=True)
def _prefect_server() -> Iterator[None]:
    with prefect_test_harness():
        yield


class FakeRuns:
    """Job runs that complete at once, each record commit named after its step."""

    def __init__(self, failing: str = "") -> None:
        self.started: list[dict[str, Any]] = []
        self.failing = failing

    def run_deployment(self, name: str, **options: Any) -> SimpleNamespace:
        step = options["idempotency_key"].split(":")[1]
        self.started.append({"deployment": name, "step": step, **options})
        state = Failed() if step == self.failing else Completed()
        return SimpleNamespace(id=uuid4(), state=state)

    def hold_record(self, procedure: str, step: str, flow_run_id: str, commit: str) -> str:
        del procedure, flow_run_id, commit
        return f"record-of-{step}"


@pytest.fixture
def runs(monkeypatch: pytest.MonkeyPatch) -> FakeRuns:
    fake = FakeRuns()
    monkeypatch.setattr(procedures, "run_deployment", fake.run_deployment)
    monkeypatch.setattr(procedures, "hold_record", fake.hold_record)
    return fake


def answer_gates(monkeypatch: pytest.MonkeyPatch, refusals: list[str]) -> list[str]:
    """Answer each suspension with the same commits, refused for the given reasons first."""
    keys: list[str] = []

    def suspend(**options: Any) -> ScoringGate:
        keys.append(options["key"])
        return ScoringGate(amendment_commit="amendment", scorer_commit="scorer")

    verdicts = iter([*refusals, ""])
    monkeypatch.setattr(procedures, "suspend_flow_run", suspend)
    monkeypatch.setattr(procedures, "judge_gate", lambda *_: next(verdicts))
    return keys


def test_the_audit_runs_its_steps_in_order_from_the_right_commits(
    runs: FakeRuns, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys = answer_gates(monkeypatch, refusals=["not yet"])

    record = procedures.usages_audit(AUDIT, "start", calibration=AUDIT)

    assert record == "record-of-score"
    assert [(run["step"], run["deployment"]) for run in runs.started] == [
        ("draw", "draw-usages-audit/runner"),
        ("judge-A", "judge-claude-code/runner"),
        ("judge-B", "judge-claude-code/runner"),
        ("pending", "command/runner"),
        ("adjudication", "judge-claude-code/runner"),
        ("score", "command/runner"),
    ]
    commits = [run["job_variables"]["env"]["INTERPRETANT_COMMIT"] for run in runs.started]
    assert commits == ["start", *["record-of-draw"] * 4, "scorer"]
    assert keys == ["scoring-0", "scoring-1"]
    procedure = {run["idempotency_key"].split(":")[0] for run in runs.started}
    assert len(procedure) == 1
    score = runs.started[-1]
    assert score["parameters"]["command"][-2:] == ["--calibration", AUDIT]
    assert score["parameters"]["outputs"] == [f"{AUDIT}/results.json"]
    assert "stay-awake" not in score["tags"]
    assert all("stay-awake" in run["tags"] for run in runs.started[:-1])


def test_the_audit_stops_at_a_step_that_failed(
    runs: FakeRuns, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs.failing = "judge-B"
    answer_gates(monkeypatch, refusals=[])

    state = procedures.usages_audit(AUDIT, "start", return_state=True)

    assert state.is_failed()
    assert [run["step"] for run in runs.started] == ["draw", "judge-A", "judge-B"]


# Starting the procedure and answering its gate from the command line


def test_submitting_the_audit_sends_only_the_options_given(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[submit.Submission] = []
    monkeypatch.setattr(submit, "publish_commit", lambda _: "c0ffee" * 6 + "abcd")
    monkeypatch.setattr(submit, "start_run", lambda submission: started.append(submission) or "id")

    result = CliRunner().invoke(
        commands.submit, ["usages-audit", AUDIT, "--seed", "20261306", "--skip-calibration"]
    )

    assert result.exit_code == 0, result.output
    [submission] = started
    assert submission.deployment == "usages-audit/procedures"
    assert submission.parameters == {
        "audit": AUDIT,
        "commit": "c0ffee" * 6 + "abcd",
        "skip_calibration": True,
        "seed": 20261306,
    }
    assert submission.job_variables == {"env": {"INTERPRETANT_COMMIT": "c0ffee" * 6 + "abcd"}}


def test_resuming_sends_the_full_commits_and_reports_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[dict[str, str]] = []
    monkeypatch.setattr(submit, "resolve_commit", lambda _, revision: f"full-{revision}")
    monkeypatch.setattr(
        submit, "resume_and_follow", lambda _, run_input, **__: sent.append(run_input) or "Paused"
    )
    run_id = str(uuid4())

    result = CliRunner().invoke(
        commands.runs, ["resume", run_id, "--amendment", "a1", "--scorer", "s2"]
    )

    assert result.exit_code == 1
    assert sent == [{"amendment_commit": "full-a1", "scorer_commit": "full-s2"}]
    assert "refused" in result.output
