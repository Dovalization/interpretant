"""Procedures: runs of several jobs in order, with the author's pre-registration steps between them.

A procedure runs on a machine that stays up (``prefect.yaml``'s procedures deployments), from a
clone of the relay with only Prefect installed, so this module imports nothing but Prefect and the
standard library. Each step is a run of a job's runner deployment, started with an idempotency key
made of the procedure's run and the step: a procedure resumed after a gate, or rerun after a crash,
attaches to the steps it already started instead of starting them again.

``usages_audit`` is the matching audit (`docs/usages_audit.md`): the draw, the two judges, the
adjudicator's items and the adjudicator, all on the runner, where the items and verdicts stay.
Then it suspends until the author resumes it with two commits: the amendment, if any, written
before the results are read (the pre-registration), and the scorer, committed before it runs.
They are checked (``check_scoring``), and the scorer runs from that commit, its results coming back
with its record.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prefect import flow, get_run_logger, runtime, task
from prefect.client.schemas.objects import FlowRun
from prefect.deployments import run_deployment
from prefect.flow_runs import suspend_flow_run
from prefect.input import RunInput

RELAY = "relay"
# Long enough for the author's calibration and amendment; a gate left longer fails the procedure
GATE_SECONDS = 30 * 24 * 3600
CODEBOOK = "curated/judging/usages/prompt.md"
SCORER = "scripts/review/score_usages.py"
# Where an amendment of the audit's plan is written
PLAN_FILES = ("docs/usages_audit.md", "docs/predictions.md", "docs/predictions_cross_perplexity.md")


class ScoringGate(RunInput):
    """What the author resumes the audit with, before it is scored."""

    amendment_commit: str
    scorer_commit: str


class StepFailedError(RuntimeError):
    """A step's run did not complete, so the procedure stops there."""


@dataclass(frozen=True)
class Step:
    """One job run of a procedure: its name, the deployment, its parameters and its commit."""

    name: str
    deployment: str
    parameters: dict[str, Any]
    commit: str


@flow(name="usages-audit", persist_result=True)
def usages_audit(
    audit: str,
    commit: str,
    seed: int = 20261304,
    calibration: str | None = None,
    *,
    skip_calibration: bool = False,
    judge_model: str = "sonnet",
    adjudicator_model: str = "opus",
    effort: str = "high",
) -> str:
    """Draw, judge and adjudicate the audit on the runner, wait for the author, then score it.

    ``audit`` is the audit's folder, ``commit`` the commit its draw runs from. ``calibration`` is
    the folder holding the author's calibration verdicts (``author.json``), the audit's own unless a
    redraw keeps the calibration of record (``skip_calibration``); none scores without it. Returns
    the scorer's record commit on the relay.
    """
    procedure = str(runtime.flow_run.id)
    judging = {"codebook": CODEBOOK, "answers": "usages", "effort": effort}
    verdicts = f"{audit}/verdicts"
    draw = Step(
        "draw",
        "draw-usages-audit/runner",
        {"out": audit, "seed": seed, "skip_calibration": skip_calibration},
        commit,
    )
    drawn = run_step(procedure, draw)
    for judge in ("A", "B"):
        parameters = {"items": audit, "verdicts": verdicts, "name": f"judge_{judge}"}
        run_step(
            procedure,
            Step(
                f"judge-{judge}",
                "judge-claude-code/runner",
                parameters | judging | {"model": judge_model},
                drawn,
            ),
        )
    pending = ["python", SCORER, audit, "pending"]
    run_step(
        procedure,
        Step(
            "pending", "command/runner", {"command": pending, "name": "score-usages-pending"}, drawn
        ),
    )
    adjudication = {
        "items": f"{audit}/adjudicate.jsonl",
        "verdicts": verdicts,
        "name": "opus_adjudication",
    }
    run_step(
        procedure,
        Step(
            "adjudication",
            "judge-claude-code/runner",
            adjudication | judging | {"model": adjudicator_model},
            drawn,
        ),
    )
    scorer_commit = wait_for_scoring(audit, drawn, calibration)
    scoring = ["python", SCORER, audit, "score"]
    if calibration is not None:
        scoring += ["--calibration", calibration]
    score = Step(
        "score",
        "command/runner",
        {"command": scoring, "name": "score-usages", "outputs": [f"{audit}/results.json"]},
        scorer_commit,
    )
    return run_step(procedure, score, stay_awake=False)


def run_step(procedure: str, step: Step, *, stay_awake: bool = True) -> str:
    """Run the step's job on the runner, or attach to the run already started; its record commit.

    A step whose run failed stops the procedure. That run is retried on its own (Prefect's retry of
    the flow run keeps its id), then the procedure rerun attaches to it. The record commit is held
    on the relay under ``refs/procedures/``, as its ``runs/`` branch goes once the author collects
    it.
    """
    # Typed for both calling conventions; called from a sync flow, run_deployment returns the run
    flow_run: FlowRun = run_deployment(  # type: ignore[assignment]
        step.deployment,
        parameters=step.parameters,
        job_variables={"env": {"INTERPRETANT_COMMIT": step.commit}},
        tags=["interpretant", f"commit:{step.commit[:8]}", *(["stay-awake"] if stay_awake else [])],
        idempotency_key=f"{procedure}:{step.name}",
    )
    state = flow_run.state
    if state is None or not state.is_completed():
        name = state.name if state is not None else "unknown"
        raise StepFailedError(f"the {step.name} step ended {name} (flow run {flow_run.id})")
    return hold_record(procedure, step.name, str(flow_run.id), step.commit)


@task(name="hold-record")
def hold_record(procedure: str, step: str, flow_run_id: str, commit: str) -> str:
    """The step's record commit, held on the relay; ``commit`` when the step recorded nothing.

    A persisted result, so a resumed procedure reads the commit it held, not the relay's branches.
    """
    root = Path.cwd()
    held = f"refs/procedures/{procedure}/{step}"
    if existing := _git(root, "ls-remote", RELAY, held):
        return existing.split()[0]
    _git(root, "fetch", "--quiet", RELAY)
    branch = _git(root, "ls-remote", RELAY, f"refs/heads/runs/{flow_run_id}")
    record = branch.split()[0] if branch else commit
    _git(root, "push", "--quiet", RELAY, f"{record}:{held}")
    return record


def wait_for_scoring(audit: str, drawn: str, calibration: str | None) -> str:
    """Suspend until the author resumes with commits that pass ``check_scoring``; the scorer's.

    Each refusal is logged and the procedure suspends again, under a key of its own, so a resumed
    procedure replays the same answers and the same verdicts on them.
    """
    attempt = 0
    while True:
        gate = suspend_flow_run(
            wait_for_input=ScoringGate, timeout=GATE_SECONDS, key=f"scoring-{attempt}"
        )
        refusal = judge_gate(gate, audit, drawn, calibration, attempt)
        if not refusal:
            return gate.scorer_commit
        get_run_logger().warning("Refused the commits to score from: %s", refusal)
        attempt += 1


@task(name="check-gate")
def judge_gate(
    gate: ScoringGate, audit: str, drawn: str, calibration: str | None, attempt: int
) -> str:
    """Why the gate's commits can't be scored from, or an empty string when they can.

    A persisted result per attempt, so a resumed procedure keeps each verdict it gave.
    """
    del attempt
    root = Path.cwd()
    _git(root, "fetch", "--quiet", RELAY)
    return check_scoring(root, gate, audit, drawn, calibration)


def check_scoring(
    root: Path, gate: ScoringGate, audit: str, drawn: str, calibration: str | None
) -> str:
    """Why the audit can't be scored from the gate's commits, or an empty string when it can.

    The scorer's commit must be on a branch of the relay and hold the draw's key and counts
    unchanged (the draw's record, collected), the scorer, the author's calibration verdicts when
    scoring with them, and no results yet. The amendment must touch the audit's plan or the
    predictions and come before the scorer's commit.
    """
    amendment, scorer = gate.amendment_commit, gate.scorer_commit
    for name, commit in (("amendment", amendment), ("scorer", scorer)):
        if not _is_commit(root, commit):
            return f"the {name} commit {commit} is not on the relay"
    if not _git(root, "for-each-ref", "--contains", scorer, f"refs/remotes/{RELAY}"):
        return f"the scorer commit {scorer[:8]} is on no branch of the relay"
    for name in ("key.json", "counts.json"):
        if _blob(root, drawn, f"{audit}/{name}") != _blob(root, scorer, f"{audit}/{name}"):
            return f"the scorer commit doesn't hold the draw's {name}: collect the draw's record"
    required = [SCORER] + ([f"{calibration}/author.json"] if calibration is not None else [])
    for path in required:
        if not _blob(root, scorer, path):
            return f"the scorer commit has no {path}"
    if _blob(root, scorer, f"{audit}/results.json"):
        return f"the scorer commit already holds {audit}/results.json"
    touched = _git(root, "diff-tree", "--no-commit-id", "--name-only", "-r", "--root", amendment)
    if not set(touched.splitlines()) & set(PLAN_FILES):
        return f"the amendment {amendment[:8]} touches neither the audit's plan nor the predictions"
    if not _is_ancestor(root, amendment, scorer):
        return f"the amendment {amendment[:8]} doesn't come before the scorer commit {scorer[:8]}"
    return ""


def _is_commit(root: Path, commit: str) -> bool:
    return _run(root, "cat-file", "-e", f"{commit}^{{commit}}").returncode == 0


def _is_ancestor(root: Path, ancestor: str, commit: str) -> bool:
    return _run(root, "merge-base", "--is-ancestor", ancestor, commit).returncode == 0


def _blob(root: Path, commit: str, path: str) -> str:
    """The id of the file at ``path`` in ``commit``; empty if it has none."""
    completed = _run(root, "rev-parse", "--verify", "--quiet", f"{commit}:{path}")
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _git(root: Path, *arguments: str) -> str:
    completed = _run(root, *arguments)
    if completed.returncode != 0:
        raise RuntimeError(f"git {arguments[0]} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _run(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, text=True, check=False
    )
