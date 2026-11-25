"""The jobs' flows: their registry against prefect.yaml, dvc.yaml and the commands; one run.

The end-to-end tests run on Prefect's throwaway test server (a loopback, no network).
"""

from __future__ import annotations

import inspect
import json
import os
import re
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import click
import pytest
import yaml
from click.testing import CliRunner
from prefect import flow
from prefect.client.orchestration import get_client
from prefect.client.schemas.filters import ArtifactFilter, ArtifactFilterType
from prefect.testing.utilities import prefect_test_harness
from rich.console import Console

from interpretant.catalog.config import CatalogConfig
from interpretant.cli.judge_claude_code import judge_claude_code_command
from interpretant.orchestration import commands, flows, jobs
from interpretant.orchestration.jobs import run_job
from interpretant.orchestration.registry import JOBS, Job
from interpretant.pipelines.progress import Progress

REPOSITORY = Path(__file__).resolve().parents[1]
ADDED_PARAMETERS = {"pull", "save_outputs"}


def read_deployments() -> list[dict[str, Any]]:
    deployments: list[dict[str, Any]] = yaml.safe_load((REPOSITORY / "prefect.yaml").read_text())[
        "deployments"
    ]
    return deployments


def get_flow(job: Job) -> Any:
    return getattr(flows, job.name.replace("-", "_"))


def get_command(job: Job) -> click.Command:
    [command] = [command for command in commands.JOB_COMMANDS if command.name == job.name]
    return command


@pytest.mark.parametrize("job", JOBS.values(), ids=lambda job: job.name)
def test_each_job_deploys_on_the_runner(job: Job) -> None:
    entrypoint = f"src/interpretant/orchestration/flows.py:{job.name.replace('-', '_')}"
    names = [
        deployment["name"]
        for deployment in read_deployments()
        if deployment["entrypoint"] == entrypoint
    ]

    assert names == ["runner"]


@pytest.mark.parametrize("job", JOBS.values(), ids=lambda job: job.name)
def test_each_flow_takes_its_registry_settings(job: Job) -> None:
    job_flow = get_flow(job)

    assert job_flow.name == job.name
    assert job_flow.timeout_seconds == job.timeout_seconds
    assert job_flow.retries == job.retries
    assert job_flow.retry_delay_seconds == job.retry_delay_seconds


@pytest.mark.parametrize("job", JOBS.values(), ids=lambda job: job.name)
def test_each_flow_takes_its_commands_options_with_their_defaults(job: Job) -> None:
    signature = inspect.signature(get_flow(job).fn)
    flow_defaults = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if name not in ADDED_PARAMETERS
    }
    command_defaults = {
        param.name: get_flow_default(param)
        for param in get_command(job).params
        if param.name != "catalog_file"
    }

    assert flow_defaults == command_defaults
    expected_added = (
        ADDED_PARAMETERS if job.stage is not None else ADDED_PARAMETERS - {"save_outputs"}
    )
    assert set(signature.parameters) - set(flow_defaults) == expected_added


def get_flow_default(param: click.Parameter) -> object:
    """The default a flow's parameter takes for a command's option: none for a required one."""
    if param.required:
        return inspect.Parameter.empty
    return () if getattr(param, "multiple", False) else param.default


def test_every_job_has_a_submission_and_nothing_else_does() -> None:
    assert {command.name for command in commands.JOB_COMMANDS} == set(JOBS)


def resolve_catalog_paths(text: str) -> str:
    catalog = yaml.safe_load((REPOSITORY / "catalog.yaml").read_text())
    return re.sub(r"\$\{(\w+)\.path\}", lambda match: catalog[match[1]]["path"], text)


def list_tracked_uncached_outputs(stage: dict[str, Any]) -> set[str]:
    """The stage's outputs DVC leaves out of its store and git doesn't ignore: git tracks them."""
    uncached = set()
    for entry in [*stage.get("outs", []), *stage.get("metrics", [])]:
        if isinstance(entry, dict):
            [(path, options)] = entry.items()
            if options.get("cache") is False:
                uncached.add(resolve_catalog_paths(path))
    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", *uncached],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    return uncached - set(ignored)


@pytest.mark.parametrize(
    "job", [job for job in JOBS.values() if job.stage is not None], ids=lambda job: job.name
)
def test_each_stage_commits_the_outputs_git_tracks(job: Job) -> None:
    stage = yaml.safe_load((REPOSITORY / "dvc.yaml").read_text())["stages"][job.stage]

    assert stage["cmd"] == f"interpretant {job.name}"
    assert set(job.tracked_outputs) == list_tracked_uncached_outputs(stage)


def test_a_submission_sends_the_flows_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        commands,
        "_submit_and_follow",
        lambda options, flow_name, parameters: sent.append((flow_name, parameters)),
    )

    result = CliRunner().invoke(
        commands.submit,
        ["build-usages", "--field", "ai", "--decade", "2010", "--resume", "--save-outputs"],
    )

    assert result.exit_code == 0, result.output
    [(flow_name, parameters)] = sent
    assert flow_name == "build-usages"
    validated = flows.build_usages.validate_parameters(parameters)
    assert validated["fields"] == ("ai",)
    assert validated["decades"] == (2010,)
    assert validated["resume"] is True
    assert validated["save_outputs"] is True


def test_a_runner_submission_takes_a_path_that_exists_only_on_the_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted: list[dict[str, Any]] = []
    monkeypatch.setattr(
        commands, "_submit_and_follow", lambda _, __, parameters: submitted.append(parameters)
    )
    codebook = REPOSITORY / "curated" / "judging" / "usages" / "prompt.md"

    result = CliRunner().invoke(
        commands.submit,
        [
            *["judge-claude-code", "data/review/only_on_the_runner/items"],
            *["data/review/only_on_the_runner/verdicts", "--name", "judge_A"],
            *f"--codebook {codebook} --answers usages --model m --effort low".split(),
        ],
    )

    assert result.exit_code == 0, result.output
    assert submitted[0]["items"] == "data/review/only_on_the_runner/items"
    assert judge_claude_code_command.params[0].type.exists  # type: ignore[attr-defined]


def test_a_submitted_path_is_relative_to_the_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(
        commands, "_submit_and_follow", lambda _, __, parameters: sent.append(parameters)
    )
    monkeypatch.chdir(REPOSITORY / "curated")

    result = CliRunner().invoke(
        commands.submit,
        [
            *[
                "judge-claude-code",
                "judging",
                "../data/verdicts",
                "--name",
                "A",
            ],
            *[
                "--codebook",
                "judging/usages/prompt.md",
                "--answers",
                "usages",
                "--model",
                "m",
                "--effort",
                "low",
            ],
        ],
    )

    assert result.exit_code == 0, result.output
    [parameters] = sent
    assert parameters["items"] == "curated/judging"
    assert parameters["verdicts"] == "data/verdicts"
    assert parameters["codebook"] == "curated/judging/usages/prompt.md"


def test_a_job_without_a_stage_has_no_save_outputs_option() -> None:
    result = CliRunner().invoke(commands.submit, ["fetch-metadata", "--save-outputs"])

    assert result.exit_code != 0
    assert "No such option" in result.output


# End to end, with a job whose work prints a table, reports progress and rewrites a tracked file

PROBE = Job("probe", 60, tracked_outputs=("metrics/probe.json",))


def run_probe(catalog: CatalogConfig, progress: Progress, console: Console) -> dict[str, int]:
    progress.stage("probing", total=3)
    progress.advance(3)
    (catalog.run_log.path.parent / "probe.json").write_text('{"probed": 3}\n')
    console.print("probe table: 3 probed")
    return {"probed": 3}


@flow(name="probe")
def probe_flow(*, save_outputs: bool = False) -> None:
    run_job(PROBE, run_probe, pull=None, save_outputs=save_outputs)


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
    shutil.copy(REPOSITORY / "catalog.yaml", root / "catalog.yaml")
    (root / "metrics").mkdir()
    (root / "metrics" / "runs.jsonl").write_text('{"command": "first"}\n')
    (root / "metrics" / "probe.json").write_text('{"probed": 0}\n')
    git(root, "add", ".")
    git(root, "commit", "--quiet", "-m", "first")
    monkeypatch.chdir(root)
    return root


def test_a_job_runs_logs_and_commits_its_record_with_its_outputs(checkout: Path) -> None:
    state = probe_flow(return_state=True)

    assert state.is_completed()
    assert git(checkout, "log", "-1", "--format=%s") == "chore(metrics): record probe"
    assert git(checkout, "status", "--porcelain") == "?? logs/"
    record = json.loads((checkout / "metrics" / "runs.jsonl").read_text().splitlines()[-1])
    assert record["command"] == "probe"
    assert record["counts"] == {"probed": 3}
    assert record["stages"][0]["name"] == "probing"
    [log] = (checkout / "logs").glob("*-probe.log")
    assert "probe table: 3 probed" in log.read_text()


@flow(name="probe-with-outputs")
def probe_with_outputs_flow() -> None:
    run_job(PROBE, run_probe, pull=None, save_outputs=False, outputs=[Path("audit/key.json")])


def test_a_runs_outputs_are_committed_with_its_record(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths: list[str] = []
    monkeypatch.setattr(jobs, "commit_record", lambda *arguments, **_: paths.extend(arguments[2]))

    probe_with_outputs_flow()

    assert "audit/key.json" in paths
    assert "metrics/probe.json" in paths


def test_a_jobs_progress_is_published_as_an_artifact(checkout: Path) -> None:
    probe_flow()

    with get_client(sync_client=True) as client:
        artifacts = client.read_artifacts(
            artifact_filter=ArtifactFilter(type=ArtifactFilterType(any_=["progress"]))
        )
    assert any(artifact.key == "probe-probing" and artifact.data == 100.0 for artifact in artifacts)


def test_saving_the_outputs_of_a_job_without_a_stage_fails_before_running(checkout: Path) -> None:
    state = probe_flow(save_outputs=True, return_state=True)

    assert state.is_failed()
    assert not (checkout / "logs").exists()
    assert git(checkout, "log", "-1", "--format=%s") == "first"
