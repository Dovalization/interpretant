"""The run log: each command's record, and the checkout state it names."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

from interpretant.catalog.config import CatalogConfig
from interpretant.pipelines.runs import checkout_state, record_run


def git(root: Path, *arguments: str) -> str:
    environment = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    completed = subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, text=True, check=True, env=environment
    )
    return completed.stdout.strip()


def repository_with_one_commit(root: Path) -> str:
    git(root, "init", "--quiet")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test")
    for name in ("code.py", "dvc.lock", "metrics/corpus.json"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("first\n")
    git(root, "add", ".")
    git(root, "commit", "--quiet", "-m", "first")
    return git(root, "rev-parse", "HEAD")


def test_record_run_appends_one_line_per_run(catalog: CatalogConfig, tmp_path: Path) -> None:
    stages = [("loading", 1.04), ("near-duplicates", 2.96)]
    record_run(
        catalog, command="build-corpus", stages=stages, counts={"abstracts": 5}, root=tmp_path
    )
    record_run(catalog, command="report-corpus", stages=[], counts={}, root=tmp_path)
    records = [json.loads(line) for line in catalog.run_log.path.read_text().splitlines()]
    assert [record["command"] for record in records] == ["build-corpus", "report-corpus"]
    first = records[0]
    assert first["seconds"] == 4.0
    assert first["stages"] == [
        {"name": "loading", "seconds": 1.0},
        {"name": "near-duplicates", "seconds": 3.0},
    ]
    assert first["counts"] == {"abstracts": 5}
    assert first["machine"] == f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPUs"
    assert platform.node() not in json.dumps(first)


def test_checkout_state_names_the_commit_and_ignores_the_run_outputs(tmp_path: Path) -> None:
    commit = repository_with_one_commit(tmp_path)
    (tmp_path / "dvc.lock").write_text("changed by the run\n")
    (tmp_path / "metrics" / "corpus.json").write_text("changed by the run\n")
    assert checkout_state(tmp_path) == (commit, False)
    (tmp_path / "code.py").write_text("edited\n")
    assert checkout_state(tmp_path) == (commit, True)


def test_checkout_state_outside_a_repository_is_unknown(tmp_path: Path) -> None:
    assert checkout_state(tmp_path) == (None, False)


def test_record_run_keeps_the_checkout_the_run_started_from(
    catalog: CatalogConfig, tmp_path: Path
) -> None:
    repository_with_one_commit(tmp_path)
    (tmp_path / "code.py").write_text("edited after the run started\n")

    record_run(
        catalog,
        command="build-corpus",
        stages=[],
        counts={},
        root=tmp_path,
        checkout=("abc", False),
    )

    [record] = [json.loads(line) for line in catalog.run_log.path.read_text().splitlines()]
    assert (record["commit"], record["uncommitted_changes"]) == ("abc", False)


def test_run_records_made_on_different_commits_combine(tmp_path: Path) -> None:
    """The repository's own .gitattributes lets a run's record commit be cherry-picked onto a
    branch that recorded another run meanwhile."""
    repository_with_one_commit(tmp_path)
    attributes = Path(__file__).resolve().parents[1] / ".gitattributes"
    (tmp_path / ".gitattributes").write_text(attributes.read_text())
    (tmp_path / "metrics" / "runs.jsonl").write_text('{"command": "first"}\n')
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "--quiet", "-m", "run log")
    base = git(tmp_path, "rev-parse", "HEAD")
    for command in ("on-the-runner", "on-this-machine"):
        git(tmp_path, "checkout", "--quiet", "-B", command, base)
        with (tmp_path / "metrics" / "runs.jsonl").open("a") as log:
            log.write(f'{{"command": "{command}"}}\n')
        git(tmp_path, "commit", "--quiet", "-am", command)

    git(tmp_path, "cherry-pick", "on-the-runner")

    lines = (tmp_path / "metrics" / "runs.jsonl").read_text().splitlines()
    assert sorted(lines) == sorted(
        ['{"command": "first"}', '{"command": "on-this-machine"}', '{"command": "on-the-runner"}']
    )
