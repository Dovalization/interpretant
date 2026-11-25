"""The pre-registration gate before an audit is scored, on real git in a temporary folder."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from interpretant.pipelines.preregistration import SCORER, PreregistrationError, check_scoring

AUDIT = Path("data/review/audit")


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
    """The plan and scorer, the draw's record, an amendment, the calibration, then a code commit."""
    root = tmp_path / "checkout"
    root.mkdir()
    git(root, "init", "--quiet")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test")
    base = commit_files(root, {"docs/usages_audit.md": "the plan\n", SCORER: "score\n"})
    drawn = commit_files(root, {f"{AUDIT}/key.json": "{}\n", f"{AUDIT}/counts.json": "{}\n"})
    amendment = commit_files(root, {"docs/usages_audit.md": "the plan, amended\n"})
    commit_files(root, {f"{AUDIT}/author.json": "{}\n"})
    commit_files(root, {SCORER: "score, as committed before the run\n"})
    return SimpleNamespace(root=root, base=base, drawn=drawn, amendment=amendment)


def test_an_audit_is_scored_from_a_checkout_that_keeps_the_pre_registration(
    history: SimpleNamespace,
) -> None:
    check_scoring(history.root, AUDIT, history.amendment, None)
    check_scoring(history.root, AUDIT, history.amendment, AUDIT)


@pytest.mark.parametrize(
    ("amendment", "reason"),
    [
        ("0" * 40, "is not a commit"),
        ("drawn", "touches neither the audit's plan nor the predictions"),
    ],
)
def test_the_gate_refuses_an_amendment_that_isnt_one(
    history: SimpleNamespace, amendment: str, reason: str
) -> None:
    commit = vars(history).get(amendment, amendment)

    with pytest.raises(PreregistrationError, match=reason):
        check_scoring(history.root, AUDIT, commit, None)


def test_the_gate_refuses_an_amendment_made_after_the_checkout(history: SimpleNamespace) -> None:
    git(history.root, "switch", "--quiet", "-c", "later")
    later = commit_files(history.root, {"docs/usages_audit.md": "amended after the scorer\n"})
    git(history.root, "switch", "--quiet", "-")

    with pytest.raises(PreregistrationError, match="doesn't come before HEAD"):
        check_scoring(history.root, AUDIT, later, None)


def test_the_gate_refuses_a_scorer_or_key_changed_since_its_commit(
    history: SimpleNamespace,
) -> None:
    (history.root / SCORER).write_text("score, edited after the commit\n")

    with pytest.raises(PreregistrationError, match=r"score_usages\.py is not committed unchanged"):
        check_scoring(history.root, AUDIT, history.amendment, None)


def test_the_gate_refuses_an_audit_without_its_calibration_or_already_scored(
    history: SimpleNamespace,
) -> None:
    with pytest.raises(PreregistrationError, match=r"of_record/author\.json"):
        check_scoring(history.root, AUDIT, history.amendment, Path(f"{AUDIT}_of_record"))
    (history.root / AUDIT / "results.json").write_text("{}\n")
    with pytest.raises(PreregistrationError, match="already scored"):
        check_scoring(history.root, AUDIT, history.amendment, None)
