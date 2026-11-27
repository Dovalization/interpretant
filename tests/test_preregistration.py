"""The pre-registration gate before an audit is scored, on real git in a temporary folder."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from interpretant.pipelines.preregistration import (
    JUNK_GATE,
    USAGES_AUDIT,
    PreregisteredAudit,
    PreregistrationError,
    check_scoring,
)

REPOSITORY = Path(__file__).resolve().parents[1]
AUDIT = Path("data/review/audit")
[JUNK_GATE_PLAN] = JUNK_GATE.plan_files
[CODEBOOK] = JUNK_GATE.fixed_files


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


def write_history(root: Path, audit: PreregisteredAudit) -> SimpleNamespace:
    """The audit's files, the draw's record, an amendment, the calibration, then a code commit."""
    root.mkdir()
    git(root, "init", "--quiet")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test")
    plan = audit.plan_files[0]
    fixed = dict.fromkeys(audit.fixed_files, "as the judges were given it\n")
    base = commit_files(root, {plan: "the plan\n", audit.scorer: "score\n"} | fixed)
    drawn = commit_files(root, {f"{AUDIT}/key.json": "{}\n", f"{AUDIT}/counts.json": "{}\n"})
    amendment = commit_files(root, {plan: "the plan, amended\n"})
    commit_files(root, {f"{AUDIT}/author.json": "{}\n"})
    commit_files(root, {audit.scorer: "score, as committed before the run\n"})
    return SimpleNamespace(root=root, base=base, drawn=drawn, amendment=amendment)


@pytest.fixture
def history(tmp_path: Path) -> SimpleNamespace:
    """A checkout that keeps the usages audit's pre-registration."""
    return write_history(tmp_path / "checkout", USAGES_AUDIT)


@pytest.fixture
def junk_gate(tmp_path: Path) -> SimpleNamespace:
    """A checkout that keeps the junk gate's pre-registration, its codebook included."""
    return write_history(tmp_path / "checkout", JUNK_GATE)


def test_an_audit_is_scored_from_a_checkout_that_keeps_the_pre_registration(
    history: SimpleNamespace,
) -> None:
    check_scoring(USAGES_AUDIT, history.root, AUDIT, history.amendment, None)
    check_scoring(USAGES_AUDIT, history.root, AUDIT, history.amendment, AUDIT)


@pytest.mark.parametrize(
    ("amendment", "reason"),
    [
        ("0" * 40, "is not a commit"),
        ("drawn", r"doesn't touch the audit's plan \(docs/usages_audit\.md, docs/predictions"),
    ],
)
def test_the_gate_refuses_an_amendment_that_isnt_one(
    history: SimpleNamespace, amendment: str, reason: str
) -> None:
    commit = vars(history).get(amendment, amendment)

    with pytest.raises(PreregistrationError, match=reason):
        check_scoring(USAGES_AUDIT, history.root, AUDIT, commit, None)


def test_the_gate_refuses_an_amendment_made_after_the_checkout(history: SimpleNamespace) -> None:
    git(history.root, "switch", "--quiet", "-c", "later")
    later = commit_files(history.root, {"docs/usages_audit.md": "amended after the scorer\n"})
    git(history.root, "switch", "--quiet", "-")

    with pytest.raises(PreregistrationError, match="doesn't come before HEAD"):
        check_scoring(USAGES_AUDIT, history.root, AUDIT, later, None)


def test_the_gate_refuses_a_scorer_or_key_changed_since_its_commit(
    history: SimpleNamespace,
) -> None:
    (history.root / USAGES_AUDIT.scorer).write_text("score, edited after the commit\n")

    with pytest.raises(PreregistrationError, match=r"score_usages\.py is not committed unchanged"):
        check_scoring(USAGES_AUDIT, history.root, AUDIT, history.amendment, None)


def test_the_gate_refuses_an_audit_without_its_calibration_or_already_scored(
    history: SimpleNamespace,
) -> None:
    with pytest.raises(PreregistrationError, match=r"of_record/author\.json"):
        check_scoring(
            USAGES_AUDIT, history.root, AUDIT, history.amendment, Path(f"{AUDIT}_of_record")
        )
    (history.root / AUDIT / "results.json").write_text("{}\n")
    with pytest.raises(PreregistrationError, match="already scored"):
        check_scoring(USAGES_AUDIT, history.root, AUDIT, history.amendment, None)


def test_an_audit_that_fixes_no_file_is_scored_whatever_a_codebook_became(
    history: SimpleNamespace,
) -> None:
    commit_files(history.root, {CODEBOOK: "written after the usages audit's amendment\n"})

    check_scoring(USAGES_AUDIT, history.root, AUDIT, history.amendment, None)


def test_the_junk_gate_is_scored_with_its_own_scorer_and_the_amendments_codebook(
    junk_gate: SimpleNamespace,
) -> None:
    check_scoring(JUNK_GATE, junk_gate.root, AUDIT, junk_gate.amendment, None)


def test_the_gate_refuses_a_codebook_changed_after_the_amendment(
    junk_gate: SimpleNamespace,
) -> None:
    commit_files(junk_gate.root, {CODEBOOK: "revised after the amendment\n"})

    with pytest.raises(PreregistrationError, match=r"prompt\.md changed after the amendment"):
        check_scoring(JUNK_GATE, junk_gate.root, AUDIT, junk_gate.amendment, None)


def test_the_gate_takes_a_codebook_revised_with_a_new_amendment(
    junk_gate: SimpleNamespace,
) -> None:
    revision = commit_files(
        junk_gate.root,
        {CODEBOOK: "revised once\n", JUNK_GATE_PLAN: "the plan, amended for the revision\n"},
    )

    check_scoring(JUNK_GATE, junk_gate.root, AUDIT, revision, None)


def test_the_gate_refuses_a_codebook_edited_since_its_commit(junk_gate: SimpleNamespace) -> None:
    (junk_gate.root / CODEBOOK).write_text("edited after the commit\n")

    with pytest.raises(PreregistrationError, match=r"prompt\.md is not committed unchanged"):
        check_scoring(JUNK_GATE, junk_gate.root, AUDIT, junk_gate.amendment, None)


def test_the_gate_refuses_an_amendment_of_another_audits_plan(junk_gate: SimpleNamespace) -> None:
    other = commit_files(junk_gate.root, {"docs/usages_audit.md": "the usages plan, amended\n"})

    with pytest.raises(
        PreregistrationError, match=r"doesn't touch the audit's plan \(docs/junk_gate\.md\)"
    ):
        check_scoring(JUNK_GATE, junk_gate.root, AUDIT, other, None)


@pytest.mark.parametrize("audit", [USAGES_AUDIT, JUNK_GATE], ids=["usages audit", "junk gate"])
def test_each_audits_plan_and_fixed_files_are_in_the_repository(
    audit: PreregisteredAudit,
) -> None:
    named = (*audit.plan_files, *audit.fixed_files)

    assert [path for path in named if not (REPOSITORY / path).is_file()] == []
