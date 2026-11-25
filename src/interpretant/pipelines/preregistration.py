"""The pre-registration gate before an audit is scored (`docs/usages_audit.md`, CLAUDE.md).

An audit is scored only from a checkout that keeps the pre-registration: the amendment (the plan's
or the predictions' last change before scoring) is committed and comes before the checkout's
commit, the scorer is committed unchanged, the draw's key and counts are committed unchanged, the
author's calibration verdicts are committed when the audit is scored with them, and the audit has
not been scored yet.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

SCORER = "scripts/review/score_usages.py"
# Where an amendment of an audit's plan is written
PLAN_FILES = ("docs/usages_audit.md", "docs/predictions.md", "docs/predictions_cross_perplexity.md")


class PreregistrationError(RuntimeError):
    """The checkout doesn't keep the pre-registration, so the audit isn't scored from it."""


def check_scoring(root: Path, audit: Path, amendment: str, calibration: Path | None) -> None:
    """Refuse to score ``audit`` from the checkout at ``root`` unless it keeps the pre-registration.

    ``audit`` and ``calibration`` are folders relative to ``root``; ``amendment`` names a commit.

    Raises:
        PreregistrationError: naming the first rule the checkout breaks.
    """
    if not _is_commit(root, amendment):
        raise PreregistrationError(f"the amendment {amendment} is not a commit here")
    touched = _git(root, "diff-tree", "--no-commit-id", "--name-only", "-r", "--root", amendment)
    if not set(touched.splitlines()) & set(PLAN_FILES):
        raise PreregistrationError(
            f"the amendment {amendment[:8]} touches neither the audit's plan nor the predictions"
        )
    if _run(root, "merge-base", "--is-ancestor", amendment, "HEAD").returncode != 0:
        raise PreregistrationError(f"the amendment {amendment[:8]} doesn't come before HEAD")
    committed = [SCORER, f"{audit}/key.json", f"{audit}/counts.json"]
    if calibration is not None:
        committed.append(f"{calibration}/author.json")
    for path in committed:
        if not _is_committed_unchanged(root, path):
            raise PreregistrationError(f"{path} is not committed unchanged at HEAD")
    if (root / audit / "results.json").exists():
        raise PreregistrationError(f"{audit}/results.json exists: the audit is already scored")


def _is_committed_unchanged(root: Path, path: str) -> bool:
    is_tracked = _run(root, "cat-file", "-e", f"HEAD:{path}").returncode == 0
    return is_tracked and not _git(root, "status", "--porcelain", "--", path)


def _is_commit(root: Path, revision: str) -> bool:
    return _run(root, "cat-file", "-e", f"{revision}^{{commit}}").returncode == 0


def _git(root: Path, *arguments: str) -> str:
    completed = _run(root, *arguments)
    if completed.returncode != 0:
        raise PreregistrationError(f"git {arguments[0]} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _run(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, text=True, check=False
    )
