"""The pre-registration gate before an audit is scored (CLAUDE.md, "Pre-registration").

Each audit names its own scorer, its plan and the files fixed with the plan (``PreregisteredAudit``;
the plans are `docs/usages_audit.md` and `docs/junk_gate.md`). An audit is scored only from a
checkout that keeps its pre-registration: the amendment (the plan's last change before scoring) is
committed and comes before the checkout's commit, the audit's scorer is committed unchanged, the
files fixed with its plan (a codebook) are committed unchanged and are the amendment's, the draw's
key and counts are committed unchanged, the author's calibration verdicts are committed when the
audit is scored with them, and the audit has not been scored yet.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreregisteredAudit:
    """The files that hold one audit's pre-registration, as paths from the repository's root.

    Attributes:
        scorer: The script that scores the audit, committed before it is run.
        plan_files: Where an amendment of the audit's plan is written.
        fixed_files: What the judges were given with the plan, such as their codebook: it may
            not change after the plan's last amendment.
    """

    scorer: str
    plan_files: tuple[str, ...]
    fixed_files: tuple[str, ...] = ()


USAGES_AUDIT = PreregisteredAudit(
    scorer="scripts/review/score_usages.py",
    plan_files=(
        "docs/usages_audit.md",
        "docs/predictions.md",
        "docs/predictions_cross_perplexity.md",
    ),
)
JUNK_GATE = PreregisteredAudit(
    scorer="scripts/review/score_junk_gate.py",
    plan_files=("docs/junk_gate.md",),
    fixed_files=("curated/judging/junk_gate/prompt.md",),
)


class PreregistrationError(RuntimeError):
    """The checkout doesn't keep the pre-registration, so the audit isn't scored from it."""


def check_scoring(
    audit: PreregisteredAudit, root: Path, folder: Path, amendment: str, calibration: Path | None
) -> None:
    """Refuse to score ``audit`` from the checkout at ``root`` unless it keeps the pre-registration.

    ``folder`` holds the audit's draw and ``calibration`` the author's verdicts, both relative to
    ``root``; ``amendment`` names a commit.

    Raises:
        PreregistrationError: naming the first rule the checkout breaks.
    """
    if not _is_commit(root, amendment):
        raise PreregistrationError(f"the amendment {amendment} is not a commit here")
    touched = _git(root, "diff-tree", "--no-commit-id", "--name-only", "-r", "--root", amendment)
    if not set(touched.splitlines()) & set(audit.plan_files):
        raise PreregistrationError(
            f"the amendment {amendment[:8]} doesn't touch the audit's plan "
            f"({', '.join(audit.plan_files)})"
        )
    if _run(root, "merge-base", "--is-ancestor", amendment, "HEAD").returncode != 0:
        raise PreregistrationError(f"the amendment {amendment[:8]} doesn't come before HEAD")
    committed = [audit.scorer, *audit.fixed_files, f"{folder}/key.json", f"{folder}/counts.json"]
    if calibration is not None:
        committed.append(f"{calibration}/author.json")
    for path in committed:
        if not _is_committed_unchanged(root, path):
            raise PreregistrationError(f"{path} is not committed unchanged at HEAD")
    for path in audit.fixed_files:
        if not _is_unchanged_since(root, amendment, path):
            raise PreregistrationError(f"{path} changed after the amendment {amendment[:8]}")
    if (root / folder / "results.json").exists():
        raise PreregistrationError(f"{folder}/results.json exists: the audit is already scored")


def _is_committed_unchanged(root: Path, path: str) -> bool:
    is_tracked = _run(root, "cat-file", "-e", f"HEAD:{path}").returncode == 0
    return is_tracked and not _git(root, "status", "--porcelain", "--", path)


def _is_unchanged_since(root: Path, revision: str, path: str) -> bool:
    return _run(root, "diff", "--quiet", revision, "HEAD", "--", path).returncode == 0


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
