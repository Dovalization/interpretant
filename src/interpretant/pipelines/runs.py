"""The run log: one record per run of a corpus command (decision agenda, step 9).

DVC's lock file records which data, settings and code produced each output, but not when a run
took place, how long it took or how much memory it needed; the run log keeps that. The machine is
described by its system, architecture and CPU count rather than its name, since the log is
committed to a public repository.
"""

from __future__ import annotations

import os
import platform
import resource
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.reports import append_run_record

REPOSITORY = Path(__file__).resolve().parents[3]
# Files a run itself changes; they do not make the code it ran from "uncommitted".
_RUN_OUTPUTS = (":!dvc.lock", ":!metrics")


def record_run(
    catalog: CatalogConfig,
    *,
    command: str,
    stages: Sequence[tuple[str, float]],
    counts: Mapping[str, int],
    root: Path = REPOSITORY,
    checkout: tuple[str | None, bool] | None = None,
) -> Path:
    """Append the record of one run to the run log and return the log's file.

    ``checkout`` is the commit and uncommitted state the run started from (``checkout_state``);
    without it they are read at the end, which is wrong if the checkout changed during the run.
    """
    commit, has_uncommitted_changes = checkout if checkout is not None else checkout_state(root)
    record = {
        "command": command,
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPUs",
        "commit": commit,
        "uncommitted_changes": has_uncommitted_changes,
        "seconds": round(sum(seconds for _, seconds in stages), 1),
        "stages": [{"name": name, "seconds": round(seconds, 1)} for name, seconds in stages],
        "counts": dict(counts),
        "peak_memory_mb": peak_memory_mb(),
    }
    return append_run_record(catalog, record)


def peak_memory_mb() -> dict[str, float]:
    """The run's peak resident memory: the main process's, and the largest worker's.

    From ``getrusage``: ``ru_maxrss`` is in kilobytes on Linux and in bytes on macOS, and for
    the children it is the largest single finished worker, not their sum.
    """
    unit = 1 if sys.platform == "darwin" else 1024
    return {
        "main": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit / 2**20, 1),
        "largest_worker": round(
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * unit / 2**20, 1
        ),
    }


def checkout_state(root: Path) -> tuple[str | None, bool]:
    """The commit checked out at ``root`` and whether tracked files differ from it.

    Without git (or outside a repository) the commit is unknown and no changes are reported.
    """
    try:
        commit = _git(root, "rev-parse", "HEAD")
        changes = _git(
            root, "status", "--porcelain", "--untracked-files=no", "--", ".", *_RUN_OUTPUTS
        )
    except (OSError, subprocess.CalledProcessError):
        return None, False
    return commit, bool(changes)


def _git(root: Path, *arguments: str) -> str:
    # Inside a git hook, git exports GIT_DIR and its kin, which would override ``root``.
    environment = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    completed = subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, text=True, check=True, env=environment
    )
    return completed.stdout.strip()
