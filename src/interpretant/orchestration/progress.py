"""A pipeline's progress as a Prefect progress artifact on the run, one per stage.

It is a ``Progress`` like the command line's (``pipelines.progress``), so the pipelines don't know
about Prefect; a flow passes it alongside the log's ``LiveProgress`` through ``TeeProgress``.
Updates go to the server at most every ``interval`` seconds, and once more when a stage ends: each
is an API call, and worker processes report thousands of items. A failure to publish is logged and
never stops the run.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
from uuid import UUID

from prefect.artifacts import create_progress_artifact, update_progress_artifact

from interpretant.pipelines.progress import format_pace

logger = logging.getLogger(__name__)


class PrefectProgress:
    """Publishes each stage's position as a progress artifact on the current flow run."""

    def __init__(
        self,
        key_prefix: str,
        *,
        interval: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.key_prefix = key_prefix
        self.interval = interval
        self.clock = clock
        self._lock = threading.Lock()
        self._artifact: UUID | None = None
        self._name = ""
        self._total: int | None = None
        self._completed = 0
        self._skipped = 0
        self._started = 0.0
        self._published = 0.0

    def stage(self, name: str, total: int | None = None) -> None:
        """Close the running stage's artifact and start one for the next stage."""
        with self._lock:
            self._publish_locked()
            self._name, self._total = name, total
            self._completed = self._skipped = 0
            self._started = self._published = self.clock()
            self._artifact = None
            try:
                artifact = create_progress_artifact(
                    progress=0.0,
                    key=format_artifact_key(self.key_prefix, name),
                    description=self._describe(self._started),
                )
                assert isinstance(artifact, UUID)  # synchronous here; its type allows a coroutine
                self._artifact = artifact
            except Exception:
                logger.warning("couldn't create the progress artifact for %s", name, exc_info=True)

    def advance(self, amount: int = 1) -> None:
        """Count ``amount`` more items done, publishing if the interval has passed."""
        self._count(amount, skipped=0)

    def skip(self, amount: int) -> None:
        """Count ``amount`` items an earlier run did."""
        self._count(amount, skipped=amount)

    def close(self) -> None:
        """Publish the last stage's final position."""
        with self._lock:
            self._publish_locked()

    def _count(self, amount: int, *, skipped: int) -> None:
        with self._lock:
            self._completed += amount
            self._skipped += skipped
            if self.clock() - self._published >= self.interval:
                self._publish_locked()

    def _publish_locked(self) -> None:
        if self._artifact is None:
            return
        now = self.clock()
        self._published = now
        share = 100 * self._completed / self._total if self._total else 0.0
        try:
            update_progress_artifact(
                artifact_id=self._artifact,
                progress=min(share, 100.0),
                description=self._describe(now),
            )
        except Exception:
            logger.warning("couldn't update the progress of %s", self._name, exc_info=True)

    def _describe(self, now: float) -> str:
        remaining = self._total - self._completed if self._total else None
        pace = format_pace(self._completed - self._skipped, now - self._started, remaining)
        position = (
            f"{self._completed:,}/{self._total:,}" if self._total else f"{self._completed:,} items"
        )
        return ", ".join(part for part in (f"{self._name}: {position}", pace) if part)


def format_artifact_key(prefix: str, stage: str) -> str:
    """An artifact key Prefect accepts: lowercase letters, digits and dashes."""
    slug = re.sub(r"[^a-z0-9]+", "-", f"{prefix}-{stage}".lower()).strip("-")
    return slug or "progress"
