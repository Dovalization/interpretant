"""Fetch each raw work's record in Crossref and PubMed (decision agenda 4.8, linked sources).

Not a stage of the rebuild: like the work metadata it runs by hand once and its output is
registered with ``dvc add``. Each raw cell's works with a DOI (Crossref) or a PubMed id (PubMed)
are looked up in batches of 200 and appended to the cell's file; works already there are skipped,
so an interrupted run resumes where it stopped. With a contact address Crossref serves the
requests from its polite pool (3 a second, 3 at once); without one, from its public pool (1 a
second, 1 at once). Biopython keeps PubMed's requests to NCBI's limit (3 a second without a key).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from functools import partial
from pathlib import Path
from typing import Any

from habanero import Crossref
from habanero.exceptions import RequestError
from pydantic_settings import BaseSettings, SettingsConfigDict

from interpretant.catalog import (
    CatalogConfig,
    LinkedSource,
    RawCell,
    append_linked_records,
    fetched_linked_work_ids,
    load_work_metadata,
)
from interpretant.corpus.linked_sources import (
    DOIS_PER_REQUEST,
    PMIDS_PER_REQUEST,
    LinkedWork,
    configure_entrez,
    fetch_crossref_records,
    fetch_pubmed_records,
    select_linked_works,
)
from interpretant.pipelines.progress import SILENT, Progress

ENV_FILE = Path(__file__).resolve().parents[3] / ".env"
TOOL_NAME = "interpretant"
# Statuses worth asking again after a pause: rate limiting and the server's own failures.
TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})

Fetcher = Callable[[Sequence[LinkedWork]], list[dict[str, Any]]]


class LinkedSourcesSettings(BaseSettings):
    """A contact address for Crossref and NCBI, from the environment or ``.env``, if given."""

    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    contact_email: str | None = None


@dataclass(frozen=True)
class SourcePlan:
    """How one source is asked: the fetch, its batch size and how many requests run at once."""

    fetch: Fetcher
    batch_size: int
    workers: int


@dataclass(frozen=True)
class CellLookup:
    """What one cell's lookup found: works already there, found now, and missing now."""

    cell: RawCell
    already_fetched: int
    found: int
    missing: int


class Throttle:
    """Lets calls start no closer together than ``interval`` seconds, across threads."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self._lock = threading.Lock()
        self._next_start = 0.0

    def wait(self) -> None:
        """Block until this call may start."""
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_start)
            self._next_start = start + self.interval
        time.sleep(start - now)


def build_source_plan(
    source: LinkedSource, settings: LinkedSourcesSettings, *, today: date
) -> SourcePlan:
    """The fetch for a run of one source, configured with the contact address if there is one."""
    if source == "pubmed":
        configure_entrez(tool=TOOL_NAME, email=settings.contact_email)
        return SourcePlan(
            fetch=partial(fetch_pubmed_records, fetched=today),
            batch_size=PMIDS_PER_REQUEST,
            workers=1,
        )
    polite = settings.contact_email is not None
    crossref = Crossref(mailto=settings.contact_email, ua_string=TOOL_NAME, timeout=120)
    throttle = Throttle(1 / 3 if polite else 1.0)

    def fetch(works: Sequence[LinkedWork]) -> list[dict[str, Any]]:
        return retry_transient(lambda: _throttled(throttle, crossref, works, today))

    return SourcePlan(fetch=fetch, batch_size=DOIS_PER_REQUEST, workers=3 if polite else 1)


def retry_transient(
    call: Callable[[], list[dict[str, Any]]], *, attempts: int = 6, first_pause: float = 5.0
) -> list[dict[str, Any]]:
    """Call again after a doubling pause while it fails with a transient error, then raise."""
    for attempt in range(attempts - 1):
        try:
            return call()
        except Exception as exc:
            if not is_transient(exc):
                raise
            time.sleep(first_pause * 2**attempt)
    return call()


def is_transient(error: Exception) -> bool:
    """Whether a failed Crossref request is worth repeating: rate limited, failed, unanswered.

    habanero raises ``RequestError`` for an error status with a JSON body, lets other error
    statuses through as the HTTP client's own error (which carries the response), and wraps a
    request that got no answer in a ``RuntimeError`` whose message starts "HTTP Exception".
    """
    if isinstance(error, RequestError):
        return int(error.status_code) in TRANSIENT_STATUSES
    response = getattr(error, "response", None)
    if response is not None and hasattr(response, "status_code"):
        return int(response.status_code) in TRANSIENT_STATUSES
    return isinstance(error, OSError) or (
        isinstance(error, RuntimeError) and str(error).startswith("HTTP Exception")
    )


def count_linked_works(
    catalog: CatalogConfig, cells: Sequence[RawCell], source: LinkedSource
) -> int:
    """The works the source can be asked about in each cell, summed over the cells."""
    return sum(
        len(select_linked_works(load_work_metadata(catalog, cell), source)) for cell in cells
    )


def fetch_cell_linked_records(
    catalog: CatalogConfig,
    cell: RawCell,
    source: LinkedSource,
    plan: SourcePlan,
    *,
    progress: Progress = SILENT,
) -> CellLookup:
    """Look up the cell's works not yet in its file for the source, a batch at a time.

    ``plan.workers`` requests run at once; their results are appended in the cell's order by this
    thread alone, so the file and the resume behave as with one request at a time. If a request
    fails, the batches not yet started are cancelled and the error is raised; finished batches
    after the failed one are fetched again on the next run.
    """
    done = fetched_linked_work_ids(catalog, cell, source)
    works = select_linked_works(load_work_metadata(catalog, cell), source)
    pending = [work for work in works if work.work_id not in done]
    batches = [
        pending[start : start + plan.batch_size]
        for start in range(0, len(pending), plan.batch_size)
    ]
    progress.advance(len(works) - len(pending))
    missing = 0
    with ThreadPoolExecutor(max_workers=plan.workers) as executor:
        try:
            for records in executor.map(plan.fetch, batches):
                append_linked_records(catalog, cell, source, records)
                missing += sum(1 for record in records if record.get("missing"))
                progress.advance(len(records))
        except BaseException:
            executor.shutdown(cancel_futures=True)
            raise
    return CellLookup(
        cell=cell,
        already_fetched=len(works) - len(pending),
        found=len(pending) - missing,
        missing=missing,
    )


def _throttled(
    throttle: Throttle, crossref: Crossref, works: Sequence[LinkedWork], today: date
) -> list[dict[str, Any]]:
    throttle.wait()
    return fetch_crossref_records(crossref, works, fetched=today)
