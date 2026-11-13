"""Fetch OpenAlex's record of every raw work beyond its abstract (decision agenda 4.8).

Not a stage of the rebuild: like the downloads it runs by hand once and its output is registered.
Each raw cell's works are looked up 100 at a time, several requests at once, and appended to the
cell's metadata file; works already there are skipped, so an interrupted run resumes where it
stopped. A free OpenAlex key allows 30 requests a second; eight at once run at about five.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from functools import partial
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

from interpretant.catalog import (
    CatalogConfig,
    RawCell,
    append_work_metadata,
    fetched_work_ids,
    list_raw_cells,
    load_raw_cell,
)
from interpretant.corpus.work_metadata import (
    IDS_PER_REQUEST,
    configure_openalex,
    fetch_work_metadata,
)

ENV_FILE = Path(__file__).resolve().parents[3] / ".env"

Fetcher = Callable[[Sequence[str]], list[dict[str, Any]]]


class OpenAlexSettings(BaseSettings):
    """The OpenAlex API key, from the environment or the repository's ``.env`` file."""

    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    openalex_api_key: str


@dataclass(frozen=True)
class CellFetch:
    """What one cell's fetch found: records already there, fetched now, and missing now."""

    cell: RawCell
    already_fetched: int
    fetched: int
    missing: int


def openalex_fetcher(settings: OpenAlexSettings, *, today: date) -> Fetcher:
    """The fetch function for a run: pyalex configured with the key, records dated today."""
    configure_openalex(settings.openalex_api_key)
    return partial(fetch_work_metadata, fetched=today)


def select_raw_cells(
    catalog: CatalogConfig, *, fields: Sequence[str], decades: Sequence[int]
) -> list[RawCell]:
    """The raw cells in the given fields and decades; every raw cell when none are given.

    Raises:
        ValueError: if no raw cell matches.
    """
    cells = [
        cell
        for cell in list_raw_cells(catalog)
        if (not fields or cell.field in fields) and (not decades or cell.decade in decades)
    ]
    if not cells:
        raise ValueError(f"no raw cell for fields {fields} in decades {decades}")
    return cells


def fetch_cell_metadata(
    catalog: CatalogConfig, cell: RawCell, fetch: Fetcher, *, workers: int = 1
) -> CellFetch:
    """Fetch the records of a cell's works not yet in its metadata file, 100 at a time.

    ``workers`` requests run at once; their results are appended in the cell's order by this
    thread alone, so the file and the resume behave as with one request at a time. If a request
    fails, the batches not yet started are cancelled and the error is raised; finished batches
    after the failed one are fetched again on the next run.
    """
    done = fetched_work_ids(catalog, cell)
    pending = list(
        dict.fromkeys(
            work.work_id for work in load_raw_cell(catalog, cell) if work.work_id not in done
        )
    )
    batches = [
        pending[start : start + IDS_PER_REQUEST]
        for start in range(0, len(pending), IDS_PER_REQUEST)
    ]
    missing = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        try:
            for records in executor.map(fetch, batches):
                append_work_metadata(catalog, cell, records)
                missing += sum(1 for record in records if record.get("missing"))
        except BaseException:
            executor.shutdown(cancel_futures=True)
            raise
    return CellFetch(
        cell=cell, already_fetched=len(done), fetched=len(pending) - missing, missing=missing
    )
