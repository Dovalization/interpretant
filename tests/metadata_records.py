"""OpenAlex metadata records for tests, shaped like the records the fetch stores."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

FETCHED = "2026-11-13"


def metadata_record(
    work: int, *, source_type: str | None = "journal", indexed_in: Sequence[str] = ("crossref",)
) -> dict[str, Any]:
    """A found work's record: its primary source's type and the indexes that list it."""
    location = {"source": {"display_name": "A source", "type": source_type}}
    return {
        "id": f"https://openalex.org/W{work}",
        "fetched": FETCHED,
        "doi": None,
        "primary_location": location if source_type else None,
        "indexed_in": list(indexed_in),
        "is_paratext": False,
        "is_retracted": False,
    }


def missing_record(work: int) -> dict[str, Any]:
    """The record of a work OpenAlex no longer returns."""
    return {"id": f"https://openalex.org/W{work}", "fetched": FETCHED, "missing": True}
