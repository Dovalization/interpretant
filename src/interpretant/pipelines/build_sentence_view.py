"""The contextual sentence view (decision agenda 2.2; `docs/sentence_view.md`), part by part.

Each worker process takes whole parts of the abstracts table and writes one row per sentence of
every kept work, with what the sentence may be used for. The view depends only on the abstracts
table, which holds the sentence boundaries, their languages and the removed ranges.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from interpretant.catalog.abstracts import abstracts_parts, load_abstracts_part
from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.sentence_view import CUT_OFF, REMOVED, save_sentence_view
from interpretant.corpus.sentence_view import build_sentence_view as sentence_rows
from interpretant.pipelines.progress import SILENT, Progress
from interpretant.pipelines.workers import map_parts

_COLUMNS = ("work_id", "text", "sentences", "removals", "flags", "dropped_reason")


@dataclass(frozen=True)
class SentenceViewResult:
    """What the stage wrote: kept works, sentences, joins, those removed or cut off, and parts."""

    work_count: int
    sentence_count: int
    joined_count: int
    removed_count: int
    cut_off_count: int
    parts: list[str]


def build_sentence_view(
    catalog: CatalogConfig, *, process_count: int = 1, progress: Progress = SILENT
) -> SentenceViewResult:
    """Write the sentence view of every kept work, one part per part of the abstracts table.

    Raises:
        FileNotFoundError: if the abstracts table has no part.
        pandera.errors.SchemaError: if a part's rows break the view's schema.
    """
    parts = abstracts_parts(catalog)
    if not parts:
        raise FileNotFoundError(f"no part of the abstracts table under {catalog.abstracts.path}")
    progress.stage("building the sentence view", total=len(parts))
    results = map_parts(
        _build_part, dict.fromkeys(parts, catalog), process_count=process_count, progress=progress
    )
    counts: Counter[str] = Counter()
    for result in results.values():
        counts.update(result)
    return SentenceViewResult(
        work_count=counts["works"],
        sentence_count=counts["sentences"],
        joined_count=counts["joined"],
        removed_count=counts[REMOVED],
        cut_off_count=counts[CUT_OFF],
        parts=parts,
    )


def _build_part(part: str, catalog: CatalogConfig) -> Counter[str]:
    """Write one part's sentence view; its counts of works, sentences, joins, removed, cut off."""
    abstracts = load_abstracts_part(catalog, part, _COLUMNS)
    view = sentence_rows(abstracts)
    save_sentence_view(catalog, view, part=part)
    return Counter(
        {
            "works": view["work_id"].n_unique(),
            "sentences": view.height,
            "joined": int(view["merge_rules"].list.len().sum()),
            REMOVED: int((view["exclusion"] == REMOVED).sum()),
            CUT_OFF: int((view["exclusion"] == CUT_OFF).sum()),
        }
    )
