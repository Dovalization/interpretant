"""Stage 1b of the run order (decision agenda 9.1): the works merged and their texts repaired.

Every selected cell's works are merged into one record per work, once every work's metadata is
checked, and grouped into parts: the works whose first field is one field, in one decade. Each
part's texts are repaired in a worker process (``corpus.repair``: spacing around punctuation,
HTML entities, web addresses) and written as the part's base texts, which the word repairs read.
"""

from __future__ import annotations

from collections import defaultdict

import polars as pl

from interpretant.catalog import REPAIRED_TEXTS_SCHEMA, CatalogConfig, save_base_texts
from interpretant.corpus.repair import repair_text
from interpretant.corpus.works import part_of
from interpretant.pipelines.corpus import (
    CorpusResult,
    corpus_cells,
    corpus_parts,
    load_corpus_works,
    part_name,
)
from interpretant.pipelines.params import CorpusSelection
from interpretant.pipelines.progress import SILENT, Progress
from interpretant.pipelines.workers import map_parts


def repair_texts(
    catalog: CatalogConfig,
    selection: CorpusSelection,
    *,
    process_count: int = 1,
    progress: Progress = SILENT,
) -> CorpusResult:
    """Merge the selected cells' works, repair their texts and write every part's base texts.

    A part with no works (every work of its cell lists an earlier field first) is written empty,
    so the next stages find every part. This stage drops, removes and flags nothing.

    Raises:
        ValueError: if no raw cell matches the selection, or a cell's metadata has not been
            fetched or is incomplete.
    """
    cells = corpus_cells(catalog, selection)
    abstracts = load_corpus_works(catalog, cells, progress)
    by_part: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    for abstract in abstracts:
        by_part[part_name(*part_of(abstract))].append((abstract.work_id, abstract.text))
    progress.stage("repairing the texts", total=len(abstracts))
    repaired = map_parts(
        _repair_part,
        dict(by_part),
        process_count=process_count,
        progress=progress,
        sizes={part: len(works) for part, works in by_part.items()},
    )
    parts = corpus_parts(cells)
    progress.stage("writing the base texts", total=len(parts))
    for part in parts:
        rows = repaired.get(part, [])
        save_base_texts(
            catalog, pl.DataFrame(rows, schema=REPAIRED_TEXTS_SCHEMA, orient="row"), part
        )
        progress.advance()
    return CorpusResult(
        abstract_count=len(abstracts),
        parts=parts,
        dropped_counts={},
        removal_counts={},
        flag_counts={},
    )


def _repair_part(_part: str, works: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """One part's works with their texts repaired."""
    return [(work_id, repair_text(text)) for work_id, text in works]
