"""The thin static view (decision agenda 9.1, stage 6): every cached document's static tokens.

Each worker process takes whole parts: it reads a part's cached spaCy documents and writes one row
per document with its static tokens, their offsets and their finite-verb flags. The view depends
only on the spaCy pass, so cleaning changes never rerun this stage; a build's removals apply to
the offsets when a cell's text is made.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import spacy

from interpretant.catalog.annotations import WORK_ID, load_annotations
from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.static_view import STATIC_VIEW_SCHEMA, save_static_view
from interpretant.corpus.static_view import static_tokens
from interpretant.pipelines.corpus import corpus_cells, corpus_parts
from interpretant.pipelines.params import CorpusSelection
from interpretant.pipelines.progress import SILENT, Progress
from interpretant.pipelines.workers import map_parts


@dataclass(frozen=True)
class StaticViewResult:
    """What the stage wrote: the documents and static tokens, and the parts."""

    document_count: int
    token_count: int
    parts: list[str]


def build_static_view(
    catalog: CatalogConfig,
    selection: CorpusSelection,
    *,
    process_count: int = 1,
    progress: Progress = SILENT,
) -> StaticViewResult:
    """Write the static view of every cached document of the corpus, per part.

    Raises:
        ValueError: if no raw cell matches the selection.
        FileNotFoundError: if a part has not been annotated.
    """
    parts = corpus_parts(corpus_cells(catalog, selection))
    progress.stage("building the static view", total=len(parts))
    results = map_parts(
        _build_part,
        dict.fromkeys(parts, catalog),
        process_count=process_count,
        progress=progress,
    )
    return StaticViewResult(
        document_count=sum(documents for documents, _ in results.values()),
        token_count=sum(tokens for _, tokens in results.values()),
        parts=parts,
    )


def _build_part(part: str, catalog: CatalogConfig) -> tuple[int, int]:
    """Write one part's static view; its documents and static tokens."""
    vocab = spacy.blank("en").vocab
    rows = []
    for doc in load_annotations(catalog, part, vocab):
        view = static_tokens(doc)
        rows.append((doc.user_data[WORK_ID], view.tokens, view.starts, view.finite_verbs))
    table = pl.DataFrame(rows, schema=STATIC_VIEW_SCHEMA, orient="row")
    save_static_view(catalog, table, part=part)
    return table.height, int(table["tokens"].list.len().sum())
