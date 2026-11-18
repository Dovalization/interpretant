"""The thin static view (decision agenda 9.1, stage 6): every cached document's static tokens.

Each worker process takes whole parts: it reads a part's cached spaCy documents, applies the lemma
exceptions to them through spaCy's attribute ruler (agenda 3.6: the documents are not parsed
again), and writes one row per document with its static tokens, their offsets and their
finite-verb flags. The view depends only on the spaCy pass and the exceptions file, so cleaning
changes never rerun this stage; a build's removals apply to the offsets when a cell's text is made.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import spacy

from interpretant.catalog.annotations import WORK_ID, load_annotations
from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.lemma_exceptions import LemmaException, load_lemma_exceptions
from interpretant.catalog.spellings import Spelling, load_spellings
from interpretant.catalog.static_view import STATIC_VIEW_SCHEMA, save_static_view
from interpretant.corpus.lemma_exceptions import lemma_exception_ruler
from interpretant.corpus.static_view import static_tokens
from interpretant.pipelines.corpus import corpus_cells, corpus_parts
from interpretant.pipelines.params import CorpusSelection
from interpretant.pipelines.progress import SILENT, Progress
from interpretant.pipelines.workers import map_parts


@dataclass(frozen=True)
class StaticViewResult:
    """What the stage wrote: documents, static tokens, parts, exceptions and spellings."""

    document_count: int
    token_count: int
    parts: list[str]
    exception_count: int
    spelling_count: int


@dataclass(frozen=True)
class _PartJob:
    """What a worker needs for its parts: the catalog, the lemma exceptions, the spelling map."""

    catalog: CatalogConfig
    exceptions: tuple[LemmaException, ...]
    spellings: tuple[Spelling, ...]


def build_static_view(
    catalog: CatalogConfig,
    selection: CorpusSelection,
    *,
    process_count: int = 1,
    progress: Progress = SILENT,
) -> StaticViewResult:
    """Write the static view of every cached document of the corpus, per part.

    The lemma exceptions and the spelling map are read once here, so a malformed file fails before
    any worker starts.

    Raises:
        ValueError: if no raw cell matches the selection, the exceptions file's header or a
            repeated form and tag is wrong, or the spelling map's header, a repeated form or a
            chained standard is.
        FileNotFoundError: if a part has not been annotated, or either file is missing.
        pydantic.ValidationError: if a row of either file is malformed.
    """
    parts = corpus_parts(corpus_cells(catalog, selection))
    job = _PartJob(catalog, tuple(load_lemma_exceptions(catalog)), tuple(load_spellings(catalog)))
    progress.stage("building the static view", total=len(parts))
    results = map_parts(
        _build_part,
        dict.fromkeys(parts, job),
        process_count=process_count,
        progress=progress,
    )
    return StaticViewResult(
        document_count=sum(documents for documents, _ in results.values()),
        token_count=sum(tokens for _, tokens in results.values()),
        parts=parts,
        exception_count=len(job.exceptions),
        spelling_count=len(job.spellings),
    )


def _build_part(part: str, job: _PartJob) -> tuple[int, int]:
    """Write one part's static view; its documents and static tokens.

    The ruler is built on the vocab the part's documents load into, since spaCy interns its
    lemmas there.
    """
    vocab = spacy.blank("en").vocab
    ruler = lemma_exception_ruler(vocab, job.exceptions)
    spellings = {spelling.form: spelling.standard for spelling in job.spellings}
    rows = []
    for doc in load_annotations(job.catalog, part, vocab):
        view = static_tokens(ruler(doc), spellings)
        rows.append((doc.user_data[WORK_ID], view.tokens, view.starts, view.finite_verbs))
    table = pl.DataFrame(rows, schema=STATIC_VIEW_SCHEMA, orient="row")
    save_static_view(job.catalog, table, part=part)
    return table.height, int(table["tokens"].list.len().sum())
