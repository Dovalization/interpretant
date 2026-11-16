"""Stage 2a of the run order (decision agenda 9.1): the spaCy pass over the raw OpenAlex cells.

Every selected cell's works are merged, so a work is one record listing all its fields, and
grouped into parts: a part holds the works whose first field is one field, in one decade. The
works of the parts asked for are repaired and annotated in one stream, and each part's documents
go to the annotation cache as the stream reaches its end. The documents carry tokens, sentences,
tags and lemmas only: sentence languages (stage 2b) and the cleaning patterns and keeping rules
(stage 2c) read the cache, so changing them does not rerun this stage. A run may annotate some
parts only. Every selected cell's metadata must have been fetched (``interpretant
fetch-metadata``), so a missing fetch stops the run before the slow pass, not after it.
"""

from __future__ import annotations

import gc
from collections import Counter
from collections.abc import Collection, Iterable, Iterator
from itertools import groupby
from typing import TypeVar

from pydantic_settings import SettingsConfigDict

from interpretant.catalog import (
    CatalogConfig,
    load_glued_words,
    load_split_words,
    save_annotations,
)
from interpretant.corpus.annotation import annotate_texts, build_pipeline
from interpretant.corpus.glued_words import glued_word_splits
from interpretant.corpus.split_words import split_word_joins
from interpretant.corpus.works import RawAbstract, part_of
from interpretant.pipelines.corpus import (
    CorpusResult,
    corpus_cells,
    load_corpus_works,
    part_name,
)
from interpretant.pipelines.params import CorpusSelection, ParamsSettings
from interpretant.pipelines.progress import SILENT, Progress

_Item = TypeVar("_Item")


class AnnotateCorpusSettings(ParamsSettings):
    """The ``annotate_corpus`` section of ``params.yaml``: what the spaCy models run."""

    model_config = SettingsConfigDict(yaml_config_section="annotate_corpus")

    named_entity_recognition: bool


def annotate_corpus(
    catalog: CatalogConfig,
    selection: CorpusSelection,
    settings: AnnotateCorpusSettings,
    *,
    fields: Collection[str] = (),
    decades: Collection[int] = (),
    batch_size: int = 256,
    process_count: int = 1,
    progress: Progress = SILENT,
) -> CorpusResult:
    """Annotate the parts of the given fields and decades (all parts when none are given).

    Saving a part replaces it. A part with no works (every work of its cell lists an earlier
    field first) is saved empty, so the next stages find every part. The result counts the
    annotated abstracts per part; this stage drops, removes and flags nothing.

    Raises:
        ValueError: if no raw cell matches the selection or the given fields and decades, or a
            cell's metadata has not been fetched or is incomplete.
        FileNotFoundError: if the glued words or the split words have not been counted.
    """
    splits = glued_word_splits(load_glued_words(catalog))
    joins = split_word_joins(load_split_words(catalog))
    cells = corpus_cells(catalog, selection)
    chosen = {
        part_name(cell.field, cell.decade)
        for cell in cells
        if (not fields or cell.field in fields) and (not decades or cell.decade in decades)
    }
    if not chosen:
        raise ValueError(f"no part of the corpus for fields {fields} in decades {decades}")
    abstracts = load_corpus_works(catalog, cells, progress)
    selected = sorted((abstract for abstract in abstracts if _part(abstract) in chosen), key=_part)
    del abstracts
    sizes = Counter(_part(abstract) for abstract in selected)
    progress.stage("loading the spaCy pipeline")
    nlp = build_pipeline(named_entity_recognition=settings.named_entity_recognition)
    progress.stage("spaCy pass", total=len(selected))
    docs = annotate_texts(
        selected,
        nlp,
        glued_word_splits=splits,
        split_word_joins=joins,
        batch_size=batch_size,
        process_count=process_count,
    )
    # spaCy forks its worker processes when the stream starts. Frozen, the objects already alive
    # (the merged works, the pipeline) are never visited by the workers' garbage collection, so
    # the memory the workers inherit stays shared instead of being copied page by page (Python's
    # documentation, gc.freeze). Unfrozen once the pass is over.
    gc.collect()
    gc.freeze()
    try:
        pairs = _advancing(zip(selected, docs, strict=True), progress)
        for part, group in groupby(pairs, key=lambda pair: _part(pair[0])):
            save_annotations(catalog, (doc for _, doc in group), part=part)
    finally:
        gc.unfreeze()
    for part in sorted(chosen - sizes.keys()):
        save_annotations(catalog, [], part=part)
    return CorpusResult(
        abstract_count=len(selected),
        parts=sorted(chosen),
        dropped_counts={},
        removal_counts={},
        flag_counts={},
    )


def _advancing(items: Iterable[_Item], progress: Progress) -> Iterator[_Item]:
    """The items unchanged, each counted as done once the next is asked for."""
    for item in items:
        yield item
        progress.advance()


def _part(abstract: RawAbstract) -> str:
    return part_name(*part_of(abstract))
