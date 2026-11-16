"""Stage 2a of the run order (decision agenda 9.1): the spaCy pass over the repaired texts.

Every selected cell's works were merged, repaired and grouped into parts by the count-word-repairs
stage: a part holds the works whose first field is one field, in one decade, with every repair
made. The pass reads the parts asked for and annotates their texts in one stream, and each part's
documents go to the annotation cache as the stream reaches its end. The documents carry tokens,
sentences, tags and lemmas only: sentence languages (stage 2b) and the cleaning patterns and
keeping rules (stage 2c) read the cache, so changing them does not rerun this stage. A run may
annotate some parts only.
"""

from __future__ import annotations

import gc
from collections.abc import Collection, Iterable, Iterator
from itertools import islice
from typing import TypeVar

from pydantic_settings import SettingsConfigDict

from interpretant.catalog import CatalogConfig, load_repaired_texts, save_annotations
from interpretant.corpus.annotation import annotate_texts, build_pipeline
from interpretant.pipelines.corpus import CorpusResult, corpus_cells, part_name
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

    Each part's works are read from its repaired texts, in their order. Saving a part replaces it;
    a part with no works (every work of its cell lists an earlier field first) is saved empty, so
    the next stages find every part. The result counts the annotated abstracts per part; this
    stage drops, removes and flags nothing.

    Raises:
        ValueError: if no raw cell matches the selection or the given fields and decades.
        FileNotFoundError: if a part's texts have not been repaired.
    """
    chosen = sorted(
        {
            part_name(cell.field, cell.decade)
            for cell in corpus_cells(catalog, selection)
            if (not fields or cell.field in fields) and (not decades or cell.decade in decades)
        }
    )
    if not chosen:
        raise ValueError(f"no part of the corpus for fields {fields} in decades {decades}")
    progress.stage("reading the repaired texts")
    texts = {part: load_repaired_texts(catalog, part) for part in chosen}
    progress.stage("loading the spaCy pipeline")
    nlp = build_pipeline(named_entity_recognition=settings.named_entity_recognition)
    total = sum(part_texts.height for part_texts in texts.values())
    progress.stage("spaCy pass", total=total)
    works = (row for part in chosen for row in texts[part].iter_rows())
    docs = annotate_texts(works, nlp, batch_size=batch_size, process_count=process_count)
    # spaCy forks its worker processes when the stream starts. Frozen, the objects already alive
    # (the texts, the pipeline) are never visited by the workers' garbage collection, so the
    # memory the workers inherit stays shared instead of being copied page by page (Python's
    # documentation, gc.freeze). Unfrozen once the pass is over.
    gc.collect()
    gc.freeze()
    try:
        stream = _advancing(docs, progress)
        for part in chosen:
            save_annotations(catalog, islice(stream, texts[part].height), part=part)
        # Running the stream to its end raises an error met while reading the texts, which
        # ``annotate_texts`` holds until the documents before it are given.
        if next(stream, None) is not None:
            raise RuntimeError("the spaCy pass gave more documents than the parts hold")
    finally:
        gc.unfreeze()
    return CorpusResult(
        abstract_count=total,
        parts=chosen,
        dropped_counts={},
        removal_counts={},
        flag_counts={},
    )


def _advancing(items: Iterable[_Item], progress: Progress) -> Iterator[_Item]:
    """The items unchanged, each counted as done once the next is asked for."""
    for item in items:
        yield item
        progress.advance()
