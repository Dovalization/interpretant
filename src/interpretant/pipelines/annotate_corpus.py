"""Stage 2 of the run order (decision agenda 9.1): the spaCy pass over the raw OpenAlex cells.

Every selected cell's works are merged, so a work is one row listing all its fields, and grouped
into parts: a part holds the works whose first field is one field, in one decade. The works of
the parts asked for are repaired and annotated in one stream; each part's documents go to the
annotation cache and its rows to the annotated table as the stream reaches its end. A run may
annotate some parts only; every work keeps its full list of fields either way. Every selected
cell's metadata must have been fetched (``interpretant fetch-metadata``), so a missing fetch stops
the run before the slow pass, not after it.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import asdict
from itertools import groupby
from typing import Any

import polars as pl
from pydantic import Field
from pydantic_settings import SettingsConfigDict

from interpretant.catalog import (
    ABSTRACTS_COLUMNS,
    CatalogConfig,
    load_cleaning_patterns,
    load_raw_cell,
    save_annotated_abstracts,
    save_annotations,
)
from interpretant.corpus.annotation import (
    AnnotatedAbstract,
    RawAbstract,
    annotate_abstracts,
    build_pipeline,
)
from interpretant.corpus.deduplication import merge_cells
from interpretant.corpus.work_metadata import index_work_metadata
from interpretant.pipelines.corpus import (
    CorpusResult,
    corpus_cells,
    load_corpus_metadata,
    part_name,
    summarise,
)
from interpretant.pipelines.params import CorpusSelection, ParamsSettings

REGULAR = "regular"


class AnnotateCorpusSettings(ParamsSettings):
    """The ``annotate_corpus`` section of ``params.yaml``: lingua's settings and its drop rule."""

    model_config = SettingsConfigDict(yaml_config_section="annotate_corpus")

    languages: tuple[str, ...] = Field(min_length=1)
    english_threshold: float = Field(gt=0, le=1)
    max_not_english_share: float = Field(ge=0, le=1)


def annotate_corpus(
    catalog: CatalogConfig,
    selection: CorpusSelection,
    settings: AnnotateCorpusSettings,
    *,
    fields: Collection[str] = (),
    decades: Collection[int] = (),
    batch_size: int = 256,
    process_count: int = 1,
) -> CorpusResult:
    """Annotate the parts of the given fields and decades (all parts when none are given).

    Saving a part replaces it. A part with no works (every work of its cell lists an earlier
    field first) is saved empty, so the next stage finds every part.

    Raises:
        ValueError: if no raw cell matches the selection or the given fields and decades, or a
            cell's metadata has not been fetched or is incomplete.
    """
    cells = corpus_cells(catalog, selection)
    chosen = {
        part_name(cell.field, cell.decade)
        for cell in cells
        if (not fields or cell.field in fields) and (not decades or cell.decade in decades)
    }
    if not chosen:
        raise ValueError(f"no part of the corpus for fields {fields} in decades {decades}")
    abstracts = merge_cells(
        ((cell.field, load_raw_cell(catalog, cell)) for cell in cells), source=REGULAR
    )
    index_work_metadata(
        load_corpus_metadata(catalog, cells), (abstract.work_id for abstract in abstracts)
    )
    selected = sorted(
        (abstract for abstract in abstracts if _part_of(abstract) in chosen), key=_part_of
    )
    patterns = load_cleaning_patterns(catalog)
    nlp = build_pipeline(
        patterns, languages=settings.languages, english_threshold=settings.english_threshold
    )
    annotated = annotate_abstracts(
        selected,
        nlp,
        patterns,
        max_not_english_share=settings.max_not_english_share,
        batch_size=batch_size,
        process_count=process_count,
    )
    tables: dict[str, pl.DataFrame] = {}
    for part, pairs in groupby(annotated, key=lambda pair: _part_of(pair[1])):
        docs, rows = zip(*pairs, strict=True)
        save_annotations(catalog, docs, part=part)
        tables[part] = _table(rows)
        save_annotated_abstracts(catalog, tables[part], part=part)
    for part in sorted(chosen - tables.keys()):
        save_annotations(catalog, [], part=part)
        tables[part] = _table(())
        save_annotated_abstracts(catalog, tables[part], part=part)
    return summarise(dict(sorted(tables.items())))


def _part_of(abstract: RawAbstract | AnnotatedAbstract) -> str:
    return part_name(abstract.fields[0], abstract.year - abstract.year % 10)


def _table(rows: tuple[AnnotatedAbstract, ...]) -> pl.DataFrame:
    records: list[dict[str, Any]] = [asdict(row) for row in rows]
    return pl.DataFrame(records, schema=ABSTRACTS_COLUMNS)
