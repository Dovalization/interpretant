"""Stage 3 of the run order (decision agenda 9.1): the corpus-level rules over the annotated table.

Reads every part of the selected cells as the spaCy pass left it (``annotate-corpus``), keeps,
flags or drops each record by its kind from the staging tables (errata, obituaries, book
reviews and the like, before the other rules count it: 2.3), runs the corpus-level rules over
the whole table at once (template sentences, repeated sentences, near-duplicates, the author
cap: 2.3, 2.5), flags abstracts for the corpus review from their works' OpenAlex metadata and
from their text (4b), and saves the abstracts table's parts. The rules compare abstracts across
the corpus, so this stage always reads every part; it takes minutes, so changing a rule's
threshold never redoes the pass.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from pydantic import Field
from pydantic_settings import SettingsConfigDict

from interpretant.catalog import (
    CatalogConfig,
    StagingSource,
    load_annotated_abstracts,
    load_record_kinds,
    load_staging_table,
    save_abstracts,
)
from interpretant.corpus.cleaning import (
    mark_author_cap,
    mark_repeated_sentences,
    mark_template_sentences,
    mark_truncated,
)
from interpretant.corpus.deduplication import mark_near_duplicates, minhash_signatures
from interpretant.corpus.record_types import (
    decide_record_types,
    join_record_signals,
    mark_record_types,
)
from interpretant.corpus.work_metadata import check_work_metadata, mark_metadata_flags
from interpretant.pipelines.corpus import CorpusResult, corpus_cells, part_name, summarise
from interpretant.pipelines.params import CorpusSelection, ParamsSettings
from interpretant.pipelines.progress import SILENT, Progress
from interpretant.pipelines.workers import map_parts

_PART = "part"
_STAGING_SOURCES: tuple[StagingSource, ...] = ("openalex", "crossref", "pubmed")


class BuildCorpusSettings(ParamsSettings):
    """The ``build_corpus`` section of ``params.yaml``: the corpus-level rules' thresholds."""

    model_config = SettingsConfigDict(yaml_config_section="build_corpus")

    template_min_abstracts: int = Field(ge=2)
    near_duplicate_threshold: float = Field(gt=0, le=1)
    near_duplicate_candidate_threshold: float = Field(gt=0, le=1)
    author_cap_share: float = Field(gt=0, le=1)
    author_cap_floor: int = Field(ge=1)


def build_corpus(
    catalog: CatalogConfig,
    selection: CorpusSelection,
    settings: BuildCorpusSettings,
    *,
    process_count: int = 1,
    progress: Progress = SILENT,
) -> CorpusResult:
    """Apply the corpus-level rules and the review flags to every annotated part, and save them.

    The near-duplicate signatures are made by part in ``process_count`` worker processes.

    Raises:
        ValueError: if no raw cell matches the selection, or a work has no row in OpenAlex's
            staging table (its metadata has not been fetched).
        FileNotFoundError: naming the parts the spaCy pass has not annotated yet, or if the
            staging tables have not been built.
    """
    cells = corpus_cells(catalog, selection)
    parts = [part_name(cell.field, cell.decade) for cell in cells]
    progress.stage("loading the annotated parts")
    annotated = load_annotated_abstracts(catalog, parts)
    table = pl.concat(
        part.with_columns(pl.lit(name).alias(_PART)) for name, part in annotated.items()
    )
    progress.stage("loading the staging tables")
    staging = {source: load_staging_table(catalog, source).collect() for source in _STAGING_SOURCES}
    check_work_metadata(table, staging["openalex"])
    progress.stage("record types")
    kinds = load_record_kinds(catalog)
    signals = join_record_signals(
        table, staging["openalex"], staging["crossref"], staging["pubmed"]
    )
    table = mark_record_types(table, decide_record_types(signals, kinds), kinds)
    progress.stage("template sentences")
    table = mark_template_sentences(table, min_abstracts=settings.template_min_abstracts)
    progress.stage("repeated sentences")
    table = mark_repeated_sentences(table)
    candidates = table.filter(pl.col("dropped_reason").is_null())
    progress.stage("near-duplicate signatures", total=candidates.height)
    signatures = _signatures(candidates, process_count=process_count, progress=progress)
    progress.stage("near-duplicates", total=candidates.height)
    table = mark_near_duplicates(
        table,
        threshold=settings.near_duplicate_threshold,
        candidate_threshold=settings.near_duplicate_candidate_threshold,
        signatures=signatures,
        advance=progress.advance,
    )
    progress.stage("author cap")
    table = mark_author_cap(
        table, max_share=settings.author_cap_share, floor=settings.author_cap_floor
    )
    progress.stage("review flags")
    table = mark_truncated(mark_metadata_flags(table, staging["openalex"]))
    tables = {part: table.filter(pl.col(_PART) == part).drop(_PART) for part in parts}
    progress.stage("saving the parts", total=len(tables))
    for part, rows in tables.items():
        save_abstracts(catalog, rows, part=part)
        progress.advance()
    return summarise(tables)


def _signatures(
    candidates: pl.DataFrame, *, process_count: int, progress: Progress
) -> dict[str, np.ndarray]:
    """Each candidate's MinHash signature by work id, made by part in worker processes."""
    by_part = candidates.partition_by(_PART, as_dict=True, maintain_order=True)
    inputs = {
        str(part): (rows["work_id"].to_list(), rows["text"].to_list(), rows["removals"].to_list())
        for (part,), rows in by_part.items()
    }
    results = map_parts(
        _part_signatures,
        inputs,
        process_count=process_count,
        progress=progress,
        sizes={part: len(rows[0]) for part, rows in inputs.items()},
    )
    return {
        work_id: signature
        for (work_ids, _, _), (signatures, has_shingles) in zip(
            inputs.values(), results.values(), strict=True
        )
        for work_id, signature, has_any in zip(work_ids, signatures, has_shingles, strict=True)
        if has_any
    }


def _part_signatures(
    part: str, rows: tuple[list[str], list[str], list[list[dict[str, object]]]]
) -> tuple[np.ndarray, np.ndarray]:
    """One part's signatures (``minhash_signatures``), in a worker process."""
    _, texts, removals = rows
    return minhash_signatures(texts, removals)
