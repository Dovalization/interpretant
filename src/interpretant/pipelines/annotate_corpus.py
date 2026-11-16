"""Stage 2a of the run order (decision agenda 9.1): the spaCy pass over the repaired texts.

Every selected cell's works were merged, repaired and grouped into parts by the count-word-repairs
stage: a part holds the works whose first field is one field, in one decade, with every repair
made. Each worker process takes whole parts, largest first: it reads a part's texts, annotates
them in order and writes the part to the annotation cache, so no document crosses processes. The
documents carry tokens, sentences, tags and lemmas only: sentence languages (stage 2b) and the
cleaning patterns and keeping rules (stage 2c) read the cache, so changing them does not rerun
this stage. A run may annotate some parts only.
"""

from __future__ import annotations

from collections.abc import Collection, Iterator
from dataclasses import dataclass
from functools import cache

import polars as pl
from pydantic_settings import SettingsConfigDict
from spacy.language import Language
from spacy.tokens import Doc

from interpretant.catalog.annotations import WORK_ID, save_annotations
from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.repaired_texts import count_repaired_texts, load_repaired_texts
from interpretant.catalog.sentences import SENTENCES_SCHEMA, save_sentences
from interpretant.corpus.annotation import annotate_texts, build_pipeline
from interpretant.pipelines.corpus import CorpusResult, corpus_cells, part_name
from interpretant.pipelines.params import CorpusSelection, ParamsSettings
from interpretant.pipelines.progress import SILENT, Progress
from interpretant.pipelines.workers import map_parts


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
    progress.stage("counting the repaired texts")
    sizes = {part: count_repaired_texts(catalog, part) for part in chosen}
    progress.stage("spaCy pass", total=sum(sizes.values()))
    job = _PassJob(catalog, settings.named_entity_recognition, batch_size)
    map_parts(
        _annotate_part,
        dict.fromkeys(chosen, job),
        process_count=process_count,
        progress=progress,
        sizes=sizes,
    )
    return CorpusResult(
        abstract_count=sum(sizes.values()),
        parts=chosen,
        dropped_counts={},
        removal_counts={},
        flag_counts={},
    )


@dataclass(frozen=True)
class _PassJob:
    """What a worker needs to annotate a part: where it is, and what the models run."""

    catalog: CatalogConfig
    named_entity_recognition: bool
    batch_size: int


def _annotate_part(part: str, job: _PassJob) -> None:
    """Annotate one part's repaired texts in this process; write the cache and the sentences."""
    texts = load_repaired_texts(job.catalog, part)
    nlp = _pipeline(named_entity_recognition=job.named_entity_recognition)
    docs = annotate_texts(texts.iter_rows(), nlp, batch_size=job.batch_size)
    sentences: list[tuple[str, int, int]] = []

    def with_sentences_recorded(annotated: Iterator[Doc]) -> Iterator[Doc]:
        for doc in annotated:
            work_id = doc.user_data[WORK_ID]
            sentences.extend((work_id, span.start_char, span.end_char) for span in doc.sents)
            yield doc

    save_annotations(job.catalog, with_sentences_recorded(docs), part=part)
    table = pl.DataFrame(sentences, schema=SENTENCES_SCHEMA, orient="row")
    save_sentences(job.catalog, table, part=part)


@cache
def _pipeline(*, named_entity_recognition: bool) -> Language:
    """The pass's pipeline, built once per worker process."""
    return build_pipeline(named_entity_recognition=named_entity_recognition)
