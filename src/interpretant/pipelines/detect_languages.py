"""Stage 1c of the run order (decision agenda 9.1, 2.4): each base sentence's language.

The base texts are the texts before the word repairs, so the repairs and every later stage can
read their languages (docs/plan_early_languages.md). Each worker process takes whole parts: it
splits a part's base texts with the spaCy pass's tokenizer settings and sentence rules and
spaCy's rule-based sentencizer, no models, has lingua read the sentences and writes one row per
sentence. Only this stage and the ones after it rerun when the candidate languages, the English
threshold or the minimum relative distance change.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache

import polars as pl
from lingua import LanguageDetector
from pydantic import Field
from pydantic_settings import SettingsConfigDict
from spacy.language import Language

from interpretant.catalog import (
    SENTENCE_LANGUAGES_SCHEMA,
    CatalogConfig,
    load_base_texts,
    save_base_sentence_languages,
)
from interpretant.corpus.annotation import build_sentence_splitter
from interpretant.corpus.language import build_language_detector, detect_sentence_languages
from interpretant.pipelines.corpus import corpus_cells, corpus_parts
from interpretant.pipelines.params import CorpusSelection, ParamsSettings
from interpretant.pipelines.progress import SILENT, Progress
from interpretant.pipelines.workers import map_parts

# lingua runs its parallel methods on rayon's thread pool, which takes every core unless this
# variable says otherwise (rayon's documentation); a worker process uses one.
_RAYON_THREADS = "RAYON_NUM_THREADS"
# Texts the sentence splitter reads at a time (spaCy's ``Language.pipe``).
_SPLIT_BATCH = 1000


class DetectLanguagesSettings(ParamsSettings):
    """The ``detect_languages`` section of ``params.yaml``: lingua's candidates and thresholds."""

    model_config = SettingsConfigDict(yaml_config_section="detect_languages")

    languages: tuple[str, ...] = Field(min_length=1)
    english_threshold: float = Field(gt=0, le=1)
    minimum_relative_distance: float = Field(ge=0, le=0.99)


@dataclass(frozen=True)
class LanguageResult:
    """What the stage wrote: the abstracts and parts read, and the sentences per language."""

    abstract_count: int
    parts: list[str]
    sentence_counts: dict[str, int]


def detect_languages(
    catalog: CatalogConfig,
    selection: CorpusSelection,
    settings: DetectLanguagesSettings,
    *,
    process_count: int = 1,
    progress: Progress = SILENT,
) -> LanguageResult:
    """Detect the language of every sentence of the base texts and write it per part.

    With worker processes, each holds its own copy of lingua's language models (about 690 MB)
    and runs lingua on one core. With one process lingua uses every core.

    Raises:
        ValueError: if no raw cell matches the selection.
        FileNotFoundError: if a part's texts have not been repaired.
    """
    parts = corpus_parts(corpus_cells(catalog, selection))
    job = _LanguageJob(
        catalog,
        settings.languages,
        settings.english_threshold,
        settings.minimum_relative_distance,
    )
    progress.stage("detecting sentence languages", total=len(parts))
    with one_lingua_thread_per_worker(process_count):
        results = map_parts(
            _detect_part, dict.fromkeys(parts, job), process_count=process_count, progress=progress
        )
    counts: Counter[str] = Counter()
    for _, part_counts in results.values():
        counts.update(part_counts)
    return LanguageResult(
        abstract_count=sum(abstract_count for abstract_count, _ in results.values()),
        parts=parts,
        sentence_counts=dict(counts.most_common()),
    )


@dataclass(frozen=True)
class _LanguageJob:
    """What a worker needs to detect a part's sentence languages."""

    catalog: CatalogConfig
    languages: tuple[str, ...]
    english_threshold: float
    minimum_relative_distance: float


def _detect_part(part: str, job: _LanguageJob) -> tuple[int, Counter[str]]:
    """Detect one part's base sentence languages, write them, and count works and languages."""
    texts = load_base_texts(job.catalog, part)
    docs = list(_sentence_splitter().pipe(texts["text"].to_list(), batch_size=_SPLIT_BATCH))
    detector = _detector(job.languages, job.minimum_relative_distance)
    labels = detect_sentence_languages(docs, detector, english_threshold=job.english_threshold)
    rows = [
        (work_id, sentence.start_char, sentence.end_char, language.language, language.confidence)
        for work_id, doc, languages in zip(texts["work_id"].to_list(), docs, labels, strict=True)
        for sentence, language in zip(doc.sents, languages, strict=True)
    ]
    table = pl.DataFrame(rows, schema=SENTENCE_LANGUAGES_SCHEMA, orient="row")
    save_base_sentence_languages(job.catalog, table, part=part)
    return len(docs), Counter(table["language"].to_list())


@cache
def _sentence_splitter() -> Language:
    """The rule-based sentence splitter, built once per process."""
    return build_sentence_splitter()


@cache
def _detector(languages: tuple[str, ...], minimum_relative_distance: float) -> LanguageDetector:
    """Lingua's detector, built once per process."""
    return build_language_detector(languages, minimum_relative_distance=minimum_relative_distance)


@contextmanager
def one_lingua_thread_per_worker(process_count: int) -> Iterator[None]:
    """Start worker processes with rayon limited to one thread; the main process is unchanged."""
    if process_count == 1:
        yield
        return
    previous = os.environ.get(_RAYON_THREADS)
    os.environ[_RAYON_THREADS] = "1"
    try:
        yield
    finally:
        if previous is None:
            del os.environ[_RAYON_THREADS]
        else:
            os.environ[_RAYON_THREADS] = previous
