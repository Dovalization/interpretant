"""Stage 2b of the run order (decision agenda 9.1, 2.4): each cached sentence's language.

lingua reads the sentences of the annotation cache, part by part in worker processes, and
writes one row per sentence. Only this stage and the ones after it rerun when the candidate
languages, the English threshold or the minimum relative distance change.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import cache, partial

import polars as pl
from lingua import LanguageDetector
from pydantic import Field
from pydantic_settings import SettingsConfigDict
from spacy.vocab import Vocab

from interpretant.catalog import (
    SENTENCE_LANGUAGES_SCHEMA,
    WORK_ID,
    CatalogConfig,
    load_annotations,
    save_sentence_languages,
)
from interpretant.corpus.language import build_language_detector, detect_sentence_languages
from interpretant.pipelines.corpus import corpus_cells, corpus_parts
from interpretant.pipelines.params import CorpusSelection, ParamsSettings
from interpretant.pipelines.progress import SILENT, Progress
from interpretant.pipelines.workers import map_parts


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
    """Detect the language of every cached sentence of the corpus and write it per part.

    Raises:
        ValueError: if no raw cell matches the selection.
        FileNotFoundError: if a part has not been annotated.
    """
    parts = corpus_parts(corpus_cells(catalog, selection))
    progress.stage("detecting sentence languages", total=len(parts))
    results = map_parts(
        partial(_detect_part, catalog, settings),
        dict.fromkeys(parts),
        process_count=process_count,
        progress=progress,
    )
    counts: Counter[str] = Counter()
    for _, sentence_counts in results.values():
        counts.update(sentence_counts)
    return LanguageResult(
        abstract_count=sum(abstract_count for abstract_count, _ in results.values()),
        parts=parts,
        sentence_counts=dict(counts.most_common()),
    )


def _detect_part(
    catalog: CatalogConfig, settings: DetectLanguagesSettings, part: str, _: None
) -> tuple[int, dict[str, int]]:
    """One part's sentence languages, written to the catalog; its abstracts and language counts."""
    detector = _detector(settings.languages, settings.minimum_relative_distance)
    rows: list[tuple[str, int, int, str, float]] = []
    abstract_count = 0
    for doc in load_annotations(catalog, part, Vocab()):
        abstract_count += 1
        languages = detect_sentence_languages(
            doc, detector, english_threshold=settings.english_threshold
        )
        rows.extend(
            (
                doc.user_data[WORK_ID],
                sentence.start_char,
                sentence.end_char,
                language.language,
                language.confidence,
            )
            for sentence, language in zip(doc.sents, languages, strict=True)
        )
    table = pl.DataFrame(rows, schema=SENTENCE_LANGUAGES_SCHEMA, orient="row")
    save_sentence_languages(catalog, table, part=part)
    return abstract_count, dict(Counter(table["language"].to_list()))


@cache
def _detector(languages: tuple[str, ...], minimum_relative_distance: float) -> LanguageDetector:
    """One detector per worker process, built on its first part (about a second to load)."""
    return build_language_detector(languages, minimum_relative_distance=minimum_relative_distance)
