"""Stage 2b of the run order (decision agenda 9.1, 2.4): each cached sentence's language.

lingua reads the sentences of the annotation cache, part by part with its multi-threaded
methods, and writes one row per sentence. Only this stage and the ones after it rerun when the
candidate languages, the English threshold or the minimum relative distance change.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import polars as pl
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
    progress: Progress = SILENT,
) -> LanguageResult:
    """Detect the language of every cached sentence of the corpus and write it per part.

    The parts are read one after another; lingua's multi-threaded methods use every core on each
    part, with one copy of the language models.

    Raises:
        ValueError: if no raw cell matches the selection.
        FileNotFoundError: if a part has not been annotated.
    """
    parts = corpus_parts(corpus_cells(catalog, selection))
    progress.stage("loading lingua")
    detector = build_language_detector(
        settings.languages, minimum_relative_distance=settings.minimum_relative_distance
    )
    progress.stage("detecting sentence languages", total=len(parts))
    counts: Counter[str] = Counter()
    abstract_count = 0
    for part in parts:
        docs = list(load_annotations(catalog, part, Vocab()))
        labels = detect_sentence_languages(
            docs, detector, english_threshold=settings.english_threshold
        )
        rows = [
            (
                doc.user_data[WORK_ID],
                sentence.start_char,
                sentence.end_char,
                language.language,
                language.confidence,
            )
            for doc, languages in zip(docs, labels, strict=True)
            for sentence, language in zip(doc.sents, languages, strict=True)
        ]
        table = pl.DataFrame(rows, schema=SENTENCE_LANGUAGES_SCHEMA, orient="row")
        save_sentence_languages(catalog, table, part=part)
        counts.update(table["language"].to_list())
        abstract_count += len(docs)
        progress.advance()
    return LanguageResult(
        abstract_count=abstract_count, parts=parts, sentence_counts=dict(counts.most_common())
    )
