"""Stage 2c of the run order (decision agenda 9.1, 2.3): the cleaning patterns and keeping rules.

Each part's cached documents are matched against the cleaning patterns, in worker processes,
with their sentence languages: a pass sentence with the same boundaries as a base sentence, as
the word repairs moved it onto the repaired text, takes its language; lingua reads the others
(6% of the pass's sentences, in 13% of the abstracts) as the language stage reads sentences
(docs/plan_early_languages.md). Each abstract gets its removal ranges, flags and drop reason,
and the part's rows go to the annotated table that the corpus-level rules read. The raw cells
are merged again for each work's metadata and raw text (the mojibake check reads the raw text).
Only this stage and the ones after it rerun when a pattern, a scope, a keeping threshold or a
language setting changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cache, partial
from typing import Any

import polars as pl
from lingua import LanguageDetector
from pydantic import Field
from pydantic_settings import SettingsConfigDict
from spacy.tokens import Doc

from interpretant.catalog import (
    ABSTRACTS_COLUMNS,
    WORK_ID,
    CatalogConfig,
    CleaningPatterns,
    load_annotations,
    load_cleaning_patterns,
    load_repaired_sentence_languages,
    save_annotated_abstracts,
)
from interpretant.corpus.language import (
    SentenceLanguage,
    build_language_detector,
    detect_sentence_languages,
    languages_by_boundaries,
)
from interpretant.corpus.removals import AnnotatedAbstract, build_cleaning_pipeline, clean_abstract
from interpretant.corpus.repair import compile_signatures
from interpretant.corpus.works import RawAbstract
from interpretant.pipelines.corpus import (
    CorpusResult,
    combine_results,
    corpus_cells,
    corpus_parts,
    load_corpus_works,
    summarise,
    works_by_part,
)
from interpretant.pipelines.detect_languages import (
    DetectLanguagesSettings,
    one_lingua_thread_per_worker,
)
from interpretant.pipelines.params import CorpusSelection, ParamsSettings
from interpretant.pipelines.progress import SILENT, Progress
from interpretant.pipelines.workers import map_parts


class CleanCorpusSettings(ParamsSettings):
    """The ``clean_corpus`` section of ``params.yaml``: the keeping thresholds."""

    model_config = SettingsConfigDict(yaml_config_section="clean_corpus")

    min_english_words: int = Field(ge=1)


def clean_corpus(
    catalog: CatalogConfig,
    selection: CorpusSelection,
    settings: CleanCorpusSettings,
    language_settings: DetectLanguagesSettings,
    *,
    process_count: int = 1,
    progress: Progress = SILENT,
) -> CorpusResult:
    """Apply the cleaning patterns and keeping rules to every part and write the annotated table.

    ``language_settings`` are lingua's, for the pass sentences no base sentence matches. With
    worker processes, each one that needs lingua holds its own copy of the language models and
    runs lingua on one core.

    Raises:
        ValueError: if no raw cell matches the selection, a cell's metadata is missing, or a
            part's cache or sentence languages do not hold exactly the part's works.
        FileNotFoundError: if a part has not been annotated or its languages detected.
    """
    cells = corpus_cells(catalog, selection)
    abstracts = works_by_part(load_corpus_works(catalog, cells, progress))
    patterns = load_cleaning_patterns(catalog)
    parts = corpus_parts(cells)
    progress.stage("cleaning the parts", total=len(parts))
    check = _LanguageCheck(
        language_settings.languages,
        language_settings.english_threshold,
        language_settings.minimum_relative_distance,
    )
    with one_lingua_thread_per_worker(process_count):
        results = map_parts(
            partial(_clean_part, catalog, patterns, settings.min_english_words, check),
            {part: abstracts.get(part, []) for part in parts},
            process_count=process_count,
            progress=progress,
        )
    return combine_results(results.values())


@dataclass(frozen=True)
class _LanguageCheck:
    """lingua's settings, for the pass sentences no base sentence matches."""

    languages: tuple[str, ...]
    english_threshold: float
    minimum_relative_distance: float


def _clean_part(
    catalog: CatalogConfig,
    patterns: CleaningPatterns,
    min_english_words: int,
    check: _LanguageCheck,
    part: str,
    abstracts: list[RawAbstract],
) -> CorpusResult:
    """One part's annotated rows, written to the catalog; the part's counts."""
    cleaning = build_cleaning_pipeline(patterns)
    signatures = compile_signatures(patterns.mojibake_signatures)
    by_work = {abstract.work_id: abstract for abstract in abstracts}
    docs = list(load_annotations(catalog, part, cleaning.vocab))
    for doc in docs:
        if doc.user_data[WORK_ID] not in by_work:
            raise ValueError(
                f"part {part!r}: {doc.user_data[WORK_ID]} is cached but not in the part's works"
            )
    early = _early_languages_by_work(load_repaired_sentence_languages(catalog, part))
    rows = []
    for doc, languages in zip(docs, _sentence_languages(docs, early, check), strict=True):
        rows.append(
            clean_abstract(
                doc,
                by_work[doc.user_data[WORK_ID]],
                languages,
                cleaning=cleaning,
                patterns=patterns,
                signatures=signatures,
                min_english_words=min_english_words,
            )
        )
    if len(rows) != len(by_work):
        raise ValueError(
            f"part {part!r}: {len(rows)} cached documents for {len(by_work)} works; "
            "rerun annotate-corpus"
        )
    table = _table(rows)
    save_annotated_abstracts(catalog, table, part=part)
    return summarise({part: table})


def _early_languages_by_work(
    table: pl.DataFrame,
) -> dict[str, list[tuple[int, int, str, float]]]:
    languages: dict[str, list[tuple[int, int, str, float]]] = {}
    for work_id, start, end, language, confidence in table.iter_rows():
        languages.setdefault(work_id, []).append((start, end, language, confidence))
    return languages


def _sentence_languages(
    docs: list[Doc],
    early: dict[str, list[tuple[int, int, str, float]]],
    check: _LanguageCheck,
) -> list[list[SentenceLanguage]]:
    """Each document's sentence languages: a base sentence's where the boundaries are the same.

    The documents with a sentence no base sentence matches are read by lingua as the language
    stage reads them, the whole abstract first (``detect_sentence_languages``), and those
    sentences take lingua's verdict.
    """
    matched = [
        languages_by_boundaries(
            [(sentence.start_char, sentence.end_char) for sentence in doc.sents],
            early.get(doc.user_data[WORK_ID], []),
        )
        for doc in docs
    ]
    unsettled = [index for index, languages in enumerate(matched) if None in languages]
    read: dict[int, list[SentenceLanguage]] = {}
    if unsettled:
        detected = detect_sentence_languages(
            [docs[index] for index in unsettled],
            _detector(check.languages, check.minimum_relative_distance),
            english_threshold=check.english_threshold,
        )
        read = dict(zip(unsettled, detected, strict=True))
    return [
        [
            language if language is not None else read[index][position]
            for position, language in enumerate(languages)
        ]
        for index, languages in enumerate(matched)
    ]


@cache
def _detector(languages: tuple[str, ...], minimum_relative_distance: float) -> LanguageDetector:
    """Lingua's detector, built once per process and only when a part needs it."""
    return build_language_detector(languages, minimum_relative_distance=minimum_relative_distance)


def _table(rows: list[AnnotatedAbstract]) -> pl.DataFrame:
    records: list[dict[str, Any]] = [asdict(row) for row in rows]
    return pl.DataFrame(records, schema=ABSTRACTS_COLUMNS)
