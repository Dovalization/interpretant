"""Stage 2c of the run order (decision agenda 9.1, 2.3): the cleaning patterns and keeping rules.

Each part's cached documents are matched against the cleaning patterns, in worker processes,
with their sentence languages: each of the pass's sentences takes the language of the base
sentences it overlaps, as the word repairs moved them onto the repaired text, and lingua reads
the few that overlap two languages (docs/plan_early_languages.md). Each abstract gets its
removal ranges, flags and drop reason, and the part's rows go to the annotated table that the
corpus-level rules read. The
raw cells are merged again for each work's metadata and raw text (the mojibake check reads the
raw text). Only this stage and the ones after it rerun when a pattern, a scope or a keeping
threshold changes.
"""

from __future__ import annotations

from dataclasses import asdict
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
    detect_texts,
    languages_by_overlap,
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
from interpretant.pipelines.detect_languages import DetectLanguagesSettings
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

    ``language_settings`` are lingua's, for the sentences whose overlapped base sentences name
    two languages.

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
    results = map_parts(
        partial(
            _clean_part,
            catalog,
            patterns,
            settings.min_english_words,
            (language_settings.languages, language_settings.minimum_relative_distance),
        ),
        {part: abstracts.get(part, []) for part in parts},
        process_count=process_count,
        progress=progress,
    )
    return combine_results(results.values())


def _clean_part(
    catalog: CatalogConfig,
    patterns: CleaningPatterns,
    min_english_words: int,
    detector_settings: tuple[tuple[str, ...], float],
    part: str,
    abstracts: list[RawAbstract],
) -> CorpusResult:
    """One part's annotated rows, written to the catalog; the part's counts."""
    cleaning = build_cleaning_pipeline(patterns)
    signatures = compile_signatures(patterns.mojibake_signatures)
    early = _early_languages_by_work(load_repaired_sentence_languages(catalog, part))
    by_work = {abstract.work_id: abstract for abstract in abstracts}
    rows = []
    for doc in load_annotations(catalog, part, cleaning.vocab):
        work_id = doc.user_data[WORK_ID]
        if work_id not in by_work:
            raise ValueError(f"part {part!r}: {work_id} is cached but not in the part's works")
        rows.append(
            clean_abstract(
                doc,
                by_work[work_id],
                _sentence_languages(doc, early.get(work_id, []), detector_settings),
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
    doc: Doc,
    early: list[tuple[int, int, str, float]],
    detector_settings: tuple[tuple[str, ...], float],
) -> list[SentenceLanguage]:
    """Each of the document's sentences' language, from the base sentences it overlaps.

    lingua reads the sentences whose overlapped base sentences name two languages.
    """
    sentences = list(doc.sents)
    languages = languages_by_overlap(
        [(sentence.start_char, sentence.end_char) for sentence in sentences], early
    )
    unsettled = [index for index, language in enumerate(languages) if language is None]
    if unsettled:
        detected = detect_texts(
            [sentences[index].text for index in unsettled], _detector(*detector_settings)
        )
        for index, language in zip(unsettled, detected, strict=True):
            languages[index] = language
    return [language for language in languages if language is not None]


@cache
def _detector(languages: tuple[str, ...], minimum_relative_distance: float) -> LanguageDetector:
    """Lingua's detector, built once per process and only when a part needs it."""
    return build_language_detector(languages, minimum_relative_distance=minimum_relative_distance)


def _table(rows: list[AnnotatedAbstract]) -> pl.DataFrame:
    records: list[dict[str, Any]] = [asdict(row) for row in rows]
    return pl.DataFrame(records, schema=ABSTRACTS_COLUMNS)
