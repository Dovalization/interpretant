"""Language before the word repairs: lingua's verdict on every sentence of the base texts.

The base texts are the texts before the word repairs, so the repairs and every later stage can
read their languages (docs/plan_early_languages.md). Each is split by the spaCy pass's tokenizer
settings and sentence rules with spaCy's rule-based sentencizer, no models, and lingua reads the
sentences as the cached-sentence stage does, in the same worker processes.
"""

from __future__ import annotations

from collections import Counter
from functools import cache

from spacy.language import Language

from interpretant.catalog import CatalogConfig, load_base_texts, save_base_sentence_languages
from interpretant.corpus.annotation import build_sentence_splitter
from interpretant.pipelines.detect_languages import (
    DetectLanguagesSettings,
    LanguageJob,
    LanguageResult,
    detect_parts,
    language_table,
)
from interpretant.pipelines.params import CorpusSelection
from interpretant.pipelines.progress import SILENT, Progress

# Texts the sentence splitter reads at a time (spaCy's ``Language.pipe``).
_SPLIT_BATCH = 1000


def detect_base_languages(
    catalog: CatalogConfig,
    selection: CorpusSelection,
    settings: DetectLanguagesSettings,
    *,
    process_count: int = 1,
    progress: Progress = SILENT,
) -> LanguageResult:
    """Detect the language of every sentence of the base texts and write it per part.

    The base texts are the texts before the word repairs, so the repairs can read their
    languages (docs/plan_early_languages.md). Each is split by ``build_sentence_splitter``: the
    spaCy pass's tokenizer settings and sentence rules with the rule-based sentencizer, no
    models. lingua reads the sentences as ``detect_languages`` does, in the same worker processes.

    Raises:
        ValueError: if no raw cell matches the selection.
        FileNotFoundError: if a part's texts have not been repaired.
    """
    return detect_parts(
        "detecting base sentence languages",
        _detect_base_part,
        catalog,
        selection,
        settings,
        process_count=process_count,
        progress=progress,
    )


def _detect_base_part(part: str, job: LanguageJob) -> tuple[int, Counter[str]]:
    """Detect one part's base sentence languages, write them, and count works and languages."""
    texts = load_base_texts(job.catalog, part)
    docs = list(_sentence_splitter().pipe(texts["text"].to_list(), batch_size=_SPLIT_BATCH))
    table = language_table(texts["work_id"].to_list(), docs, job)
    save_base_sentence_languages(job.catalog, table, part=part)
    return len(docs), Counter(table["language"].to_list())


@cache
def _sentence_splitter() -> Language:
    """The rule-based sentence splitter, built once per process."""
    return build_sentence_splitter()
