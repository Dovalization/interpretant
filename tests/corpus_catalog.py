"""A small corpus in a temporary catalog, for the corpus stages' tests.

Three raw cells with their work metadata; the cleaning rules and patterns are the repository's.
Work 1 sits in AI and computer science, work 2 is too short, work 3 is a repository-only record,
work 4 revises work 3's text (a near-duplicate) and work 5 is in a later decade.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Sequence
from pathlib import Path

from interpretant.catalog import CatalogConfig, RawCell, append_work_metadata, read_catalog
from interpretant.pipelines.annotate_corpus import AnnotateCorpusSettings, annotate_corpus
from interpretant.pipelines.build_linked_tables import build_linked_tables
from interpretant.pipelines.clean_corpus import CleanCorpusSettings, clean_corpus
from interpretant.pipelines.corpus import CorpusResult
from interpretant.pipelines.count_word_repairs import (
    CountWordRepairsSettings,
    GluedWordSettings,
    LowerGluedWordSettings,
    SplitWordSettings,
    WordRepairsResult,
    count_word_repairs,
)
from interpretant.pipelines.detect_languages import DetectLanguagesSettings, detect_languages
from interpretant.pipelines.params import CorpusSelection
from interpretant.pipelines.repair_texts import repair_texts
from tests.metadata_records import metadata_record

PARAGRAPH = (
    "Attention mechanisms let a neural network weigh the parts of its input that matter most for "
    "the current prediction. They were first used in machine translation, where the decoder looks "
    "back at every encoded word instead of a single summary vector. Since then the idea has spread "
    "to speech recognition, image captioning and protein structure prediction, and it now sits at "
    "the centre of the transformer architecture. We review how the term attention moved from "
    "psychology into computer science, compare the main families of attention mechanisms, and ask "
    "which properties of human attention the computational versions keep and which they discard. "
    "We close with open questions about interpretability and the cost of long inputs."
)
REVISED = PARAGRAPH.replace("We close with", "We end with")
SHARED = (
    "Language models learn word senses from the contexts in which the words appear, and those "
    "senses drift as fields borrow terms from one another over the decades."
)
CELLS = {
    ("ai", 2010): [(1, SHARED, 2012), (2, "Short note here.", 2012), (3, PARAGRAPH, 2012)],
    ("compsci", 2010): [(1, SHARED, 2012), (4, REVISED, 2015)],
    ("ai", 2020): [(5, SHARED, 2021)],
}
# The two 2010s cells, read by English and German candidates.
SELECTION = CorpusSelection(fields=("ai", "compsci"), decades=(2010,))
ANNOTATE_SETTINGS = AnnotateCorpusSettings(named_entity_recognition=False)
LANGUAGE_SETTINGS = DetectLanguagesSettings(languages=("en", "de"))
CLEAN_SETTINGS = CleanCorpusSettings()
WORD_REPAIRS_SETTINGS = CountWordRepairsSettings(
    glued_words=GluedWordSettings(
        apart_margin=10, opener_count=30, opener_min_left_letters=4, opener_min_right_letters=2
    ),
    split_words=SplitWordSettings(min_word_count=2, join_margin=10, hyphen_margin=100),
    lower_glued_words=LowerGluedWordSettings(
        apart_margin=10, min_part_count=2, min_first_part_letters=2, min_dense_splits=2
    ),
)


def repair_and_count(
    catalog: CatalogConfig,
    selection: CorpusSelection = SELECTION,
    settings: CountWordRepairsSettings = WORD_REPAIRS_SETTINGS,
) -> WordRepairsResult:
    """Repair the texts, read their languages and count the word repairs, as ``dvc repro`` runs
    the three stages."""
    repair_texts(catalog, selection)
    detect_languages(catalog, selection, LANGUAGE_SETTINGS)
    return count_word_repairs(catalog, selection, settings)


def prepare_annotated_corpus(
    catalog: CatalogConfig, selection: CorpusSelection = SELECTION
) -> CorpusResult:
    """Run the stages before the corpus-level rules, as ``dvc repro`` does: the staging tables,
    the text repair, the sentence languages, the word repairs, the spaCy pass and the cleaning;
    return the cleaning's result."""
    build_linked_tables(catalog)
    repair_and_count(catalog, selection)
    annotate_corpus(catalog, selection, ANNOTATE_SETTINGS)
    return clean_corpus(catalog, selection, CLEAN_SETTINGS, LANGUAGE_SETTINGS)


def write_corpus_catalog(folder: Path) -> Path:
    """Write the catalog file, the raw cells and their metadata under ``folder``, and count the
    word repairs, which the spaCy pass reads."""
    repository = read_catalog()
    catalog_file = folder / "catalog.yaml"
    catalog_file.write_text(
        "raw_abstracts:\n  path: raw\n  format: jsonl.gz\n"
        "work_metadata:\n  path: metadata\n  format: jsonl.gz\n"
        "crossref_records:\n  path: crossref\n  format: jsonl.gz\n"
        "pubmed_records:\n  path: pubmed\n  format: jsonl.gz\n"
        "openalex_works:\n  path: openalex_works.parquet\n  format: parquet\n"
        "crossref_works:\n  path: crossref_works.parquet\n  format: parquet\n"
        "pubmed_works:\n  path: pubmed_works.parquet\n  format: parquet\n"
        "annotated_abstracts:\n  path: annotated\n  format: parquet\n"
        "abstracts:\n  path: abstracts\n  format: parquet\n"
        f"cleaning_rules:\n  path: '{repository.cleaning_rules.path}'\n  format: yaml\n"
        f"cleaning_patterns:\n  path: '{repository.cleaning_patterns.path}'\n  format: jsonl\n"
        f"doi_corrections:\n  path: '{repository.doi_corrections.path}'\n  format: yaml\n"
        f"record_types:\n  path: '{repository.record_types.path}'\n  format: yaml\n"
        f"kept_joined_words:\n  path: '{repository.kept_joined_words.path}'\n  format: yaml\n"
        f"kept_apart_words:\n  path: '{repository.kept_apart_words.path}'\n  format: yaml\n"
        "glued_words:\n  path: glued_words.parquet\n  format: parquet\n"
        "split_words:\n  path: split_words.parquet\n  format: parquet\n"
        "lower_glued_words:\n  path: lower_glued_words.parquet\n  format: parquet\n"
        "base_texts:\n  path: base_texts\n  format: parquet\n"
        "repaired_texts:\n  path: repaired_texts\n  format: parquet\n"
        "annotations:\n  path: annotations\n  format: docbin\n"
        "base_sentence_languages:\n  path: base_sentence_languages\n  format: parquet\n"
        "repaired_sentence_languages:\n  path: repaired_sentence_languages\n  format: parquet\n"
        "corpus_report:\n  path: corpus_report.md\n  format: markdown\n"
        "corpus_comparison:\n  path: corpus_comparison.md\n  format: markdown\n"
        "corpus_metrics:\n  path: metrics/corpus.json\n  format: json\n"
        "run_log:\n  path: metrics/runs.jsonl\n  format: jsonl\n"
    )
    for (field, decade), works in CELLS.items():
        write_raw_cell(catalog_file, field, decade, works)
    repair_and_count(read_catalog(catalog_file), CorpusSelection())
    return catalog_file


def write_raw_cell(
    catalog_file: Path, field: str, decade: int, works: Sequence[tuple[int, str, int]]
) -> None:
    """One raw cell of (work, text, year) entries, and its metadata (work 3: repository-only)."""
    path = catalog_file.parent / "raw" / field / f"{decade}.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as cell:
        for work, text, year in works:
            record = {
                "id": f"https://openalex.org/W{work}",
                "year": year,
                "type": "article",
                "authors": [f"https://openalex.org/A{work}"],
                "abstract": text,
            }
            cell.write(json.dumps(record) + "\n")
    records = [
        metadata_record(work, source_type="repository", indexed_in=())
        if work == 3
        else metadata_record(work)
        for work, _, _ in works
    ]
    append_work_metadata(read_catalog(catalog_file), RawCell(field, decade), records)
