"""The data catalog: every dataset by name, loaded and saved with its schema checked.

Only pipelines and experiment runners call it; jobs receive data, never paths (decision agenda,
step 8.3).
"""

from interpretant.catalog.abstracts import (
    load_abstracts,
    load_abstracts_from,
    load_annotated_abstracts,
    save_abstracts,
    save_annotated_abstracts,
)
from interpretant.catalog.annotations import WORK_ID, load_annotations, save_annotations
from interpretant.catalog.config import CATALOG_FILE, CatalogConfig, DatasetEntry, read_catalog
from interpretant.catalog.curated import (
    START_ID,
    CleaningPattern,
    CleaningPatterns,
    DOICorrection,
    Scope,
    load_cleaning_patterns,
    load_doi_corrections,
)
from interpretant.catalog.languages import (
    SENTENCE_LANGUAGES_SCHEMA,
    load_sentence_languages,
    save_sentence_languages,
)
from interpretant.catalog.linked_records import (
    LINKED_SOURCES,
    LinkedSource,
    LinkedWork,
    append_linked_records,
    build_linked_record,
    fetched_linked_work_ids,
    load_linked_records_table,
    select_linked_works,
)
from interpretant.catalog.metadata import (
    WorkMetadata,
    append_work_metadata,
    fetched_work_ids,
    index_work_metadata,
    load_work_metadata,
    load_work_metadata_table,
)
from interpretant.catalog.raw import RawCell, RawWork, list_raw_cells, load_raw_cell
from interpretant.catalog.reports import (
    append_run_record,
    load_corpus_metrics,
    save_corpus_comparison,
    save_corpus_metrics,
    save_corpus_report,
)
from interpretant.catalog.schemas import (
    ABSTRACTS_COLUMNS,
    ABSTRACTS_SCHEMA,
    CROSSREF_WORKS_COLUMNS,
    OPENALEX_WORKS_COLUMNS,
    PUBMED_WORKS_COLUMNS,
    REMOVAL,
    SENTENCE,
)
from interpretant.catalog.staging import StagingSource, load_staging_table, save_staging_table

__all__ = [
    "ABSTRACTS_COLUMNS",
    "ABSTRACTS_SCHEMA",
    "CATALOG_FILE",
    "CROSSREF_WORKS_COLUMNS",
    "LINKED_SOURCES",
    "OPENALEX_WORKS_COLUMNS",
    "PUBMED_WORKS_COLUMNS",
    "REMOVAL",
    "SENTENCE",
    "SENTENCE_LANGUAGES_SCHEMA",
    "START_ID",
    "WORK_ID",
    "CatalogConfig",
    "CleaningPattern",
    "CleaningPatterns",
    "DOICorrection",
    "DatasetEntry",
    "LinkedSource",
    "LinkedWork",
    "RawCell",
    "RawWork",
    "Scope",
    "StagingSource",
    "WorkMetadata",
    "append_linked_records",
    "append_run_record",
    "append_work_metadata",
    "build_linked_record",
    "fetched_linked_work_ids",
    "fetched_work_ids",
    "index_work_metadata",
    "list_raw_cells",
    "load_abstracts",
    "load_abstracts_from",
    "load_annotated_abstracts",
    "load_annotations",
    "load_cleaning_patterns",
    "load_corpus_metrics",
    "load_doi_corrections",
    "load_linked_records_table",
    "load_raw_cell",
    "load_sentence_languages",
    "load_staging_table",
    "load_work_metadata",
    "load_work_metadata_table",
    "read_catalog",
    "save_abstracts",
    "save_annotated_abstracts",
    "save_annotations",
    "save_corpus_comparison",
    "save_corpus_metrics",
    "save_corpus_report",
    "save_sentence_languages",
    "save_staging_table",
    "select_linked_works",
]
