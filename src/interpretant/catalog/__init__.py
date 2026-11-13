"""The data catalog: every dataset by name, loaded and saved with its schema checked.

Only pipelines and experiment runners call it; jobs receive data, never paths (decision agenda,
step 8.3).
"""

from interpretant.catalog.abstracts import load_abstracts, save_abstracts
from interpretant.catalog.config import CATALOG_FILE, CatalogConfig, DatasetEntry, read_catalog
from interpretant.catalog.curated import (
    START_ID,
    CleaningPattern,
    CleaningPatterns,
    Scope,
    load_cleaning_patterns,
)
from interpretant.catalog.schemas import ABSTRACTS_COLUMNS, ABSTRACTS_SCHEMA, REMOVAL, SENTENCE

__all__ = [
    "ABSTRACTS_COLUMNS",
    "ABSTRACTS_SCHEMA",
    "CATALOG_FILE",
    "REMOVAL",
    "SENTENCE",
    "START_ID",
    "CatalogConfig",
    "CleaningPattern",
    "CleaningPatterns",
    "DatasetEntry",
    "Scope",
    "load_abstracts",
    "load_cleaning_patterns",
    "read_catalog",
    "save_abstracts",
]
