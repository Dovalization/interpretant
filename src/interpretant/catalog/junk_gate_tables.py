"""The junk gate's outputs (`docs/junk_gate.md`, "Storage"), written by the build_junk_gate stage.

- **The records' scores** (``junk_scores``): one row per record behind a drawn usage, with its 13
  surface features, the classifier's score and call, and the fold that held it out (L's records
  only; every other record is scored by the final model).
- **The usages' OCR scores** (``usage_ocr_scores``): one row per drawn usage, with the share of
  its sentence's words that are unknown (null under the minimum) and how many words were read.
- **The cells** (``junk_cells``): one row per word (its lemma) and decade, with the screen's
  counts, rate, tail probability and flag; the draws read three of its columns
  (``catalog.junk_gate.load_junk_cells``).
- **The classifier** (``junk_classifier``): the final model as JSON (``review.junk_classifier``
  writes and parses it).
- **The metrics** (``junk_gate_metrics``): the gate's figures, tracked by git, so a word appears by
  its reported name only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl

from interpretant.catalog.config import CatalogConfig

# The classifier's surface features (``review.junk_classifier.FEATURE_NAMES``; a test keeps the two
# in step): the catalog imports no other package, so they are spelt out here
SURFACE_FEATURES = (
    "log_characters",
    "log_words",
    "log_sentences",
    "capitalised_word_share",
    "capitals_word_share",
    "digit_share",
    "punctuation_share",
    "list_separators_per_100_words",
    "has_finite_verb",
    "unknown_word_share",
    "removed_share",
    "ends_with_stop",
    "words_per_sentence",
)
JUNK_SCORES_COLUMNS = pl.Schema(
    {
        "work_id": pl.String(),
        **dict.fromkeys(SURFACE_FEATURES, pl.Float64()),
        "score": pl.Float64(),
        "predicted_junk": pl.Boolean(),
        "fold": pl.Int64(),
    }
)
USAGE_OCR_SCORES_COLUMNS = pl.Schema(
    {
        "work_id": pl.String(),
        "word": pl.String(),
        "content_key": pl.String(),
        "ocr_unknown_share": pl.Float64(),
        "ocr_words": pl.Int64(),
    }
)
JUNK_CELLS_TABLE_COLUMNS = pl.Schema(
    {
        "word": pl.String(),
        "decade": pl.Int64(),
        "drawn": pl.Int64(),
        "predicted_junk": pl.Int64(),
        "decade_rate": pl.Float64(),
        "tail_probability": pl.Float64(),
        "flagged": pl.Boolean(),
        "gated": pl.Boolean(),
    }
)


def save_junk_scores(catalog: CatalogConfig, scores: pl.DataFrame) -> Path:
    """Write the records' scores, replacing the file.

    Raises:
        ValueError: if the columns are not the table's (``JUNK_SCORES_COLUMNS``).
    """
    return _save_table(catalog.junk_scores.path, scores, JUNK_SCORES_COLUMNS)


def save_usage_ocr_scores(catalog: CatalogConfig, scores: pl.DataFrame) -> Path:
    """Write the drawn usages' OCR scores, replacing the file.

    Raises:
        ValueError: if the columns are not the table's (``USAGE_OCR_SCORES_COLUMNS``).
    """
    return _save_table(catalog.usage_ocr_scores.path, scores, USAGE_OCR_SCORES_COLUMNS)


def save_junk_cells(catalog: CatalogConfig, cells: pl.DataFrame) -> Path:
    """Write the cells, replacing the file.

    Raises:
        ValueError: if the columns are not the table's (``JUNK_CELLS_TABLE_COLUMNS``).
    """
    return _save_table(catalog.junk_cells.path, cells, JUNK_CELLS_TABLE_COLUMNS)


def save_junk_classifier(catalog: CatalogConfig, text: str) -> Path:
    """Write the classifier's JSON, replacing the file."""
    return _save_text(catalog.junk_classifier.path, text)


def save_junk_gate_metrics(catalog: CatalogConfig, metrics: Mapping[str, Any]) -> Path:
    """Write the gate's metrics as indented JSON, replacing the file."""
    return _save_text(catalog.junk_gate_metrics.path, json.dumps(metrics, indent=1) + "\n")


def _save_table(path: Path, table: pl.DataFrame, columns: pl.Schema) -> Path:
    """Check the table's columns and write it whole, through a partial file."""
    _check_columns(table, columns, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".parquet.partial")
    table.write_parquet(partial)
    partial.replace(path)
    return path


def _save_text(path: Path, text: str) -> Path:
    """Write the text whole, through a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)
    return path


def _check_columns(table: pl.DataFrame, columns: pl.Schema, path: Path) -> None:
    if table.schema != columns:
        raise ValueError(f"{path.name}: expected the columns {columns}, got {table.schema}")
