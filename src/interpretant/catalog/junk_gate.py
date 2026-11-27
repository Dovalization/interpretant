"""The junk gate's shared datasets (`docs/junk_gate.md`): run 3's records, the cells and the labels.

- **Run 3's records** (``run3_records``): the records run 3's author judged, with the text and the
  removed ranges they had in that run, tracked by DVC. The author's verdicts sit beside them in
  ``verdicts.jsonl`` (committed): one row per judged record (``part`` ``record``) or sentence-break
  item.
- **The cells** (``junk_cells``): one row per word and decade, written by the gate's scoring stage.
  The draws read each cell's word (its lemma, as in the usages table), decade and whether the screen
  flags it; the stage may add other columns.
- **The labels** (``junk_labels``): each judged item's final answer without its note, beside the
  usage it marks, and the sha256 of the codebook the judges read. Committed, so a word appears by
  its reported name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict

from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.schemas import REMOVAL

# Run 3's verdicts, kept in the folder of its records
RUN3_VERDICTS = "verdicts.jsonl"
RUN3_RECORDS_COLUMNS = pl.Schema(
    {"work_id": pl.String(), "text": pl.String(), "removals": pl.List(REMOVAL)}
)
JUNK_CELLS_COLUMNS = pl.Schema({"word": pl.String(), "decade": pl.Int64(), "flagged": pl.Boolean()})
# The answers of the junk gate's codebook (``review.judging``'s ``RECORD_*``; a test keeps the two
# in step): the catalog imports no other package, so they are spelt out here
RecordVerdict = Literal["abstract", "mixed", "junk"]
RecordKind = Literal[
    "none",
    "citation_or_metadata",
    "publisher_or_web",
    "contents_or_list",
    "archive_or_catalogue",
    "other_work",
    "letter_obituary_erratum_news",
    "fragment",
    "not_english",
    "other",
]
RecordSentence = Literal["own", "other", "none"]
RecordDamage = Literal["yes", "no"]
# The sets whose items are labelled: L, E and the confirmation samples
LabelledSet = Literal["labelled", "enrichment", "confirmation"]


class Run3Verdict(BaseModel):
    """The author's verdict on one of run 3's records, judged on 2026-11-14."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    work_id: str
    decade: int
    field: str
    verdict: Literal["abstract", "not_abstract", "unsure"]


class JunkLabel(BaseModel):
    """One item's final answer, without its note, and the usage it marks.

    ``stratum`` is an enrichment item's; ``word`` is the word's reported name.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    set: LabelledSet
    stratum: str | None
    work_id: str
    part: str
    decade: int
    word: str
    content_key: str
    verdict: RecordVerdict
    kind: RecordKind
    sentence: RecordSentence
    damaged: RecordDamage


class JunkLabels(BaseModel):
    """The labels by item id, and the sha256 of the codebook every judge read."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    codebook_sha256: str
    items: dict[str, JunkLabel]


def load_run3_records(catalog: CatalogConfig) -> pl.DataFrame:
    """Run 3's records: each work's text and removed ranges as they were in that run.

    Raises:
        FileNotFoundError: if the file is missing; it is tracked by DVC.
        ValueError: if its columns are not a work id, a text and removed ranges.
    """
    path = catalog.run3_records.path
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: it is tracked by DVC (dvc pull)")
    records = pl.read_parquet(path)
    _check_columns(records, RUN3_RECORDS_COLUMNS, path)
    return records


def save_run3_records(catalog: CatalogConfig, records: pl.DataFrame) -> Path:
    """Write run 3's records, replacing the file.

    Raises:
        ValueError: if the columns are not a work id, a text and removed ranges.
    """
    path = catalog.run3_records.path
    _check_columns(records, RUN3_RECORDS_COLUMNS, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records.write_parquet(path)
    return path


def load_run3_verdicts(catalog: CatalogConfig) -> list[Run3Verdict]:
    """The author's verdicts on run 3's records, in the file's order; sentence-break items aside.

    Raises:
        FileNotFoundError: if the verdicts file is missing.
        pydantic.ValidationError: if a record's row is malformed.
    """
    path = catalog.run3_records.path.with_name(RUN3_VERDICTS)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [Run3Verdict.model_validate(row) for row in rows if row["part"] == "record"]


def load_junk_cells(catalog: CatalogConfig) -> pl.DataFrame:
    """Each cell's word (its lemma), decade and whether the screen flags it.

    Raises:
        FileNotFoundError: if the stage that writes the cells has not run.
        ValueError: if one of these columns is missing or has another type.
    """
    path = catalog.junk_cells.path
    if not path.exists():
        raise FileNotFoundError(f"no cells table at {path}: the gate's scoring stage writes it")
    cells = pl.read_parquet(path)
    missing = [name for name in JUNK_CELLS_COLUMNS if name not in cells.columns]
    if missing:
        raise ValueError(f"{path} lacks the columns {missing}")
    cells = cells.select(list(JUNK_CELLS_COLUMNS))
    _check_columns(cells, JUNK_CELLS_COLUMNS, path)
    return cells


def load_junk_labels(catalog: CatalogConfig) -> JunkLabels:
    """The judges' labels on the gate's items.

    Raises:
        FileNotFoundError: if no labels have been written.
        pydantic.ValidationError: if a label is malformed.
    """
    path = catalog.junk_labels.path
    if not path.exists():
        raise FileNotFoundError(f"no labels at {path}: score_junk_gate.py labels writes them")
    return JunkLabels.model_validate_json(path.read_text(encoding="utf-8"))


def save_junk_labels(catalog: CatalogConfig, labels: JunkLabels) -> Path:
    """Write the labels, replacing the file."""
    path = catalog.junk_labels.path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(labels.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return path


def _check_columns(table: pl.DataFrame, columns: pl.Schema, path: Path) -> None:
    if table.schema != columns:
        raise ValueError(f"{path.name}: expected the columns {columns}, got {table.schema}")
