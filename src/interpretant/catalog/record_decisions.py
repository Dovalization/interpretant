"""The decision models' scores of the records (`docs/junk_gate.md`, the deviation of 2026-11-28).

Two passes, each in its own folder under the dataset's path: ``screen`` (every record behind a
drawn usage) and ``confirm`` (the records the screening model could call junk, and L's and E's).
A pass holds numbered parts, each one chunk of records with its model's probability that the
record is not the work's own prose, and a manifest of what those scores depend on. Parts are only
ever added, so a stopped run resumes where it stopped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pandera.polars as pa
import polars as pl

from interpretant.catalog.config import CatalogConfig

Pass = Literal["screen", "confirm"]
PASSES: tuple[Pass, ...] = ("screen", "confirm")
RECORD_DECISIONS_COLUMNS = pl.Schema({"work_id": pl.String(), "probability": pl.Float64()})
_PART_PREFIX = "part-"
_SUFFIX = ".parquet"
_MANIFEST = "manifest.json"

RECORD_DECISIONS_SCHEMA = pa.DataFrameSchema(
    {
        "work_id": pa.Column(
            RECORD_DECISIONS_COLUMNS["work_id"],
            checks=pa.Check.str_startswith("https://openalex.org/W"),
        ),
        "probability": pa.Column(
            RECORD_DECISIONS_COLUMNS["probability"], checks=pa.Check.in_range(0.0, 1.0)
        ),
    },
    unique=["work_id"],
    strict=True,
    name="record_decisions",
    description="One row per record scored in a pass: the probability it is not its own prose.",
)


def pass_folder(root: Path, pass_name: Pass) -> Path:
    """The folder of one pass under the dataset's root."""
    return root / pass_name


def default_root(catalog: CatalogConfig) -> Path:
    """The dataset's root in the catalog."""
    return catalog.record_decisions.path


def save_decision_part(root: Path, pass_name: Pass, scores: pl.DataFrame) -> Path:
    """Check one chunk's scores against the schema and add it as the pass's next part.

    Raises:
        pandera.errors.SchemaError: if the rows break the schema; nothing is written.
    """
    validated: pl.DataFrame = RECORD_DECISIONS_SCHEMA.validate(scores)
    folder = pass_folder(root, pass_name)
    folder.mkdir(parents=True, exist_ok=True)
    numbers = [int(path.stem.removeprefix(_PART_PREFIX)) for path in _parts(folder)]
    target = folder / f"{_PART_PREFIX}{max(numbers, default=-1) + 1:05d}{_SUFFIX}"
    staging = target.with_suffix(f"{_SUFFIX}.partial")
    validated.write_parquet(staging)
    staging.replace(target)
    return target


def load_decisions(root: Path, pass_name: Pass) -> pl.DataFrame:
    """Every part of a pass as one table, sorted by work id; empty when none is saved.

    Raises:
        ValueError: if two parts score the same record.
    """
    parts = _parts(pass_folder(root, pass_name))
    if not parts:
        return pl.DataFrame(schema=RECORD_DECISIONS_COLUMNS)
    scores = pl.read_parquet(parts, schema=RECORD_DECISIONS_COLUMNS).sort("work_id")
    if scores["work_id"].is_duplicated().any():
        raise ValueError(f"two parts of the {pass_name} pass score the same record")
    return scores


def read_pass_manifest(root: Path, pass_name: Pass) -> dict[str, Any]:
    """The pass's manifest, empty when the pass has none yet."""
    path = pass_folder(root, pass_name) / _MANIFEST
    if not path.exists():
        return {}
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return manifest


def write_pass_manifest(root: Path, pass_name: Pass, manifest: dict[str, Any]) -> Path:
    """Write the pass's manifest."""
    path = pass_folder(root, pass_name) / _MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _parts(folder: Path) -> list[Path]:
    return sorted(folder.glob(f"{_PART_PREFIX}*{_SUFFIX}")) if folder.exists() else []
