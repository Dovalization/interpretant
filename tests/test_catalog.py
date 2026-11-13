"""The data catalog: reading the catalog file, and saving and loading the abstracts table."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandera.errors
import polars as pl
import pydantic
import pytest

from interpretant.catalog import (
    ABSTRACTS_COLUMNS,
    CATALOG_FILE,
    CatalogConfig,
    load_abstracts,
    read_catalog,
    save_abstracts,
)

TEXT = "Attention helps. See https://example.org for more."


def abstracts_row(**changes: Any) -> pl.DataFrame:
    row: dict[str, Any] = {
        "work_id": "https://openalex.org/W1",
        "year": 2017,
        "decade": 2010,
        "fields": ["ai"],
        "source": "regular",
        "document_type": "article",
        "author_ids": ["https://openalex.org/A1"],
        "text": TEXT,
        "sentences": [{"start": 0, "end": 16, "language": "en", "confidence": 0.99}],
        "removals": [{"start": 21, "end": 40, "reason": "url"}],
        "dropped_reason": None,
    }
    row.update(changes)
    return pl.DataFrame([row], schema=ABSTRACTS_COLUMNS)


@pytest.fixture
def catalog(tmp_path: Path) -> CatalogConfig:
    catalog_file = tmp_path / "catalog.yaml"
    catalog_file.write_text("abstracts:\n  path: data/abstracts\n  format: parquet\n")
    return read_catalog(catalog_file)


def test_read_catalog_resolves_paths_against_the_catalog_file(
    catalog: CatalogConfig, tmp_path: Path
) -> None:
    assert catalog.abstracts.path == tmp_path / "data/abstracts"


def test_read_catalog_rejects_an_unknown_format(tmp_path: Path) -> None:
    catalog_file = tmp_path / "catalog.yaml"
    catalog_file.write_text("abstracts:\n  path: data/abstracts\n  format: csv\n")
    with pytest.raises(pydantic.ValidationError):
        read_catalog(catalog_file)


def test_read_catalog_rejects_a_misspelt_dataset(tmp_path: Path) -> None:
    catalog_file = tmp_path / "catalog.yaml"
    catalog_file.write_text("abstract:\n  path: data/abstracts\n  format: parquet\n")
    with pytest.raises(pydantic.ValidationError):
        read_catalog(catalog_file)


def test_the_repository_catalog_file_is_valid() -> None:
    assert read_catalog(CATALOG_FILE).abstracts.format == "parquet"


def test_save_then_load_returns_the_same_abstracts(catalog: CatalogConfig) -> None:
    saved = pl.concat(
        [abstracts_row(), abstracts_row(work_id="https://openalex.org/W2", removals=[])]
    )
    written = save_abstracts(catalog, saved, part="ai_2010")
    assert written == catalog.abstracts.path / "ai_2010.parquet"
    assert load_abstracts(catalog).collect().equals(saved)


def test_saving_a_part_again_replaces_it(catalog: CatalogConfig) -> None:
    save_abstracts(catalog, abstracts_row(), part="ai_2010")
    save_abstracts(catalog, abstracts_row(year=2018), part="ai_2010")
    assert load_abstracts(catalog).collect()["year"].to_list() == [2018]


@pytest.mark.parametrize(
    "changes",
    [
        {"removals": [{"start": 21, "end": 400, "reason": "url"}]},
        {"sentences": [{"start": 10, "end": 5, "language": "en", "confidence": 0.9}]},
        {"decade": 2000},
        {"fields": []},
        {"source": "somewhere"},
        {"work_id": "W1"},
    ],
    ids=["span past the text", "span ending before it starts", "decade", "no field", "source",
         "work id"],
)  # fmt: skip
def test_save_rejects_rows_that_break_the_schema(
    catalog: CatalogConfig, changes: dict[str, Any]
) -> None:
    with pytest.raises(pandera.errors.SchemaError):
        save_abstracts(catalog, abstracts_row(**changes), part="ai_2010")
    assert not catalog.abstracts.path.exists() or not any(catalog.abstracts.path.iterdir())


def test_save_rejects_duplicate_work_ids(catalog: CatalogConfig) -> None:
    with pytest.raises(pandera.errors.SchemaError):
        save_abstracts(catalog, pl.concat([abstracts_row(), abstracts_row()]), part="ai_2010")


def test_save_rejects_an_unsafe_part_name(catalog: CatalogConfig) -> None:
    with pytest.raises(ValueError, match="part name"):
        save_abstracts(catalog, abstracts_row(), part="../outside")


def test_load_without_saved_parts_raises(catalog: CatalogConfig) -> None:
    with pytest.raises(FileNotFoundError):
        load_abstracts(catalog)
