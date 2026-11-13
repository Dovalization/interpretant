"""The data catalog: the catalog file, the raw cells, and the abstracts table."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pandera.errors
import polars as pl
import pydantic
import pytest

from interpretant.catalog import (
    CATALOG_FILE,
    CatalogConfig,
    RawCell,
    append_work_metadata,
    fetched_work_ids,
    list_raw_cells,
    load_abstracts,
    load_raw_cell,
    load_work_metadata,
    read_catalog,
    save_abstracts,
)
from tests.abstracts_tables import abstract_row, abstracts_table
from tests.metadata_records import metadata_record, missing_record

SENTENCES = ["Attention helps.", "See https://example.org for more."]
URL_REMOVAL = {"start": 21, "end": 40, "reason": "url"}


def abstracts_row(**changes: Any) -> pl.DataFrame:
    return abstracts_table([abstract_row(1, SENTENCES, **{"removals": [URL_REMOVAL], **changes})])


def write_raw_cell(catalog: CatalogConfig, field: str, decade: int, lines: list[str]) -> None:
    path = catalog.raw_abstracts.path / field / f"{decade}.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as cell:
        cell.writelines(f"{line}\n" for line in lines)


def raw_line(work: int) -> str:
    return json.dumps(
        {
            "id": f"https://openalex.org/W{work}",
            "year": 2017,
            "type": "article",
            "authors": ["https://openalex.org/A1"],
            "abstract": "Models learn senses.",
        }
    )


def test_list_raw_cells_sorts_cells_by_field_and_decade(catalog: CatalogConfig) -> None:
    for field, decade in [("compsci", 2010), ("ai", 2020), ("ai", 1990)]:
        write_raw_cell(catalog, field, decade, [raw_line(1)])
    assert list_raw_cells(catalog) == [
        RawCell("ai", 1990),
        RawCell("ai", 2020),
        RawCell("compsci", 2010),
    ]


def test_load_raw_cell_reads_works_under_their_table_names(catalog: CatalogConfig) -> None:
    write_raw_cell(catalog, "ai", 2010, [raw_line(1), raw_line(2)])
    works = list(load_raw_cell(catalog, RawCell("ai", 2010)))
    assert [work.work_id for work in works] == [
        "https://openalex.org/W1",
        "https://openalex.org/W2",
    ]
    assert works[0].author_ids == ("https://openalex.org/A1",)
    assert works[0].text == "Models learn senses."


def test_load_raw_cell_rejects_a_line_that_is_not_a_work(catalog: CatalogConfig) -> None:
    write_raw_cell(catalog, "ai", 2010, ['{"id": "https://openalex.org/W1", "year": 2017}'])
    with pytest.raises(pydantic.ValidationError):
        list(load_raw_cell(catalog, RawCell("ai", 2010)))


def test_append_work_metadata_adds_to_a_cell_file_across_runs(catalog: CatalogConfig) -> None:
    cell = RawCell("ai", 2010)
    append_work_metadata(catalog, cell, [metadata_record(1)])
    append_work_metadata(catalog, cell, [metadata_record(2), missing_record(3)])
    records = list(load_work_metadata(catalog, cell))
    assert [record.work_id for record in records] == [
        "https://openalex.org/W1",
        "https://openalex.org/W2",
        "https://openalex.org/W3",
    ]
    assert [record.missing for record in records] == [False, False, True]
    assert fetched_work_ids(catalog, cell) == {record.work_id for record in records}


def test_fetched_work_ids_is_empty_before_a_cell_is_fetched(catalog: CatalogConfig) -> None:
    assert fetched_work_ids(catalog, RawCell("ai", 2010)) == set()


def test_append_work_metadata_writes_nothing_when_a_record_is_malformed(
    catalog: CatalogConfig,
) -> None:
    cell = RawCell("ai", 2010)
    with pytest.raises(pydantic.ValidationError):
        append_work_metadata(catalog, cell, [metadata_record(1), {"id": "W2"}])
    assert fetched_work_ids(catalog, cell) == set()


def test_read_catalog_resolves_paths_against_the_catalog_file(
    catalog: CatalogConfig, tmp_path: Path
) -> None:
    assert catalog.abstracts.path == tmp_path / "data/abstracts"


@pytest.mark.parametrize(
    ("original", "broken"),
    [("format: parquet", "format: csv"), ("cleaning_rules:", "cleaning_rule:")],
    ids=["unknown format", "misspelt dataset"],
)
def test_read_catalog_rejects_a_malformed_catalog(
    tmp_path: Path, catalog_text: str, original: str, broken: str
) -> None:
    catalog_file = tmp_path / "catalog.yaml"
    catalog_file.write_text(catalog_text.replace(original, broken))
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
        {"duplicate_of": "W2"},
    ],
    ids=["span past the text", "span ending before it starts", "decade", "no field", "source",
         "work id", "duplicate of"],
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
