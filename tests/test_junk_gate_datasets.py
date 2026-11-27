"""The junk gate's shared datasets: run 3's records and verdicts, the cells and the labels."""

from __future__ import annotations

import json
from typing import get_args

import polars as pl
import pydantic
import pytest

from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.junk_gate import (
    RUN3_RECORDS_COLUMNS,
    JunkLabel,
    JunkLabels,
    RecordDamage,
    RecordKind,
    RecordSentence,
    RecordVerdict,
    load_junk_cells,
    load_junk_labels,
    load_run3_records,
    load_run3_verdicts,
    save_junk_labels,
    save_run3_records,
)
from interpretant.review.judging import (
    RECORD_DAMAGE,
    RECORD_KINDS,
    RECORD_SENTENCES,
    RECORD_VERDICTS,
)

LABEL = JunkLabel(
    set="enrichment",
    stratum="list_like",
    work_id="https://openalex.org/W1",
    part="ai_1990",
    decade=1990,
    word="attention",
    content_key="0" * 20,
    verdict="junk",
    kind="contents_or_list",
    sentence="other",
    damaged="no",
)


def test_the_labels_allow_exactly_the_codebooks_answers() -> None:
    assert get_args(RecordVerdict) == RECORD_VERDICTS
    assert get_args(RecordKind) == RECORD_KINDS
    assert get_args(RecordSentence) == RECORD_SENTENCES
    assert get_args(RecordDamage) == RECORD_DAMAGE


def test_the_labels_are_read_back_as_written(catalog: CatalogConfig) -> None:
    labels = JunkLabels(codebook_sha256="a" * 64, items={"S0001": LABEL})

    save_junk_labels(catalog, labels)

    assert load_junk_labels(catalog) == labels
    written = json.loads(catalog.junk_labels.path.read_text())
    assert set(written["items"]["S0001"]) == {
        *("set", "stratum", "work_id", "part", "decade", "word", "content_key"),
        *("verdict", "kind", "sentence", "damaged"),
    }


def test_a_label_with_a_note_is_refused(catalog: CatalogConfig) -> None:
    item = LABEL.model_dump() | {"note": "a note can quote the corpus"}
    catalog.junk_labels.path.parent.mkdir(parents=True)
    catalog.junk_labels.path.write_text(
        json.dumps({"codebook_sha256": "a" * 64, "items": {"S0001": item}})
    )

    with pytest.raises(pydantic.ValidationError):
        load_junk_labels(catalog)


def test_the_cells_are_read_by_word_decade_and_flag(catalog: CatalogConfig) -> None:
    catalog.junk_cells.path.parent.mkdir(parents=True)
    pl.DataFrame(
        {"word": ["attention"], "decade": [1990], "flagged": [True], "drawn": [120]}
    ).write_parquet(catalog.junk_cells.path)

    assert load_junk_cells(catalog).to_dicts() == [
        {"word": "attention", "decade": 1990, "flagged": True}
    ]


def test_cells_without_a_flag_are_refused(catalog: CatalogConfig) -> None:
    catalog.junk_cells.path.parent.mkdir(parents=True)
    pl.DataFrame({"word": ["attention"], "decade": [1990]}).write_parquet(catalog.junk_cells.path)

    with pytest.raises(ValueError, match="flagged"):
        load_junk_cells(catalog)


def test_run3s_verdicts_are_its_record_rows_and_its_records_are_read_back(
    catalog: CatalogConfig,
) -> None:
    rows = [
        {"id": "b000", "part": "sentence_breaks", "work_id": "W9", "cuts": 3},
        {"id": "r000", "part": "record", "work_id": "W1", "decade": 1950, "field": "ai"}
        | {"verdict": "unsure", "note": "a note", "judged": "2026-11-14"},
    ]
    records = pl.DataFrame(
        {"work_id": ["W1"], "text": ["An abstract."], "removals": [[]]},
        schema=RUN3_RECORDS_COLUMNS,
    )
    save_run3_records(catalog, records)
    catalog.run3_records.path.with_name("verdicts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )

    [verdict] = load_run3_verdicts(catalog)

    assert (verdict.id, verdict.work_id, verdict.verdict) == ("r000", "W1", "unsure")
    assert load_run3_records(catalog)["text"].to_list() == ["An abstract."]
