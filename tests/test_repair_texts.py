"""The repair-texts stage: the works merged and repaired into every part's base texts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from interpretant.__main__ import cli
from interpretant.catalog import load_base_texts, read_catalog
from interpretant.pipelines.params import CorpusSelection
from interpretant.pipelines.repair_texts import repair_texts
from tests.corpus_catalog import SHARED, write_raw_cell

GLUED_STOP = "Models learn senses.The effect is large and lasting."
HCI_SELECTION = CorpusSelection(fields=("ai", "compsci", "hci"), decades=(2010,))


def test_repair_texts_writes_every_part_s_repaired_texts(corpus_catalog_file: Path) -> None:
    write_raw_cell(corpus_catalog_file, "hci", 2010, [(70, GLUED_STOP, 2012)])
    catalog = read_catalog(corpus_catalog_file)
    result = repair_texts(catalog, HCI_SELECTION)
    assert result.parts == ["ai_2010", "compsci_2010", "hci_2010"]
    hci = load_base_texts(catalog, "hci_2010")
    assert hci.rows() == [
        ("https://openalex.org/W70", "Models learn senses. The effect is large and lasting.")
    ]


def test_repair_texts_writes_a_part_whose_works_all_list_an_earlier_field(
    corpus_catalog_file: Path,
) -> None:
    write_raw_cell(corpus_catalog_file, "hci", 2010, [(1, SHARED, 2012)])
    catalog = read_catalog(corpus_catalog_file)
    repair_texts(catalog, HCI_SELECTION)
    assert load_base_texts(catalog, "hci_2010").is_empty()


def test_repair_texts_gives_the_same_texts_in_worker_processes(corpus_catalog_file: Path) -> None:
    catalog = read_catalog(corpus_catalog_file)
    result = repair_texts(catalog, CorpusSelection())
    alone = {part: load_base_texts(catalog, part) for part in result.parts}
    repair_texts(catalog, CorpusSelection(), process_count=2)
    assert all(load_base_texts(catalog, part).equals(alone[part]) for part in alone)


def test_repair_texts_stops_without_metadata(corpus_catalog_file: Path) -> None:
    catalog = read_catalog(corpus_catalog_file)
    shutil.rmtree(catalog.work_metadata.path / "compsci")
    with pytest.raises(ValueError, match="compsci 2010: run `interpretant fetch-metadata`"):
        repair_texts(catalog, CorpusSelection())


def test_repair_texts_command_records_the_run(corpus_catalog_file: Path) -> None:
    outcome = CliRunner().invoke(
        cli, ["repair-texts", "--catalog", str(corpus_catalog_file), "--process-count", "1"]
    )
    assert outcome.exit_code == 0, outcome.output
    record = json.loads(read_catalog(corpus_catalog_file).run_log.path.read_text().splitlines()[-1])
    assert record["command"] == "repair-texts"
    assert record["counts"]["abstracts"] > 0
