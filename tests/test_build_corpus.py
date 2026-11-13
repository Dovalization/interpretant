"""The build_corpus pipeline and its command, end to end on synthetic raw cells.

The cells live in a temporary catalog; the cleaning rules and patterns are the repository's.
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner
from spacy.language import Language

from interpretant.__main__ import cli
from interpretant.catalog import (
    CatalogConfig,
    RawCell,
    append_work_metadata,
    load_abstracts,
    load_annotations,
    read_catalog,
)
from interpretant.corpus.work_metadata import REPOSITORY_ONLY
from interpretant.pipelines.build_corpus import BuildCorpusSettings, build_corpus
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


def raw_line(work: int, text: str, year: int = 2012) -> str:
    return json.dumps(
        {
            "id": f"https://openalex.org/W{work}",
            "year": year,
            "type": "article",
            "authors": [f"https://openalex.org/A{work}"],
            "abstract": text,
        }
    )


@pytest.fixture
def catalog_file(tmp_path: Path) -> Path:
    """A catalog over temporary folders, with the repository's cleaning rules and patterns."""
    repository = read_catalog()
    catalog_file = tmp_path / "catalog.yaml"
    catalog_file.write_text(
        "raw_abstracts:\n  path: raw\n  format: jsonl.gz\n"
        "work_metadata:\n  path: metadata\n  format: jsonl.gz\n"
        "abstracts:\n  path: abstracts\n  format: parquet\n"
        f"cleaning_rules:\n  path: '{repository.cleaning_rules.path}'\n  format: yaml\n"
        f"cleaning_patterns:\n  path: '{repository.cleaning_patterns.path}'\n  format: jsonl\n"
        "annotations:\n  path: annotations\n  format: docbin\n"
    )
    cells = {
        ("ai", 2010): [
            raw_line(1, SHARED),
            raw_line(2, "Short note here."),
            raw_line(3, PARAGRAPH),
        ],
        ("compsci", 2010): [raw_line(1, SHARED), raw_line(4, REVISED, year=2015)],
        ("ai", 2020): [raw_line(5, SHARED, year=2021)],
    }
    for (field, decade), lines in cells.items():
        path = tmp_path / "raw" / field / f"{decade}.jsonl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as cell:
            cell.writelines(f"{line}\n" for line in lines)
        # Work 3 is a repository-only record; the others sit in indexed journals.
        records = [
            metadata_record(work, source_type="repository", indexed_in=())
            if work == 3
            else metadata_record(work)
            for work in (int(json.loads(line)["id"].rsplit("W", 1)[1]) for line in lines)
        ]
        append_work_metadata(read_catalog(catalog_file), RawCell(field, decade), records)
    return catalog_file


def settings(**changes: object) -> BuildCorpusSettings:
    return BuildCorpusSettings.model_validate(
        BuildCorpusSettings().model_dump()
        | {"fields": ("ai", "compsci"), "decades": (2010,), "languages": ("en", "de")}
        | changes
    )


def test_build_corpus_annotates_merges_and_applies_the_corpus_rules(catalog_file: Path) -> None:
    catalog = read_catalog(catalog_file)
    result = build_corpus(catalog, settings())
    assert (result.cell_count, result.abstract_count) == (2, 4)
    assert result.parts == ["ai_2010", "compsci_2010"]
    assert result.dropped_counts == {"too_short": 1, "near_duplicate": 1}
    table = load_abstracts(catalog).sort("work_id").collect()
    assert table["fields"].to_list() == [["ai", "compsci"], ["ai"], ["ai"], ["compsci"]]
    assert table["duplicate_of"].to_list() == [None, None, None, "https://openalex.org/W3"]
    assert table["flags"].to_list() == [[], [], [REPOSITORY_ONLY], []]


def test_build_corpus_refuses_cells_whose_metadata_was_not_fetched(catalog_file: Path) -> None:
    catalog = read_catalog(catalog_file)
    shutil.rmtree(catalog.work_metadata.path / "compsci")
    with pytest.raises(ValueError, match="compsci 2010: run `interpretant fetch-metadata`"):
        build_corpus(catalog, settings())


def test_build_corpus_caches_each_part_in_the_order_of_its_rows(
    catalog_file: Path, nlp: Language
) -> None:
    catalog: CatalogConfig = read_catalog(catalog_file)
    build_corpus(catalog, settings())
    rows = load_abstracts(catalog).collect()
    for part, field in [("ai_2010", "ai"), ("compsci_2010", "compsci")]:
        docs = list(load_annotations(catalog, part, nlp.vocab))
        part_texts = rows.filter(rows["fields"].list.first() == field)["text"].to_list()
        assert [doc.text for doc in docs] == part_texts


def test_build_corpus_refuses_a_selection_without_raw_cells(catalog_file: Path) -> None:
    with pytest.raises(ValueError, match="no raw cell"):
        build_corpus(read_catalog(catalog_file), settings(decades=(1950,)))


def test_build_corpus_command_prints_what_each_rule_took(catalog_file: Path) -> None:
    arguments = [
        "build-corpus",
        "--catalog",
        str(catalog_file),
        "--field",
        "ai",
        "--decade",
        "2010",
    ]
    outcome = CliRunner().invoke(cli, arguments)
    assert outcome.exit_code == 0, outcome.output
    assert "3 abstracts from 1 cells" in outcome.output
    assert "too_short" in outcome.output


def test_the_repository_params_file_is_valid() -> None:
    assert "ai" in BuildCorpusSettings().fields
