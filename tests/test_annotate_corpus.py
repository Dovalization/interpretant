"""The annotate-corpus stage (the spaCy pass) and its command, on the small test corpus."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner
from spacy.language import Language

from interpretant.__main__ import cli
from interpretant.catalog import load_annotated_abstracts, load_annotations, read_catalog
from interpretant.pipelines.annotate_corpus import AnnotateCorpusSettings, annotate_corpus
from interpretant.pipelines.params import CorpusSelection
from tests.corpus_catalog import ANNOTATE_SETTINGS, SELECTION, SHARED, write_raw_cell


def test_annotate_corpus_merges_works_and_writes_every_part(corpus_catalog_file: Path) -> None:
    catalog = read_catalog(corpus_catalog_file)
    result = annotate_corpus(catalog, SELECTION, ANNOTATE_SETTINGS)
    assert (result.abstract_count, result.parts) == (4, ["ai_2010", "compsci_2010"])
    assert result.dropped_counts == {"too_short": 1}
    parts = load_annotated_abstracts(catalog, result.parts)
    assert parts["ai_2010"]["fields"].to_list() == [["ai", "compsci"], ["ai"], ["ai"]]
    assert parts["compsci_2010"]["work_id"].to_list() == ["https://openalex.org/W4"]


def test_annotate_corpus_caches_each_part_in_the_order_of_its_rows(
    corpus_catalog_file: Path, nlp: Language
) -> None:
    catalog = read_catalog(corpus_catalog_file)
    result = annotate_corpus(catalog, SELECTION, ANNOTATE_SETTINGS)
    rows = load_annotated_abstracts(catalog, result.parts)
    for part in result.parts:
        docs = list(load_annotations(catalog, part, nlp.vocab))
        assert [doc.text for doc in docs] == rows[part]["text"].to_list()


def test_annotate_corpus_annotates_only_the_parts_asked_for(corpus_catalog_file: Path) -> None:
    catalog = read_catalog(corpus_catalog_file)
    result = annotate_corpus(catalog, SELECTION, ANNOTATE_SETTINGS, fields=("compsci",))
    assert result.parts == ["compsci_2010"]
    assert not (catalog.annotated_abstracts.path / "ai_2010.parquet").exists()


def test_annotate_corpus_saves_a_part_whose_works_all_list_an_earlier_field(
    corpus_catalog_file: Path,
) -> None:
    write_raw_cell(corpus_catalog_file, "hci", 2010, [(1, SHARED, 2012)])
    catalog = read_catalog(corpus_catalog_file)
    selection = CorpusSelection(fields=("ai", "compsci", "hci"), decades=(2010,))
    result = annotate_corpus(catalog, selection, ANNOTATE_SETTINGS)
    parts = load_annotated_abstracts(catalog, result.parts)
    assert parts["hci_2010"].height == 0
    assert parts["ai_2010"]["fields"].to_list()[0] == ["ai", "compsci", "hci"]


def test_annotate_corpus_stops_before_the_pass_without_metadata(
    corpus_catalog_file: Path,
) -> None:
    catalog = read_catalog(corpus_catalog_file)
    shutil.rmtree(catalog.work_metadata.path / "compsci")
    with pytest.raises(ValueError, match="compsci 2010: run `interpretant fetch-metadata`"):
        annotate_corpus(catalog, SELECTION, ANNOTATE_SETTINGS)
    assert not catalog.annotated_abstracts.path.exists()


def test_annotate_corpus_refuses_fields_outside_the_corpus(corpus_catalog_file: Path) -> None:
    catalog = read_catalog(corpus_catalog_file)
    with pytest.raises(ValueError, match="no part of the corpus"):
        annotate_corpus(catalog, SELECTION, ANNOTATE_SETTINGS, fields=("hci",))


def test_annotate_corpus_command_prints_what_each_rule_took(corpus_catalog_file: Path) -> None:
    arguments = ["annotate-corpus", "--catalog", str(corpus_catalog_file), "--field", "ai"]
    outcome = CliRunner().invoke(cli, arguments)
    assert outcome.exit_code == 0, outcome.output
    assert "4 abstracts in 2 parts" in outcome.output
    assert "too_short" in outcome.output


def test_the_repository_params_file_is_valid_for_the_pass() -> None:
    assert "en" in AnnotateCorpusSettings().languages
    assert "ai" in CorpusSelection().fields
