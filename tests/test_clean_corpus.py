"""The clean-corpus stage: the cleaning patterns and keeping rules over the cache."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner
from spacy.language import Language

from interpretant.__main__ import cli
from interpretant.catalog import load_annotated_abstracts, load_annotations, read_catalog
from interpretant.pipelines.annotate_corpus import annotate_corpus
from interpretant.pipelines.clean_corpus import CleanCorpusSettings, clean_corpus
from interpretant.pipelines.detect_languages import detect_languages
from interpretant.pipelines.params import CorpusSelection
from tests.corpus_catalog import (
    ANNOTATE_SETTINGS,
    CLEAN_SETTINGS,
    LANGUAGE_SETTINGS,
    SELECTION,
    SHARED,
    prepare_annotated_corpus,
    write_raw_cell,
)


def test_clean_corpus_writes_the_rows_with_their_fields_and_drops(
    corpus_catalog_file: Path,
) -> None:
    catalog = read_catalog(corpus_catalog_file)
    result = prepare_annotated_corpus(catalog)
    assert (result.abstract_count, result.parts) == (4, ["ai_2010", "compsci_2010"])
    assert result.dropped_counts == {"too_short": 1}
    parts = load_annotated_abstracts(catalog, result.parts)
    assert parts["ai_2010"]["fields"].to_list() == [["ai", "compsci"], ["ai"], ["ai"]]
    assert parts["compsci_2010"]["work_id"].to_list() == ["https://openalex.org/W4"]


def test_clean_corpus_keeps_the_cache_order_and_text(
    corpus_catalog_file: Path, nlp: Language
) -> None:
    catalog = read_catalog(corpus_catalog_file)
    result = prepare_annotated_corpus(catalog)
    rows = load_annotated_abstracts(catalog, result.parts)
    for part in result.parts:
        docs = list(load_annotations(catalog, part, nlp.vocab))
        assert [doc.text for doc in docs] == rows[part]["text"].to_list()


def test_clean_corpus_saves_an_empty_part(corpus_catalog_file: Path) -> None:
    write_raw_cell(corpus_catalog_file, "hci", 2010, [(1, SHARED, 2012)])
    catalog = read_catalog(corpus_catalog_file)
    selection = CorpusSelection(fields=("ai", "compsci", "hci"), decades=(2010,))
    result = prepare_annotated_corpus(catalog, selection)
    parts = load_annotated_abstracts(catalog, result.parts)
    assert parts["hci_2010"].height == 0
    assert parts["ai_2010"]["fields"].to_list()[0] == ["ai", "compsci", "hci"]


def test_clean_corpus_refuses_a_cache_older_than_the_raw_cells(
    corpus_catalog_file: Path,
) -> None:
    catalog = read_catalog(corpus_catalog_file)
    annotate_corpus(catalog, SELECTION, ANNOTATE_SETTINGS)
    detect_languages(catalog, SELECTION, LANGUAGE_SETTINGS)
    write_raw_cell(corpus_catalog_file, "compsci", 2010, [(4, SHARED, 2015), (6, SHARED, 2016)])
    with pytest.raises(ValueError, match="rerun annotate-corpus"):
        clean_corpus(catalog, SELECTION, CLEAN_SETTINGS)


def test_clean_corpus_gives_the_same_rows_in_worker_processes(
    corpus_catalog_file: Path,
) -> None:
    catalog = read_catalog(corpus_catalog_file)
    result = prepare_annotated_corpus(catalog)
    alone = load_annotated_abstracts(catalog, result.parts)
    clean_corpus(catalog, SELECTION, CLEAN_SETTINGS, process_count=2)
    in_workers = load_annotated_abstracts(catalog, result.parts)
    assert all(in_workers[part].equals(alone[part]) for part in result.parts)


def test_clean_corpus_command_prints_what_each_rule_took(corpus_catalog_file: Path) -> None:
    catalog_option = ["--catalog", str(corpus_catalog_file)]
    workers = ["--process-count", "1"]
    runner = CliRunner()
    assert runner.invoke(cli, ["annotate-corpus", *catalog_option, *workers]).exit_code == 0
    assert runner.invoke(cli, ["detect-languages", *catalog_option, *workers]).exit_code == 0
    outcome = runner.invoke(cli, ["clean-corpus", *catalog_option, *workers])
    assert outcome.exit_code == 0, outcome.output
    assert "5 abstracts in 3 parts" in outcome.output
    assert "too_short" in outcome.output


def test_the_repository_params_file_is_valid_for_cleaning() -> None:
    assert CleanCorpusSettings().min_english_words >= 1
