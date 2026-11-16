"""The detect-languages stage: lingua on every cached sentence, and its command."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner
from spacy.language import Language

from interpretant.__main__ import cli
from interpretant.catalog.annotations import load_annotations
from interpretant.catalog.config import read_catalog
from interpretant.catalog.languages import load_sentence_languages
from interpretant.pipelines.annotate_corpus import annotate_corpus
from interpretant.pipelines.detect_languages import DetectLanguagesSettings, detect_languages
from tests.corpus_catalog import ANNOTATE_SETTINGS, LANGUAGE_SETTINGS, SELECTION


def test_detect_languages_writes_a_row_per_cached_sentence(
    corpus_catalog_file: Path, nlp: Language
) -> None:
    catalog = read_catalog(corpus_catalog_file)
    annotate_corpus(catalog, SELECTION, ANNOTATE_SETTINGS)
    result = detect_languages(catalog, SELECTION, LANGUAGE_SETTINGS)
    sentences = [
        (sentence.start_char, sentence.end_char)
        for doc in load_annotations(catalog, "ai_2010", nlp.vocab)
        for sentence in doc.sents
    ]
    table = load_sentence_languages(catalog, "ai_2010")
    assert list(zip(table["start"], table["end"], strict=True)) == sentences
    assert set(table["language"]) <= {"en", "de", "und"}
    assert result.abstract_count == 4
    heights = sum(load_sentence_languages(catalog, part).height for part in result.parts)
    assert sum(result.sentence_counts.values()) == heights


def test_detect_languages_writes_the_same_rows_in_worker_processes(
    corpus_catalog_file: Path,
) -> None:
    catalog = read_catalog(corpus_catalog_file)
    annotate_corpus(catalog, SELECTION, ANNOTATE_SETTINGS)
    alone = detect_languages(catalog, SELECTION, LANGUAGE_SETTINGS)
    tables = {part: load_sentence_languages(catalog, part) for part in alone.parts}
    in_workers = detect_languages(catalog, SELECTION, LANGUAGE_SETTINGS, process_count=2)
    assert in_workers == alone
    assert all(load_sentence_languages(catalog, part).equals(tables[part]) for part in tables)
    assert "RAYON_NUM_THREADS" not in os.environ


def test_detect_languages_needs_the_pass_s_sentences(corpus_catalog_file: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no sentences for part"):
        detect_languages(read_catalog(corpus_catalog_file), SELECTION, LANGUAGE_SETTINGS)


def test_detect_languages_command_counts_the_sentences(corpus_catalog_file: Path) -> None:
    catalog_option = ["--catalog", str(corpus_catalog_file)]
    runner = CliRunner()
    annotated = runner.invoke(cli, ["annotate-corpus", *catalog_option, "--process-count", "1"])
    assert annotated.exit_code == 0
    outcome = runner.invoke(cli, ["detect-languages", *catalog_option, "--process-count", "1"])
    assert outcome.exit_code == 0, outcome.output
    assert "sentences of" in outcome.output
    run_log = read_catalog(corpus_catalog_file).run_log.path.read_text().splitlines()
    record = json.loads(run_log[-1])
    assert record["command"] == "detect-languages"
    assert record["counts"]["abstracts"] == 5


def test_the_repository_params_file_is_valid_for_languages() -> None:
    settings = DetectLanguagesSettings()
    assert "en" in settings.languages
    assert 0 < settings.english_threshold <= 1
