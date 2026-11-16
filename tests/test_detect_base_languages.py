"""Language before the word repairs: the rule-based sentence split and the base-language stage."""

from __future__ import annotations

import json
import os
from pathlib import Path

from click.testing import CliRunner

from interpretant.__main__ import cli
from interpretant.catalog import load_base_sentence_languages, load_base_texts, read_catalog
from interpretant.corpus.annotation import build_sentence_splitter
from interpretant.pipelines.detect_base_languages import detect_base_languages
from interpretant.pipelines.repair_texts import repair_texts
from tests.corpus_catalog import LANGUAGE_SETTINGS, SELECTION


def test_build_sentence_splitter_keeps_abbreviations_and_brackets_inside_sentences() -> None:
    splitter = build_sentence_splitter()
    doc = splitter("See Vol. 15, pp. 79-88 for details [cf. Fig. 2. Left]. The effect is large.")
    assert [sentence.text for sentence in doc.sents] == [
        "See Vol. 15, pp. 79-88 for details [cf. Fig. 2. Left].",
        "The effect is large.",
    ]


def test_detect_base_languages_writes_a_row_per_sentence_of_the_base_texts(
    corpus_catalog_file: Path,
) -> None:
    catalog = read_catalog(corpus_catalog_file)
    repair_texts(catalog, SELECTION)
    result = detect_base_languages(catalog, SELECTION, LANGUAGE_SETTINGS)
    splitter = build_sentence_splitter()
    texts = load_base_texts(catalog, "ai_2010")
    sentences = [
        (work_id, sentence.start_char, sentence.end_char)
        for work_id, text in texts.iter_rows()
        for sentence in splitter(text).sents
    ]
    table = load_base_sentence_languages(catalog, "ai_2010")
    assert list(table.select("work_id", "start", "end").iter_rows()) == sentences
    assert set(table["language"]) <= {"en", "de", "und"}
    assert result.abstract_count == 4


def test_detect_base_languages_writes_the_same_rows_in_worker_processes(
    corpus_catalog_file: Path,
) -> None:
    catalog = read_catalog(corpus_catalog_file)
    repair_texts(catalog, SELECTION)
    alone = detect_base_languages(catalog, SELECTION, LANGUAGE_SETTINGS)
    tables = {part: load_base_sentence_languages(catalog, part) for part in alone.parts}
    in_workers = detect_base_languages(catalog, SELECTION, LANGUAGE_SETTINGS, process_count=2)
    assert in_workers == alone
    assert all(load_base_sentence_languages(catalog, part).equals(tables[part]) for part in tables)
    assert "RAYON_NUM_THREADS" not in os.environ


def test_detect_base_languages_command_records_the_run(corpus_catalog_file: Path) -> None:
    options = ["--catalog", str(corpus_catalog_file), "--process-count", "1"]
    runner = CliRunner()
    assert runner.invoke(cli, ["repair-texts", *options]).exit_code == 0
    outcome = runner.invoke(cli, ["detect-base-languages", *options])
    assert outcome.exit_code == 0, outcome.output
    record = json.loads(read_catalog(corpus_catalog_file).run_log.path.read_text().splitlines()[-1])
    assert record["command"] == "detect-base-languages"
    assert record["counts"]["abstracts"] == 5
