"""The count-word-repairs stage and its command, and the spaCy pass reading its decisions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from spacy.language import Language

from interpretant.__main__ import cli
from interpretant.catalog import load_annotations, load_glued_words, read_catalog
from interpretant.pipelines.annotate_corpus import annotate_corpus
from interpretant.pipelines.count_word_repairs import CountWordRepairsSettings, count_word_repairs
from interpretant.pipelines.params import CorpusSelection
from tests.corpus_catalog import ANNOTATE_SETTINGS, WORD_REPAIRS_SETTINGS, write_raw_cell

# A cell of its own: a heading run into the text three times and written apart in thirty works,
# ten times as often, and a name written only joined.
GLUED_CELL = [
    (10 + work, f"Abstract The study {work} looks at attention in models and their uses.", 2012)
    for work in range(30)
] + [
    (50 + work, f"AbstractThe study {work} looks at PubMed records of attention research.", 2012)
    for work in range(3)
]
GLUED_SELECTION = CorpusSelection(fields=("hci",), decades=(2010,))


def test_count_word_repairs_writes_each_glued_word_with_its_decision(
    corpus_catalog_file: Path,
) -> None:
    write_raw_cell(corpus_catalog_file, "hci", 2010, GLUED_CELL)
    catalog = read_catalog(corpus_catalog_file)
    result = count_word_repairs(catalog, GLUED_SELECTION, WORD_REPAIRS_SETTINGS)
    table = load_glued_words(catalog)
    decisions = dict(table.select("word", "split").rows())
    assert decisions == {"AbstractThe": True, "PubMed": False}
    assert (result.abstract_count, result.glued_word_count) == (33, 2)
    assert result.glued_split_counts == {"apart": 1}
    assert result.glued_split_occurrences == 3


def test_annotate_corpus_splits_the_glued_words_before_the_pass(
    corpus_catalog_file: Path, nlp: Language
) -> None:
    write_raw_cell(corpus_catalog_file, "hci", 2010, GLUED_CELL)
    catalog = read_catalog(corpus_catalog_file)
    count_word_repairs(catalog, GLUED_SELECTION, WORD_REPAIRS_SETTINGS)
    annotate_corpus(catalog, GLUED_SELECTION, ANNOTATE_SETTINGS)
    texts = [doc.text for doc in load_annotations(catalog, "hci_2010", nlp.vocab)]
    assert "AbstractThe study 0 looks at PubMed records of attention research." not in texts
    assert "Abstract The study 0 looks at PubMed records of attention research." in texts


def test_annotate_corpus_needs_the_glued_words(corpus_catalog_file: Path) -> None:
    catalog = read_catalog(corpus_catalog_file)
    catalog.glued_words.path.unlink()
    with pytest.raises(FileNotFoundError, match="no glued words table"):
        annotate_corpus(catalog, GLUED_SELECTION, ANNOTATE_SETTINGS)


def test_count_word_repairs_command_records_the_run(corpus_catalog_file: Path) -> None:
    write_raw_cell(corpus_catalog_file, "hci", 2010, GLUED_CELL)
    outcome = CliRunner().invoke(
        cli,
        ["count-word-repairs", "--catalog", str(corpus_catalog_file), "--process-count", "1"],
    )
    assert outcome.exit_code == 0, outcome.output
    assert "Reason" in outcome.output
    run_log = read_catalog(corpus_catalog_file).run_log.path.read_text().splitlines()
    record = json.loads(run_log[-1])
    assert record["command"] == "count-word-repairs"
    assert record["counts"]["glued_words_split"] == 1


def test_the_repository_params_file_is_valid_for_word_repairs() -> None:
    glued = CountWordRepairsSettings().glued_words
    assert glued.apart_margin > 0
    assert glued.opener_min_left_letters >= 2
    assert glued.opener_min_right_letters >= 1
