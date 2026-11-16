"""The count-word-repairs stage and its command, and the spaCy pass reading its decisions."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from click.testing import CliRunner
from spacy.language import Language

from interpretant.__main__ import cli
from interpretant.catalog import (
    load_annotations,
    load_glued_words,
    load_lower_glued_words,
    load_repaired_texts,
    load_split_words,
    read_catalog,
)
from interpretant.pipelines.annotate_corpus import annotate_corpus
from interpretant.pipelines.count_word_repairs import CountWordRepairsSettings
from interpretant.pipelines.params import CorpusSelection
from tests.corpus_catalog import ANNOTATE_SETTINGS, repair_and_count, write_raw_cell

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
# A cell of its own: a word written whole in thirty works and broken at a line end in two,
# fifteen times as often.
SPLIT_CELL = [
    (10 + work, f"The information study {work} looks at models and their uses.", 2012)
    for work in range(30)
] + [
    (50 + work, f"The infor- mation study {work} looks at records of research.", 2012)
    for work in range(2)
]


def test_count_word_repairs_writes_each_glued_word_with_its_decision(
    corpus_catalog_file: Path,
) -> None:
    write_raw_cell(corpus_catalog_file, "hci", 2010, GLUED_CELL)
    catalog = read_catalog(corpus_catalog_file)
    result = repair_and_count(catalog, GLUED_SELECTION)
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
    repair_and_count(catalog, GLUED_SELECTION)
    annotate_corpus(catalog, GLUED_SELECTION, ANNOTATE_SETTINGS)
    texts = [doc.text for doc in load_annotations(catalog, "hci_2010", nlp.vocab)]
    assert "AbstractThe study 0 looks at PubMed records of attention research." not in texts
    assert "Abstract The study 0 looks at PubMed records of attention research." in texts


# A cell of its own: "of the" written apart in thirty works and glued once.
LOWER_GLUED_CELL = [
    (10 + work, f"The study {work} looks at the role of the models and their uses.", 2012)
    for work in range(30)
] + [(50, "The study looks at the role ofthe models in research.", 2012)]


def test_count_word_repairs_splits_lower_case_glued_words_and_writes_their_decisions(
    corpus_catalog_file: Path,
) -> None:
    write_raw_cell(corpus_catalog_file, "hci", 2010, LOWER_GLUED_CELL)
    catalog = read_catalog(corpus_catalog_file)
    result = repair_and_count(catalog, GLUED_SELECTION)
    row = load_lower_glued_words(catalog).filter(pl.col("word") == "ofthe").row(0, named=True)
    assert (row["parts"], row["joined"], row["split"]) == (["of", "the"], 1, True)
    assert result.lower_glued_split_occurrences == {"two_words": 1}
    texts = load_repaired_texts(catalog, "hci_2010")
    assert "the role of the models in research" in texts["text"].str.join(" ").item()


def test_count_word_repairs_writes_each_broken_word_with_its_decision(
    corpus_catalog_file: Path,
) -> None:
    write_raw_cell(corpus_catalog_file, "hci", 2010, SPLIT_CELL)
    catalog = read_catalog(corpus_catalog_file)
    result = repair_and_count(catalog, GLUED_SELECTION)
    table = load_split_words(catalog).filter(pl.col("word") == "information")
    row = table.row(0, named=True)
    assert (row["left"], row["right"], row["joined"], row["hyphen_space"]) == (
        "infor",
        "mation",
        30,
        2,
    )
    assert row["join"]
    assert result.split_join_counts["spaced"] >= 1
    assert result.split_joined_count <= result.split_word_count
    assert result.split_join_occurrences >= 2


def test_annotate_corpus_joins_the_broken_words_before_the_pass(
    corpus_catalog_file: Path, nlp: Language
) -> None:
    write_raw_cell(corpus_catalog_file, "hci", 2010, SPLIT_CELL)
    catalog = read_catalog(corpus_catalog_file)
    repair_and_count(catalog, GLUED_SELECTION)
    annotate_corpus(catalog, GLUED_SELECTION, ANNOTATE_SETTINGS)
    texts = [doc.text for doc in load_annotations(catalog, "hci_2010", nlp.vocab)]
    assert "The infor- mation study 0 looks at records of research." not in texts
    assert "The information study 0 looks at records of research." in texts


def test_count_word_repairs_writes_every_part_s_texts_with_the_repairs_made(
    corpus_catalog_file: Path,
) -> None:
    write_raw_cell(corpus_catalog_file, "hci", 2010, SPLIT_CELL)
    catalog = read_catalog(corpus_catalog_file)
    selection = CorpusSelection(fields=("ai", "hci"), decades=(2010,))
    repair_and_count(catalog, selection)
    hci = load_repaired_texts(catalog, "hci_2010")
    assert hci["work_id"].to_list()[:2] == ["https://openalex.org/W10", "https://openalex.org/W11"]
    assert "The information study 0 looks at records of research." in hci["text"].to_list()
    assert load_repaired_texts(catalog, "ai_2010").height == 3


def test_count_word_repairs_command_records_the_run(corpus_catalog_file: Path) -> None:
    write_raw_cell(corpus_catalog_file, "hci", 2010, GLUED_CELL)
    outcome = CliRunner().invoke(
        cli,
        ["repair-texts", "--catalog", str(corpus_catalog_file), "--process-count", "1"],
    )
    assert outcome.exit_code == 0, outcome.output
    outcome = CliRunner().invoke(
        cli,
        ["detect-base-languages", "--catalog", str(corpus_catalog_file), "--process-count", "1"],
    )
    assert outcome.exit_code == 0, outcome.output
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
    settings = CountWordRepairsSettings()
    assert settings.glued_words.apart_margin > 0
    assert settings.glued_words.opener_min_left_letters >= 2
    assert settings.glued_words.opener_min_right_letters >= 1
    assert settings.split_words.hyphen_margin >= settings.split_words.join_margin
