"""The annotate-corpus stage (the spaCy pass) and its command, on the small test corpus."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from click.testing import CliRunner
from spacy.language import Language

from interpretant.__main__ import cli
from interpretant.catalog.annotations import WORK_ID, load_annotations
from interpretant.catalog.config import read_catalog
from interpretant.catalog.sentences import load_sentences
from interpretant.cli.options import default_process_count
from interpretant.corpus.removals import SPANS_KEY
from interpretant.pipelines.annotate_corpus import AnnotateCorpusSettings, annotate_corpus
from interpretant.pipelines.count_word_repairs import count_word_repairs
from interpretant.pipelines.params import CorpusSelection
from tests.corpus_catalog import (
    ANNOTATE_SETTINGS,
    SELECTION,
    SHARED,
    WORD_REPAIRS_SETTINGS,
    write_raw_cell,
)


def cached_work_ids(catalog_file: Path, part: str, nlp: Language) -> list[str]:
    catalog = read_catalog(catalog_file)
    return [doc.user_data[WORK_ID] for doc in load_annotations(catalog, part, nlp.vocab)]


def test_annotate_corpus_merges_works_and_caches_every_part(
    corpus_catalog_file: Path, nlp: Language
) -> None:
    result = annotate_corpus(read_catalog(corpus_catalog_file), SELECTION, ANNOTATE_SETTINGS)
    assert (result.abstract_count, result.parts) == (4, ["ai_2010", "compsci_2010"])
    assert cached_work_ids(corpus_catalog_file, "ai_2010", nlp) == [
        "https://openalex.org/W1",
        "https://openalex.org/W2",
        "https://openalex.org/W3",
    ]
    assert cached_work_ids(corpus_catalog_file, "compsci_2010", nlp) == ["https://openalex.org/W4"]


def test_annotate_corpus_writes_the_same_cache_in_worker_processes(
    corpus_catalog_file: Path,
) -> None:
    catalog = read_catalog(corpus_catalog_file)

    def cache_files() -> dict[str, bytes]:
        files = sorted(catalog.annotations.path.rglob("*.spacy"))
        return {path.name: path.read_bytes() for path in files}

    annotate_corpus(catalog, SELECTION, ANNOTATE_SETTINGS)
    alone = cache_files()
    annotate_corpus(catalog, SELECTION, ANNOTATE_SETTINGS, process_count=2)
    assert list(alone) == ["ai_2010.spacy", "compsci_2010.spacy"]
    assert cache_files() == alone


def test_annotate_corpus_caches_sentences_and_lemmas_without_cleaning(
    corpus_catalog_file: Path, nlp: Language
) -> None:
    catalog = read_catalog(corpus_catalog_file)
    annotate_corpus(catalog, SELECTION, ANNOTATE_SETTINGS)
    (doc,) = load_annotations(catalog, "compsci_2010", nlp.vocab)
    assert len(list(doc.sents)) > 1
    assert doc.has_annotation("LEMMA")
    assert SPANS_KEY not in doc.spans


def test_annotate_corpus_writes_each_cached_document_s_sentences(
    corpus_catalog_file: Path, nlp: Language
) -> None:
    catalog = read_catalog(corpus_catalog_file)
    result = annotate_corpus(catalog, SELECTION, ANNOTATE_SETTINGS)
    for part in result.parts:
        cached = [
            (doc.user_data[WORK_ID], sentence.start_char, sentence.end_char)
            for doc in load_annotations(catalog, part, nlp.vocab)
            for sentence in doc.sents
        ]
        assert list(load_sentences(catalog, part).iter_rows()) == cached


def test_annotate_corpus_annotates_only_the_parts_asked_for(corpus_catalog_file: Path) -> None:
    catalog = read_catalog(corpus_catalog_file)
    result = annotate_corpus(catalog, SELECTION, ANNOTATE_SETTINGS, fields=("compsci",))
    assert result.parts == ["compsci_2010"]
    assert not (catalog.annotations.path / "ai_2010.spacy").exists()


def test_annotate_corpus_saves_a_part_whose_works_all_list_an_earlier_field(
    corpus_catalog_file: Path, nlp: Language
) -> None:
    write_raw_cell(corpus_catalog_file, "hci", 2010, [(1, SHARED, 2012)])
    selection = CorpusSelection(fields=("ai", "compsci", "hci"), decades=(2010,))
    catalog = read_catalog(corpus_catalog_file)
    count_word_repairs(catalog, selection, WORD_REPAIRS_SETTINGS)
    annotate_corpus(catalog, selection, ANNOTATE_SETTINGS)
    assert cached_work_ids(corpus_catalog_file, "hci_2010", nlp) == []


def test_annotate_corpus_needs_each_part_s_repaired_texts(corpus_catalog_file: Path) -> None:
    catalog = read_catalog(corpus_catalog_file)
    (catalog.repaired_texts.path / "compsci_2010.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="no repaired texts for part 'compsci_2010'"):
        annotate_corpus(catalog, SELECTION, ANNOTATE_SETTINGS)
    assert not catalog.annotations.path.exists()


def test_annotate_corpus_refuses_fields_outside_the_corpus(corpus_catalog_file: Path) -> None:
    catalog = read_catalog(corpus_catalog_file)
    with pytest.raises(ValueError, match="no part of the corpus"):
        annotate_corpus(catalog, SELECTION, ANNOTATE_SETTINGS, fields=("hci",))


def test_annotate_corpus_command_counts_the_abstracts(corpus_catalog_file: Path) -> None:
    arguments = ["annotate-corpus", "--catalog", str(corpus_catalog_file), "--field", "ai"]
    outcome = CliRunner().invoke(cli, [*arguments, "--process-count", "1"])
    assert outcome.exit_code == 0, outcome.output
    assert "4 abstracts in 2 parts" in outcome.output


def test_the_default_process_count_leaves_two_cpus_free() -> None:
    assert default_process_count() == max(1, (os.cpu_count() or 2) - 2)


def test_the_repository_params_file_is_valid_for_the_pass() -> None:
    assert AnnotateCorpusSettings().named_entity_recognition is False
    assert "ai" in CorpusSelection().fields
