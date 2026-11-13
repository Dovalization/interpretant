"""The build-corpus stage (the corpus-level rules and review flags) and its command."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from interpretant.__main__ import cli
from interpretant.catalog import load_abstracts, read_catalog
from interpretant.corpus.work_metadata import REPOSITORY_ONLY
from interpretant.pipelines.annotate_corpus import annotate_corpus
from interpretant.pipelines.build_corpus import BuildCorpusSettings, build_corpus
from tests.corpus_catalog import ANNOTATE_SETTINGS, SELECTION


def test_build_corpus_applies_the_corpus_rules_and_the_review_flags(
    corpus_catalog_file: Path,
) -> None:
    catalog = read_catalog(corpus_catalog_file)
    annotate_corpus(catalog, SELECTION, ANNOTATE_SETTINGS)
    result = build_corpus(catalog, SELECTION, BuildCorpusSettings())
    assert result.parts == ["ai_2010", "compsci_2010"]
    assert result.dropped_counts == {"too_short": 1, "near_duplicate": 1}
    assert result.flag_counts == {REPOSITORY_ONLY: 1}
    table = load_abstracts(catalog).sort("work_id").collect()
    assert table["duplicate_of"].to_list() == [None, None, None, "https://openalex.org/W3"]
    assert table["flags"].to_list() == [[], [], [REPOSITORY_ONLY], []]


def test_build_corpus_names_the_parts_not_yet_annotated(corpus_catalog_file: Path) -> None:
    catalog = read_catalog(corpus_catalog_file)
    annotate_corpus(catalog, SELECTION, ANNOTATE_SETTINGS, fields=("ai",))
    with pytest.raises(FileNotFoundError, match="compsci_2010"):
        build_corpus(catalog, SELECTION, BuildCorpusSettings())


def test_build_corpus_command_runs_after_the_pass(corpus_catalog_file: Path) -> None:
    catalog_option = ["--catalog", str(corpus_catalog_file)]
    runner = CliRunner()
    annotated = runner.invoke(cli, ["annotate-corpus", *catalog_option])
    assert annotated.exit_code == 0, annotated.output
    built = runner.invoke(cli, ["build-corpus", *catalog_option])
    assert built.exit_code == 0, built.output
    assert "5 abstracts in 3 parts" in built.output
    assert "near_duplicate" in built.output


def test_the_repository_params_file_is_valid_for_the_corpus_rules() -> None:
    assert BuildCorpusSettings().template_min_abstracts >= 2
