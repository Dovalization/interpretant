"""The build-corpus stage (the corpus-level rules and review flags) and its command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from interpretant.__main__ import cli
from interpretant.catalog import load_abstracts, read_catalog
from interpretant.corpus.work_metadata import REPOSITORY_ONLY
from interpretant.pipelines.build_corpus import BuildCorpusSettings, build_corpus
from interpretant.pipelines.params import CorpusSelection
from tests.corpus_catalog import SELECTION, prepare_annotated_corpus


def test_build_corpus_applies_the_corpus_rules_and_the_review_flags(
    corpus_catalog_file: Path,
) -> None:
    catalog = read_catalog(corpus_catalog_file)
    prepare_annotated_corpus(catalog)
    result = build_corpus(catalog, SELECTION, BuildCorpusSettings())
    assert result.parts == ["ai_2010", "compsci_2010"]
    assert result.dropped_counts == {"too_short": 1, "near_duplicate": 1}
    assert result.flag_counts == {REPOSITORY_ONLY: 1}
    table = load_abstracts(catalog).sort("work_id").collect()
    assert table["duplicate_of"].to_list() == [None, None, None, "https://openalex.org/W3"]
    assert table["flags"].to_list() == [[], [], [REPOSITORY_ONLY], []]


def test_build_corpus_gives_the_same_table_with_signatures_made_in_workers(
    corpus_catalog_file: Path,
) -> None:
    catalog = read_catalog(corpus_catalog_file)
    prepare_annotated_corpus(catalog)
    build_corpus(catalog, SELECTION, BuildCorpusSettings())
    alone = load_abstracts(catalog).collect()
    build_corpus(catalog, SELECTION, BuildCorpusSettings(), process_count=2)
    assert load_abstracts(catalog).collect().equals(alone)


def test_build_corpus_names_the_parts_not_yet_annotated(corpus_catalog_file: Path) -> None:
    catalog = read_catalog(corpus_catalog_file)
    prepare_annotated_corpus(catalog, CorpusSelection(fields=("ai",), decades=(2010,)))
    with pytest.raises(FileNotFoundError, match="compsci_2010"):
        build_corpus(catalog, SELECTION, BuildCorpusSettings())


def test_build_corpus_command_runs_after_the_earlier_stages(corpus_catalog_file: Path) -> None:
    catalog_option = ["--catalog", str(corpus_catalog_file)]
    runner = CliRunner()
    for command, options in (
        ("build-linked-tables", []),
        ("repair-texts", ["--process-count", "1"]),
        ("detect-languages", ["--process-count", "1"]),
        ("count-word-repairs", ["--process-count", "1"]),
        ("annotate-corpus", ["--process-count", "1"]),
        ("clean-corpus", ["--process-count", "1"]),
    ):
        earlier = runner.invoke(cli, [command, *catalog_option, *options])
        assert earlier.exit_code == 0, earlier.output
    built = runner.invoke(cli, ["build-corpus", *catalog_option, "--process-count", "1"])
    assert built.exit_code == 0, built.output
    assert "5 abstracts in 3 parts" in built.output
    assert "near_duplicate" in built.output
    run_log = read_catalog(corpus_catalog_file).run_log.path.read_text().splitlines()
    runs = [json.loads(line) for line in run_log]
    assert [run["command"] for run in runs] == [
        "build-linked-tables",
        "repair-texts",
        "detect-languages",
        "count-word-repairs",
        "annotate-corpus",
        "clean-corpus",
        "build-corpus",
    ]
    assert runs[-1]["counts"] == {"abstracts": 5, "dropped": 3, "parts": 3}
    assert "near-duplicates" in [stage["name"] for stage in runs[-1]["stages"]]
    assert set(runs[-1]["peak_memory_mb"]) == {"main", "largest_worker"}


def test_the_repository_params_file_is_valid_for_the_corpus_rules() -> None:
    assert BuildCorpusSettings().template_min_abstracts >= 2
