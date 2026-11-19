"""The build-usages stage on the synthetic corpus: every mention found, checked, and one drawn."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import spacy
from click.testing import CliRunner

from interpretant.__main__ import cli
from interpretant.catalog.abstracts import load_abstracts, load_abstracts_part
from interpretant.catalog.annotation_cache import (
    ANNOTATION_CACHE_COLUMNS,
    load_annotation_cache,
    save_annotation_cache,
)
from interpretant.catalog.config import CatalogConfig, read_catalog
from interpretant.catalog.lemma_exceptions import LemmaException
from interpretant.catalog.parts import part_path
from interpretant.catalog.usages import USAGES_COLUMNS, load_usages
from interpretant.pipelines import build_usages as build_usages_module
from interpretant.pipelines.build_corpus import BuildCorpusSettings, build_corpus
from interpretant.pipelines.build_sentence_view import build_sentence_view
from interpretant.pipelines.build_static_view import build_static_view
from interpretant.pipelines.build_usages import (
    TextMismatchError,
    UsagesSettings,
    build_usages,
)
from interpretant.usages.checks import validate_usages_against_text
from interpretant.usages.target_matching import TargetMention
from interpretant.usages.usage_selection import NAME_LABELS
from tests.corpus_catalog import SELECTION, prepare_annotated_corpus

SETTINGS = UsagesSettings(seed=7, model="en_core_web_sm", name_labels=tuple(sorted(NAME_LABELS)))


def prepare_views(catalog: CatalogConfig) -> None:
    """Run every stage before the usages, as ``dvc repro`` does."""
    prepare_annotated_corpus(catalog)
    build_corpus(catalog, SELECTION, BuildCorpusSettings())
    build_sentence_view(catalog)
    build_static_view(catalog, SELECTION)


@pytest.fixture
def prepared_catalog(corpus_catalog_file: Path) -> CatalogConfig:
    catalog = read_catalog(corpus_catalog_file)
    prepare_views(catalog)
    return catalog


@pytest.fixture
def stand_in_transformer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The small model's full pipeline in place of the transformer, which needs a GPU."""
    nlp = spacy.load("en_core_web_sm")
    monkeypatch.setattr(build_usages_module, "_transformer", lambda _process_count: nlp)


def test_build_usages_finds_the_targets_and_checks_against_the_text(
    prepared_catalog: CatalogConfig,
) -> None:
    result = build_usages(prepared_catalog, SELECTION, SETTINGS)
    usages = load_usages(prepared_catalog).collect()
    texts = dict(load_abstracts(prepared_catalog).select("work_id", "text").collect().iter_rows())
    assert result.parts == ["ai_2010", "compsci_2010"]
    assert {"attention", "model"} <= set(usages["word"])
    validate_usages_against_text(usages, texts)
    left_out = sum(
        result.counts.get(key, 0)
        for key in ("inside_removed_ranges", "outside_sentences", "in_removed_sentences")
    )
    assert result.counts["mentions"] == result.counts["usages"] + left_out
    assert result.counts["usages"] == usages.height


def test_build_usages_draws_one_eligible_usage_per_work_and_word(
    prepared_catalog: CatalogConfig,
) -> None:
    build_usages(prepared_catalog, SELECTION, SETTINGS)
    usages = load_usages(prepared_catalog).collect()
    eligible = usages.filter(pl.col("in_scope") & pl.col("sentence_exclusion").is_null())
    drawn = usages.filter("drawn")
    assert drawn.height == eligible.select("work_id", "word").n_unique()
    smallest = eligible.group_by("work_id", "word").agg(pl.col("draw_key").min())
    assert sorted(drawn["draw_key"]) == sorted(smallest["draw_key"])


def test_build_usages_gives_the_same_table_in_workers_and_a_new_draw_for_a_new_seed(
    prepared_catalog: CatalogConfig,
) -> None:
    build_usages(prepared_catalog, SELECTION, SETTINGS)
    alone = load_usages(prepared_catalog).collect().sort("work_id", "word", "content_key")
    build_usages(prepared_catalog, SELECTION, SETTINGS, process_count=2)
    in_workers = load_usages(prepared_catalog).collect().sort("work_id", "word", "content_key")
    assert alone.equals(in_workers)
    build_usages(prepared_catalog, SELECTION, SETTINGS.model_copy(update={"seed": 8}))
    reseeded = load_usages(prepared_catalog).collect().sort("work_id", "word", "content_key")
    assert reseeded["content_key"].to_list() == alone["content_key"].to_list()
    assert reseeded["draw_key"].to_list() != alone["draw_key"].to_list()


def test_build_usages_refuses_a_cached_document_whose_text_changed(
    prepared_catalog: CatalogConfig,
) -> None:
    part = load_abstracts_part(prepared_catalog, "ai_2010")
    changed = part.with_columns(pl.col("text").str.replace("attention", "intention"))
    changed.write_parquet(part_path(prepared_catalog.abstracts.path, "ai_2010", ".parquet"))
    with pytest.raises(TextMismatchError):
        build_usages(prepared_catalog, SELECTION, SETTINGS)


@pytest.mark.usefixtures("stand_in_transformer")
def test_build_usages_command_takes_one_field_and_prints_the_counts(
    prepared_catalog: CatalogConfig, corpus_catalog_file: Path
) -> None:
    outcome = CliRunner().invoke(
        cli,
        [
            "build-usages",
            "--catalog",
            str(corpus_catalog_file),
            "--field",
            "ai",
            "--decade",
            "2010",
            "--process-count",
            "1",
            "--gpu-process-count",
            "1",
        ],
    )
    assert outcome.exit_code == 0, outcome.output
    assert "The usages of 1 parts" in outcome.output
    assert "Drawn" in outcome.output
    assert load_usages(prepared_catalog).collect()["work_id"].n_unique() >= 1


@pytest.mark.usefixtures("stand_in_transformer")
def test_build_usages_with_the_transformer_annotates_every_usage_of_the_same_rows(
    prepared_catalog: CatalogConfig,
) -> None:
    build_usages(prepared_catalog, SELECTION, SETTINGS)
    small = load_usages(prepared_catalog).collect().sort("work_id", "word", "content_key")
    transformer = SETTINGS.model_copy(update={"model": "en_core_web_trf"})
    result = build_usages(prepared_catalog, SELECTION, transformer, gpu_process_count=1)
    table = load_usages(prepared_catalog).collect().sort("work_id", "word", "content_key")
    keys = ["work_id", "word", "content_key", "draw_key", "target_start", "target_end"]
    assert table.select(keys).equals(small.select(keys))
    assert set(table["model"]) == {"en_core_web_trf"}
    assert table["in_scope"].null_count() == 0
    assert (
        result.counts["staged_sentences"] == table.select("work_id", "sentence_number").n_unique()
    )
    assert list(prepared_catalog.usages.path.parent.glob("staging_*")) == []


@pytest.mark.usefixtures("stand_in_transformer")
def test_build_usages_resumes_from_the_parts_the_transformer_already_wrote(
    prepared_catalog: CatalogConfig,
) -> None:
    transformer = SETTINGS.model_copy(update={"model": "en_core_web_trf"})
    first = build_usages(prepared_catalog, SELECTION, transformer, gpu_process_count=1)
    again = build_usages(prepared_catalog, SELECTION, transformer, gpu_process_count=1, resume=True)
    assert again.counts["skipped_parts"] == len(first.parts)
    assert "staged_sentences" not in again.counts


def test_batches_keep_within_the_document_and_token_caps() -> None:
    lengths = [10] * 100 + [900, 1500, 3000, 5000]
    batches = build_usages_module._batches(lengths)
    assert [len(batch) for batch in batches] == [64, 37, 1, 1, 1]
    assert sum(len(batch) for batch in batches) == len(lengths)
    for batch in batches:
        tokens = sum(lengths[i] for i in batch)
        assert len(batch) <= 64
        assert tokens <= 2000 or len(batch) == 1


def test_staging_reads_a_window_around_the_target_of_a_long_sentence() -> None:
    nlp = spacy.blank("en")
    doc = nlp.make_doc(" ".join(["word"] * 899 + ["attention"] + ["word"] * 300))
    sentence = build_usages_module._Sentence(0, 0, len(doc.text), 1200, "en", 0.99, None)
    work = build_usages_module._Work("https://openalex.org/W1", doc, doc.text, [], [sentence], {})
    target = doc[899]
    mention = TargetMention("attention", 899, target.idx, target.idx + len(target.text), None, None)
    row: list[object] = [None] * len(USAGES_COLUMNS)
    row[build_usages_module._WINDOWED] = False
    sink = build_usages_module._StagedUsages()
    sink.add(work, sentence, mention, row, None)  # type: ignore[arg-type]
    [(key, (words, _))] = sink.sentences.items()
    assert key == ("https://openalex.org/W1", 0, 899 - 256)
    assert len(words) == 513 and words[256] == "attention"
    assert sink.rows[0][-4:-1][0] == 256
    assert row[build_usages_module._WINDOWED] is True


@pytest.mark.usefixtures("stand_in_transformer")
def test_build_usages_reuses_the_annotation_cache_for_the_same_table(
    prepared_catalog: CatalogConfig,
) -> None:
    transformer = SETTINGS.model_copy(update={"model": "en_core_web_trf"})
    cold = build_usages(prepared_catalog, SELECTION, transformer, gpu_process_count=1)
    first = load_usages(prepared_catalog).collect().sort("work_id", "word", "content_key")
    warm = build_usages(prepared_catalog, SELECTION, transformer, gpu_process_count=1)
    second = load_usages(prepared_catalog).collect().sort("work_id", "word", "content_key")
    assert cold.counts["cache_misses"] == cold.counts["staged_sentences"]
    assert cold.counts.get("cache_hits", 0) == 0
    assert warm.counts["cache_hits"] == warm.counts["staged_sentences"]
    assert warm.counts.get("cache_misses", 0) == 0
    assert second.equals(first)


def test_the_annotation_key_changes_with_the_lemma_exceptions() -> None:
    exception = LemmaException(form="data", lemma="data", tag="NNS", source="author")
    words, spaces = ["The", "data"], [True, False]
    plain = build_usages_module._annotation_prefix(())
    with_exception = build_usages_module._annotation_prefix((exception,))
    assert build_usages_module._annotation_key(plain, words, spaces) != (
        build_usages_module._annotation_key(with_exception, words, spaces)
    )
    assert build_usages_module._annotation_key(plain, words, spaces) != (
        build_usages_module._annotation_key(plain, words, [False, False])
    )


def test_the_annotation_cache_refuses_a_repeated_key(prepared_catalog: CatalogConfig) -> None:
    assert load_annotation_cache(prepared_catalog, "ai_2010").height == 0
    row = ("k1", ["NN"], ["NOUN"], [""], ["data"], ["O"])
    cache = pl.DataFrame([row, row], schema=ANNOTATION_CACHE_COLUMNS, orient="row")
    with pytest.raises(ValueError, match="repeats a key"):
        save_annotation_cache(prepared_catalog, cache, "ai_2010")


def test_split_by_cache_groups_repeated_sentences_under_their_first_appearance() -> None:
    hits, sharing = build_usages_module._split_by_cache(["a", "b", "c", "b", "d", "b"], {"c": ()})
    assert hits == [2]
    assert sharing == {"a": [0], "b": [1, 3, 5], "d": [4]}
    assert [positions[0] for positions in sharing.values()] == [0, 1, 4]
