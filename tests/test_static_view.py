"""The thin static view: one document's static tokens, the stage that writes them, its command."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
import spacy
from click.testing import CliRunner
from spacy.language import Language
from spacy.tokens import Doc

from interpretant.__main__ import cli
from interpretant.catalog.annotations import load_annotations
from interpretant.catalog.compounds import load_compounds
from interpretant.catalog.config import CatalogConfig, read_catalog
from interpretant.catalog.lemma_exceptions import load_lemma_exceptions
from interpretant.catalog.spellings import load_spellings
from interpretant.catalog.static_view import load_static_view, save_static_view
from interpretant.corpus.annotation import annotate_texts
from interpretant.corpus.compound_spelling import NO_COMPOUNDS, build_compound_map
from interpretant.corpus.lemma_exceptions import lemma_exception_ruler
from interpretant.corpus.static_view import fold, static_tokens
from interpretant.pipelines.annotate_corpus import annotate_corpus
from interpretant.pipelines.build_static_view import build_static_view
from tests.corpus_catalog import ANNOTATE_SETTINGS, SELECTION


def annotated(text: str, nlp: Language) -> Doc:
    (doc,) = annotate_texts([("https://openalex.org/W1", text)], nlp)
    return doc


def test_static_tokens_drops_punctuation_and_numbers_and_lowers_case(nlp: Language) -> None:
    view = static_tokens(annotated("The model learns 12 tasks, in 2019.", nlp), {}, NO_COMPOUNDS)
    assert view.tokens == ["the", "model", "learn", "task", "in"]


def test_static_tokens_lemmatises_plural_nouns_and_third_person_verbs_only(nlp: Language) -> None:
    view = static_tokens(
        annotated("Attention improves results. Earlier models improved.", nlp), {}, NO_COMPOUNDS
    )
    assert view.tokens == ["attention", "improve", "result", "earlier", "model", "improved"]


def test_static_tokens_leave_auxiliaries_as_they_are(nlp: Language) -> None:
    view = static_tokens(annotated("The model is large and has improved.", nlp), {}, NO_COMPOUNDS)
    assert view.tokens == ["the", "model", "is", "large", "and", "has", "improved"]


def test_static_tokens_keep_offsets_and_finite_verbs(nlp: Language) -> None:
    doc = annotated("Attention improves results.", nlp)
    view = static_tokens(doc, {}, NO_COMPOUNDS)
    assert view.starts == [0, 10, 19]
    assert [doc.text[start : start + 4] for start in view.starts] == ["Atte", "impr", "resu"]
    assert view.finite_verbs == [False, True, False]


def test_static_tokens_fold_accents(nlp: Language) -> None:
    assert (
        static_tokens(annotated("Naïve Bayes performs well.", nlp), {}, NO_COMPOUNDS).tokens[0]
        == "naive"
    )


def test_static_tokens_spell_the_written_word_lemmas_included(nlp: Language) -> None:
    spellings = {"behaviour": "behavior", "colour": "color"}
    doc = annotated("Their behaviours changed the colour of results.", nlp)
    view = static_tokens(doc, spellings, NO_COMPOUNDS)
    assert view.tokens == ["their", "behavior", "changed", "the", "color", "of", "result"]
    assert view.starts == static_tokens(doc, {}, NO_COMPOUNDS).starts


def test_static_tokens_leave_accented_words_unspelled(nlp: Language) -> None:
    spellings = {"realisation": "realization"}
    doc = annotated("La réalisation. The realisation of a model.", nlp)
    view = static_tokens(doc, spellings, NO_COMPOUNDS)
    assert "realisation" in view.tokens
    assert "realization" in view.tokens
    assert view.tokens.index("realisation") < view.tokens.index("realization")


def tagged(words: list[str], pos: list[str], tags: list[str]) -> Doc:
    """A document with fixed tags, so the compound tests do not depend on the tagger."""
    return Doc(spacy.blank("en").vocab, words=words, pos=pos, tags=tags)


def test_a_spaced_noun_compound_writes_its_standard_then_an_empty_entry() -> None:
    doc = tagged(
        ["The", "data", "base", "grew"], ["DET", "NOUN", "NOUN", "VERB"], ["DT", "NN", "NN", "VBD"]
    )
    compounds = build_compound_map([("data base", "database")])
    view = static_tokens(doc, {}, compounds)
    assert view.tokens == ["the", "database", "", "grew"]
    assert view.starts == static_tokens(doc, {}, NO_COMPOUNDS).starts


def test_the_gate_keeps_other_spaced_pairs_apart() -> None:
    doc = tagged(
        ["This", "is", "a", "long", "term", "plan"],
        ["DET", "AUX", "DET", "ADJ", "NOUN", "NOUN"],
        ["DT", "VBZ", "DT", "JJ", "NN", "NN"],
    )
    compounds = build_compound_map([("is a", "isa"), ("long term", "long-term")])
    assert static_tokens(doc, {}, compounds).tokens == ["this", "is", "a", "long", "term", "plan"]


def test_one_token_forms_are_rewritten_whole_and_may_split() -> None:
    doc = tagged(
        ["The", "user-interface", "of", "non-profit", "groups"],
        ["DET", "NOUN", "ADP", "ADJ", "NOUN"],
        ["DT", "NN", "IN", "JJ", "NN"],
    )
    compounds = build_compound_map(
        [("user-interface", "user interface"), ("non-profit", "nonprofit")]
    )
    view = static_tokens(doc, {}, compounds)
    assert view.tokens == ["the", "user interface", "of", "nonprofit", "groups"]


def test_spelling_comes_before_the_compound_map() -> None:
    doc = tagged(["A", "colour-blind", "test"], ["DET", "ADJ", "NOUN"], ["DT", "JJ", "NN"])
    compounds = build_compound_map([("color-blind", "colorblind")])
    assert static_tokens(doc, {"colour": "color"}, compounds).tokens == ["a", "colorblind", "test"]


def test_accented_tokens_are_left_out_of_the_compound_map() -> None:
    doc = tagged(["café", "bar", "café-bar"], ["NOUN", "NOUN", "NOUN"], ["NN", "NN", "NN"])
    compounds = build_compound_map([("cafe bar", "cafebar"), ("cafe-bar", "cafebar")])
    assert static_tokens(doc, {}, compounds).tokens == ["cafe", "bar", "cafe-bar"]


def test_overlapping_compounds_are_taken_left_to_right() -> None:
    doc = tagged(["data", "base", "line"], ["NOUN", "NOUN", "NOUN"], ["NN", "NN", "NN"])
    compounds = build_compound_map([("data base", "database"), ("base line", "baseline")])
    assert static_tokens(doc, {}, compounds).tokens == ["database", "", "line"]


def test_fold_lowers_case_folds_accents_and_writes_the_en_dash_as_a_hyphen() -> None:
    assert fold("Café–Style") == "cafe-style"


def test_build_static_view_writes_a_row_per_cached_document(
    corpus_catalog_file: Path, nlp: Language
) -> None:
    catalog = read_catalog(corpus_catalog_file)
    annotate_corpus(catalog, SELECTION, ANNOTATE_SETTINGS)
    result = build_static_view(catalog, SELECTION)
    ruler = lemma_exception_ruler(nlp.vocab, load_lemma_exceptions(catalog))
    docs = [ruler(doc) for doc in load_annotations(catalog, "ai_2010", nlp.vocab)]
    spellings = {spelling.form: spelling.standard for spelling in load_spellings(catalog)}
    compounds = build_compound_map((c.form, c.standard) for c in load_compounds(catalog))
    table = load_static_view(catalog, "ai_2010")
    views = [static_tokens(doc, spellings, compounds) for doc in docs]
    assert table["tokens"].to_list() == [view.tokens for view in views]
    assert table["starts"].to_list() == [view.starts for view in views]
    assert result.exception_count == len(load_lemma_exceptions(catalog))
    assert result.spelling_count == len(spellings)
    heights = sum(load_static_view(catalog, part).height for part in result.parts)
    assert result.document_count == heights
    assert result.token_count == sum(len(tokens) for tokens in table["tokens"]) + sum(
        int(load_static_view(catalog, part)["tokens"].list.len().sum())
        for part in result.parts
        if part != "ai_2010"
    )


def test_build_static_view_applies_the_catalogs_lemma_exceptions(
    corpus_catalog_file: Path, tmp_path: Path
) -> None:
    catalog = read_catalog(corpus_catalog_file)
    annotate_corpus(catalog, SELECTION, ANNOTATE_SETTINGS)
    build_static_view(catalog, SELECTION)
    repository_tokens = load_static_view(catalog, "ai_2010")["tokens"].explode().to_list()
    assert "sense" in repository_tokens
    assert "sens" not in repository_tokens
    assert "model" in repository_tokens
    exceptions_file = tmp_path / "lemma_exceptions.tsv"
    exceptions_file.write_text(
        "form\tlemma\ttag\tsource\nmodels\tmodels\tNNS\tauthor\n", encoding="utf-8"
    )
    entry = catalog.lemma_exceptions.model_copy(update={"path": exceptions_file})
    other = catalog.model_copy(update={"lemma_exceptions": entry})
    result = build_static_view(other, SELECTION)
    tokens = load_static_view(other, "ai_2010")["tokens"].explode().to_list()
    assert result.exception_count == 1
    assert "models" in tokens
    assert "model" not in tokens
    assert "sens" in tokens


def test_build_static_view_applies_the_catalogs_spelling_map(
    corpus_catalog_file: Path, tmp_path: Path
) -> None:
    catalog = read_catalog(corpus_catalog_file)
    annotate_corpus(catalog, SELECTION, ANNOTATE_SETTINGS)
    spellings_file = tmp_path / "spellings.tsv"
    spellings_file.write_text("form\tstandard\tsource\nmodel\tmodelx\trules\n", encoding="utf-8")
    entry = catalog.spellings.model_copy(update={"path": spellings_file})
    other = catalog.model_copy(update={"spellings": entry})
    result = build_static_view(other, SELECTION)
    tokens = load_static_view(other, "ai_2010")["tokens"].explode().to_list()
    assert result.spelling_count == 1
    assert "modelx" in tokens
    assert "model" not in tokens


def test_build_static_view_applies_the_catalogs_compound_map(
    corpus_catalog_file: Path, tmp_path: Path
) -> None:
    catalog = read_catalog(corpus_catalog_file)
    annotate_corpus(catalog, SELECTION, ANNOTATE_SETTINGS)
    compounds_file = tmp_path / "compounds.tsv"
    compounds_file.write_text(
        "form\tstandard\tsource\nmachine translation\tmachine-translation\tfrequency\n",
        encoding="utf-8",
    )
    entry = catalog.compounds.model_copy(update={"path": compounds_file})
    other = catalog.model_copy(update={"compounds": entry})
    result = build_static_view(other, SELECTION)
    rows = load_static_view(other, "ai_2010")["tokens"].to_list()
    tokens = [token for row in rows for token in row]
    assert result.compound_count == 1
    assert "machine-translation" in tokens
    assert tokens[tokens.index("machine-translation") + 1] == ""
    assert "translation" not in tokens


def test_build_static_view_writes_the_same_rows_in_worker_processes(
    corpus_catalog_file: Path,
) -> None:
    catalog = read_catalog(corpus_catalog_file)
    annotate_corpus(catalog, SELECTION, ANNOTATE_SETTINGS)
    alone = build_static_view(catalog, SELECTION)
    tables = {part: load_static_view(catalog, part) for part in alone.parts}
    in_workers = build_static_view(catalog, SELECTION, process_count=2)
    assert in_workers == alone
    assert all(load_static_view(catalog, part).equals(tables[part]) for part in tables)


def test_save_static_view_rejects_other_columns(catalog: CatalogConfig) -> None:
    with pytest.raises(ValueError, match="static view's columns"):
        save_static_view(catalog, pl.DataFrame({"work_id": ["W1"]}), part="ai_2010")


def test_load_static_view_needs_the_part_built(catalog: CatalogConfig) -> None:
    with pytest.raises(FileNotFoundError, match="no static view for part"):
        load_static_view(catalog, "ai_2010")


def test_build_static_view_command_counts_the_documents(corpus_catalog_file: Path) -> None:
    catalog_option = ["--catalog", str(corpus_catalog_file)]
    runner = CliRunner()
    annotated_run = runner.invoke(cli, ["annotate-corpus", *catalog_option, "--process-count", "1"])
    assert annotated_run.exit_code == 0
    outcome = runner.invoke(cli, ["build-static-view", *catalog_option, "--process-count", "1"])
    assert outcome.exit_code == 0, outcome.output
    assert "static view of" in outcome.output
    run_log = read_catalog(corpus_catalog_file).run_log.path.read_text().splitlines()
    record = json.loads(run_log[-1])
    assert record["command"] == "build-static-view"
    assert record["counts"]["documents"] == 5
