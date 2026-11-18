"""The lemma exceptions: the file, read and validated, and the attribute ruler applying it."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pydantic
import pytest
import spacy
from spacy.tokens import Doc, DocBin

from interpretant.catalog.config import CatalogConfig, read_catalog
from interpretant.catalog.lemma_exceptions import LemmaException, load_lemma_exceptions
from interpretant.corpus.audit import load_target_words
from interpretant.corpus.lemma_exceptions import lemma_exception_ruler
from interpretant.corpus.static_view import static_tokens

HEADER = "# A comment line.\nform\tlemma\ttag\tsource\n"


def write_exceptions(catalog: CatalogConfig, rows: str, header: str = HEADER) -> CatalogConfig:
    """Write a lemma exceptions file at the catalog's path and return the catalog."""
    path = catalog.lemma_exceptions.path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + rows, encoding="utf-8")
    return catalog


def test_load_lemma_exceptions_reads_the_repository_file() -> None:
    exceptions = load_lemma_exceptions(read_catalog())
    assert len(exceptions) == 3297
    assert {exception.tag for exception in exceptions} == {"NNS", "VBZ"}
    by_form = {exception.form: exception for exception in exceptions}
    assert by_form['clitics"--entities'].lemma == 'clitics"--entity'
    assert by_form["data"].lemma == "data"
    assert by_form["senses"].lemma == "sense"


def test_a_header_only_file_has_no_exceptions(catalog: CatalogConfig) -> None:
    assert load_lemma_exceptions(write_exceptions(catalog, "")) == []


@pytest.mark.parametrize(
    ("header", "rows", "error"),
    [
        ("form\tlemma\ttag\n", "data\tdata\tNNS\n", ValueError),
        (HEADER, "data\tdata\t\toverride\n", pydantic.ValidationError),
        (HEADER, "1990s\t1990\tNNS\tagents\n", pydantic.ValidationError),
        (HEADER, "data\tdata\tNNS\toverride\ndata\tdatum\tNNS\tagents\n", ValueError),
    ],
    ids=["header", "empty tag", "no letter", "repeated form and tag"],
)
def test_load_lemma_exceptions_refuses_a_bad_file(
    catalog: CatalogConfig, header: str, rows: str, error: type[Exception]
) -> None:
    with pytest.raises(error):
        load_lemma_exceptions(write_exceptions(catalog, rows, header))


def test_a_form_may_take_one_lemma_per_reading(catalog: CatalogConfig) -> None:
    rows = "thanks\tthanks\tNNS\tauthor_call\nthanks\tthank\tVBZ\tagents\n"
    exceptions = load_lemma_exceptions(write_exceptions(catalog, rows))
    assert [(exception.tag, exception.lemma) for exception in exceptions] == [
        ("NNS", "thanks"),
        ("VBZ", "thank"),
    ]


def test_no_lemma_exception_maps_a_target_word_away() -> None:
    targets = {
        word for word in load_target_words(Path("data/words_of_interest.txt")) if "_" not in word
    }
    exceptions = load_lemma_exceptions(read_catalog())
    mapped_away = [e.form for e in exceptions if e.form in targets and e.lemma != e.form]
    assert mapped_away == []


def exception(form: str, lemma: str, tag: Literal["NNS", "VBZ"]) -> LemmaException:
    """One exception from the author, for a test."""
    return LemmaException(form=form, lemma=lemma, tag=tag, source="author")


def test_an_exception_sets_the_lemma_only_under_its_tag() -> None:
    vocab = spacy.blank("en").vocab
    doc = Doc(vocab, words=["means", "means"], tags=["NNS", "VBZ"], lemmas=["mean", "mean"])
    lemma_exception_ruler(vocab, [exception("means", "means", "NNS")])(doc)
    assert [token.lemma_ for token in doc] == ["means", "mean"]


def test_an_exception_matches_the_form_in_any_case() -> None:
    vocab = spacy.blank("en").vocab
    doc = Doc(vocab, words=["Senses"], tags=["NNS"], lemmas=["sens"])
    lemma_exception_ruler(vocab, [exception("senses", "sense", "NNS")])(doc)
    assert doc[0].lemma_ == "sense"


def test_auxiliaries_keep_their_form_whatever_the_exceptions() -> None:
    vocab = spacy.blank("en").vocab
    words = ["The", "model", "has", "senses"]
    doc = Doc(
        vocab,
        words=words,
        tags=["DT", "NN", "VBZ", "NNS"],
        pos=["DET", "NOUN", "AUX", "NOUN"],
        lemmas=["the", "model", "have", "sens"],
    )
    ruler = lemma_exception_ruler(
        vocab, [exception("has", "hav", "VBZ"), exception("senses", "sense", "NNS")]
    )
    assert static_tokens(ruler(doc), {}).tokens == ["the", "model", "has", "sense"]


def test_the_ruler_applies_to_documents_loaded_from_the_cache() -> None:
    written = Doc(
        spacy.blank("en").vocab,
        words=["Others", "agree"],
        tags=["NNS", "VBP"],
        lemmas=["other", "agree"],
    )
    cache = DocBin(docs=[written], store_user_data=True).to_bytes()
    vocab = spacy.blank("en").vocab
    (doc,) = DocBin().from_bytes(cache).get_docs(vocab)
    lemma_exception_ruler(vocab, [exception("others", "others", "NNS")])(doc)
    assert [token.lemma_ for token in doc] == ["others", "agree"]
