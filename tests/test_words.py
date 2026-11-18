"""The word table: read, validated, and kept apart from the lemma exceptions."""

from __future__ import annotations

import pydantic
import pytest

from interpretant.catalog.config import CatalogConfig, read_catalog
from interpretant.catalog.lemma_exceptions import load_lemma_exceptions
from interpretant.catalog.words import load_words

HEADER = "# A comment line.\nlemma\tpos\tmatch\theld_out\n"


def write_words(catalog: CatalogConfig, rows: str, header: str = HEADER) -> CatalogConfig:
    """Write a word table at the catalog's path and return the catalog."""
    path = catalog.words.path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + rows, encoding="utf-8")
    return catalog


def test_load_words_reads_the_repository_table() -> None:
    words = {word.lemma: word for word in load_words(read_catalog())}
    assert len(words) == 53
    assert words["learning"].pos == ("NOUN", "PROPN", "VERB")
    assert words["having"].match == "form"
    assert words["neural network"].is_phrase
    assert not any(word.held_out for word in words.values())


@pytest.mark.parametrize(
    ("header", "rows", "error"),
    [
        ("lemma\tpos\tmatch\n", "model\tNOUN\tlemma\n", ValueError),
        (HEADER, "Model\tNOUN\tlemma\tno\n", pydantic.ValidationError),
        (HEADER, "model\tNOUNS\tlemma\tno\n", pydantic.ValidationError),
        (HEADER, "model\tNOUN\tstem\tno\n", pydantic.ValidationError),
        (HEADER, "model\tNOUN\tlemma\tmaybe\n", ValueError),
        (HEADER, "model\tNOUN\tlemma\tno\nmodel\tVERB\tlemma\tno\n", ValueError),
    ],
    ids=["header", "uppercase", "unknown tag", "unknown match", "held out", "repeated lemma"],
)
def test_load_words_refuses_a_bad_table(
    catalog: CatalogConfig, header: str, rows: str, error: type[Exception]
) -> None:
    with pytest.raises(error):
        load_words(write_words(catalog, rows, header))


def test_no_lemma_exception_maps_a_word_of_the_table_away() -> None:
    catalog = read_catalog()
    words = {word.lemma for word in load_words(catalog) if not word.is_phrase}
    mapped_away = [
        exception.form
        for exception in load_lemma_exceptions(catalog)
        if exception.form in words and exception.lemma != exception.form
    ]
    assert mapped_away == []
