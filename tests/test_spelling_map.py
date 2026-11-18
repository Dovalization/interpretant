"""The spelling map: read, validated, and kept from sending a word of the table away."""

from __future__ import annotations

import pydantic
import pytest

from interpretant.catalog.config import CatalogConfig, read_catalog
from interpretant.catalog.spellings import load_spellings
from interpretant.catalog.words import load_words
from interpretant.corpus.spelling import mapped_away, spell

HEADER = "# A comment line.\nform\tstandard\tsource\n"


def write_spellings(catalog: CatalogConfig, rows: str, header: str = HEADER) -> CatalogConfig:
    """Write a spelling map at the catalog's path and return the catalog."""
    path = catalog.spellings.path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + rows, encoding="utf-8")
    return catalog


def test_load_spellings_reads_the_repository_map() -> None:
    spellings = {spelling.form: spelling for spelling in load_spellings(read_catalog())}
    assert len(spellings) == 1946
    assert spellings["behaviour"].standard == "behavior"
    assert spellings["dialog"].standard == "dialogue"
    assert {"programme", "metre", "prologue"}.isdisjoint(spellings)


def test_load_spellings_reads_a_small_map(catalog: CatalogConfig) -> None:
    rows = "colour\tcolor\tnorm_table\noptimisation\toptimization\trules\n"
    spellings = load_spellings(write_spellings(catalog, rows))
    assert [(s.form, s.standard, s.source) for s in spellings] == [
        ("colour", "color", "norm_table"),
        ("optimisation", "optimization", "rules"),
    ]


@pytest.mark.parametrize(
    ("header", "rows", "error"),
    [
        ("form\tstandard\n", "colour\tcolor\n", ValueError),
        (HEADER, "Colour\tcolor\tnorm_table\n", pydantic.ValidationError),
        (HEADER, "edge-colour\tedge-color\tnorm_table\n", pydantic.ValidationError),
        (HEADER, "colour\tcolor\tagents\n", pydantic.ValidationError),
        (HEADER, "colour\tcolor\tnorm_table\ncolour\tcolors\tnorm_table\n", ValueError),
        (HEADER, "colour\tcolour\tnorm_table\n", ValueError),
        (HEADER, "colour\tcolor\tnorm_table\ncolor\tkolor\tnorm_table\n", ValueError),
    ],
    ids=["header", "uppercase", "hyphen", "unknown source", "repeated form", "itself", "chain"],
)
def test_load_spellings_refuses_a_bad_map(
    catalog: CatalogConfig, header: str, rows: str, error: type[Exception]
) -> None:
    with pytest.raises(error):
        load_spellings(write_spellings(catalog, rows, header))


def test_no_spelling_sends_a_word_of_the_table_away() -> None:
    catalog = read_catalog()
    spellings = {spelling.form: spelling.standard for spelling in load_spellings(catalog)}
    words = [word.lemma for word in load_words(catalog)]
    assert mapped_away(spellings, words) == []


def test_no_standard_merges_two_words_of_the_table() -> None:
    catalog = read_catalog()
    spellings = {spelling.form: spelling.standard for spelling in load_spellings(catalog)}
    lemmas = [word.lemma for word in load_words(catalog) if not word.is_phrase]
    assert len({spell(lemma, spellings) for lemma in lemmas}) == len(lemmas)
