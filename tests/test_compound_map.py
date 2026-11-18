"""The compound map: read, validated, and kept from writing a word of the table away."""

from __future__ import annotations

import pydantic
import pytest

from interpretant.catalog.compounds import load_compounds
from interpretant.catalog.config import CatalogConfig, read_catalog
from interpretant.catalog.words import load_words
from interpretant.corpus.compound_spelling import build_compound_map, compounds_written_away
from tests.sealed import fail_sealed

HEADER = "# A comment line.\nform\tstandard\tsource\n"


def write_compounds(catalog: CatalogConfig, rows: str, header: str = HEADER) -> CatalogConfig:
    """Write a compound map at the catalog's path and return the catalog."""
    path = catalog.compounds.path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + rows, encoding="utf-8")
    return catalog


def test_load_compounds_reads_a_small_map(catalog: CatalogConfig) -> None:
    rows = (
        "data base\tdatabase\tfrequency\n"
        "data-base\tdatabase\tfrequency\n"
        "user-interface\tuser interface\tword_table\n"
    )
    compounds = load_compounds(write_compounds(catalog, rows))
    assert [(c.form, c.standard, c.source, c.is_spaced) for c in compounds] == [
        ("data base", "database", "frequency", True),
        ("data-base", "database", "frequency", False),
        ("user-interface", "user interface", "word_table", False),
    ]


def test_load_compounds_reads_the_repository_map() -> None:
    compounds = {compound.form: compound for compound in load_compounds(read_catalog())}
    assert len(compounds) == 1971
    assert compounds["on-line"].standard == "online"
    assert compounds["data base"].standard == "database"
    assert compounds["user-interface"].standard == "user interface"
    assert compounds["net-work"].source == "word_table"
    assert {"re-presentation", "re-creation", "w hen", "world wide"}.isdisjoint(compounds)


@pytest.mark.parametrize(
    ("header", "rows", "error"),
    [
        ("form\tstandard\n", "data base\tdatabase\n", ValueError),
        (HEADER, "Data base\tdatabase\tfrequency\n", pydantic.ValidationError),
        (HEADER, "state-of-the-art\tstate of the art\tfrequency\n", pydantic.ValidationError),
        (HEADER, "covid-19\tcovid19\tfrequency\n", pydantic.ValidationError),
        (HEADER, "data base\tdatabank\tfrequency\n", pydantic.ValidationError),
        (HEADER, "data-base\tdatab-ase\tfrequency\n", pydantic.ValidationError),
        (HEADER, "data base\tdata base\tfrequency\n", pydantic.ValidationError),
        (HEADER, "data base\tdatabase\tagents\n", pydantic.ValidationError),
        (HEADER, "data base\tdatabase\tfrequency\ndata base\tdata-base\tfrequency\n", ValueError),
        (HEADER, "data base\tdata-base\tfrequency\ndata-base\tdatabase\tfrequency\n", ValueError),
        (HEADER, "data base\tdatabase\tfrequency\ndata-base\tdata base\tfrequency\n", ValueError),
        (HEADER, "web site\twebsite\tfrequency\nsite\tsi-te\tfrequency\n", ValueError),
    ],
    ids=[
        "header",
        "uppercase",
        "three parts",
        "digits",
        "other letters",
        "other split",
        "itself",
        "unknown source",
        "repeated form",
        "chain",
        "two standards",
        "spaced word is a form",
    ],
)
def test_load_compounds_refuses_a_bad_map(
    catalog: CatalogConfig, header: str, rows: str, error: type[Exception]
) -> None:
    with pytest.raises(error):
        load_compounds(write_compounds(catalog, rows, header))


def test_no_compound_writes_a_word_of_the_table_away() -> None:
    catalog = read_catalog()
    compounds = build_compound_map(
        (compound.form, compound.standard) for compound in load_compounds(catalog)
    )
    words = [word.lemma for word in load_words(catalog)]
    fail_sealed(compounds_written_away(compounds, words), "written away by a compound")
