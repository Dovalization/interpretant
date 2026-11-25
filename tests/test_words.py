"""The word table: read, validated, and kept apart from the lemma exceptions."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pydantic
import pytest

from interpretant.catalog.config import CatalogConfig, read_catalog
from interpretant.catalog.lemma_exceptions import load_lemma_exceptions
from interpretant.catalog.words import Word, load_words, seal_held_out
from tests.sealed import fail_sealed, requires_held_out

HEADER = "# A comment line.\nlemma\tpos\tmatch\n"
HELD_OUT_HEADER = "lemma\tpos\tmatch\tsealed_name\n"
SEALED = "held-out:0123456789ab"


def write_words(
    catalog: CatalogConfig,
    rows: str,
    header: str = HEADER,
    held_out_rows: str = "",
    held_out_header: str = HELD_OUT_HEADER,
) -> CatalogConfig:
    """Write the development and held-out word tables at the catalog's paths; the catalog."""
    for path, text in (
        (catalog.words.path, header + rows),
        (catalog.held_out_words.path, held_out_header + held_out_rows),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return catalog


HELD_OUT_SHA256 = "18dfd3a5fcf3c08289d11e62e5252e9e47af5827de3cb78312ce3601b23aa1b6"


@requires_held_out
def test_load_words_reads_the_repository_table() -> None:
    words = {word.lemma: word for word in load_words(read_catalog())}
    assert len(words) == 99
    assert words["learning"].pos == ("NOUN", "PROPN", "VERB")
    assert words["having"].match == "form"
    assert words["neural network"].is_phrase
    held_out = sorted(lemma for lemma, word in words.items() if word.held_out)
    assert len(held_out) == 46
    # The sealed set's hash (docs/held_out.md): a change to it fails here without naming a word.
    listing = "".join(f"{lemma}\n" for lemma in held_out)
    assert hashlib.sha256(listing.encode()).hexdigest() == HELD_OUT_SHA256


def test_load_words_reads_the_development_words_then_the_held_out_ones(
    catalog: CatalogConfig,
) -> None:
    write_words(catalog, "model\tNOUN\tlemma\n", held_out_rows=f"query\tNOUN\tlemma\t{SEALED}\n")

    model, query = load_words(catalog)

    assert (model.lemma, model.held_out, model.reported_name) == ("model", False, "model")
    assert (query.lemma, query.held_out, query.reported_name) == ("query", True, SEALED)


def test_load_words_refuses_to_run_without_the_held_out_words(catalog: CatalogConfig) -> None:
    write_words(catalog, "model\tNOUN\tlemma\n")
    catalog.held_out_words.path.unlink()

    with pytest.raises(FileNotFoundError, match="dvc pull"):
        load_words(catalog)


@pytest.mark.parametrize(
    ("header", "rows", "held_out_rows", "error"),
    [
        ("lemma\tpos\n", "model\tNOUN\n", "", ValueError),
        (HEADER, "Model\tNOUN\tlemma\n", "", pydantic.ValidationError),
        (HEADER, "model\tNOUNS\tlemma\n", "", pydantic.ValidationError),
        (HEADER, "model\tNOUN\tstem\n", "", pydantic.ValidationError),
        (HEADER, "model\tNOUN\tlemma\nmodel\tVERB\tlemma\n", "", ValueError),
        (HEADER, "model\tNOUN\tlemma\n", f"model\tNOUN\tlemma\t{SEALED}\n", ValueError),
        (HEADER, "", "query\tNOUN\tlemma\tquery-hash\n", pydantic.ValidationError),
    ],
    ids=[
        "header",
        "uppercase",
        "unknown tag",
        "unknown match",
        "repeated lemma",
        "in both tables",
        "bad sealed name",
    ],
)
def test_load_words_refuses_a_bad_table(
    catalog: CatalogConfig, header: str, rows: str, held_out_rows: str, error: type[Exception]
) -> None:
    with pytest.raises(error):
        load_words(write_words(catalog, rows, header, held_out_rows))


@requires_held_out
def test_no_lemma_exception_maps_a_word_of_the_table_away() -> None:
    catalog = read_catalog()
    words = {word.lemma for word in load_words(catalog) if not word.is_phrase}
    mapped_away = [
        exception.form
        for exception in load_lemma_exceptions(catalog)
        if exception.form in words and exception.lemma != exception.form
    ]
    fail_sealed(mapped_away, "mapped away by a lemma exception")


def test_reported_name_shows_a_held_out_word_only_by_its_sealed_name() -> None:
    shown = Word(lemma="attention", pos=("NOUN",), match="lemma", held_out=False)
    sealed = Word(
        lemma="attention", pos=("NOUN",), match="lemma", held_out=True, sealed_name=SEALED
    )
    assert shown.reported_name == "attention"
    assert sealed.reported_name == SEALED


def test_only_a_held_out_word_has_a_sealed_name() -> None:
    with pytest.raises(pydantic.ValidationError):
        Word(lemma="attention", pos=("NOUN",), match="lemma", held_out=True)
    with pytest.raises(pydantic.ValidationError):
        Word(lemma="attention", pos=("NOUN",), match="lemma", held_out=False, sealed_name=SEALED)


def test_seal_held_out_names_a_held_out_word_in_its_common_forms_by_its_sealed_name() -> None:
    sealed = Word(lemma="encode", pos=("VERB",), match="lemma", held_out=True, sealed_name=SEALED)
    query = Word(
        lemma="query",
        pos=("NOUN",),
        match="lemma",
        held_out=True,
        sealed_name="held-out:ba9876543210",
    )
    shown = Word(lemma="attention", pos=("NOUN",), match="lemma", held_out=False)
    text = "Encoding, encodes and encoded; two queries; attention stays; encoder stays."
    name, other = sealed.reported_name, query.reported_name
    assert seal_held_out(text, [sealed, query, shown]) == (
        f"{name}, {name} and {name}; two {other}; attention stays; encoder stays."
    )
    assert seal_held_out(text, [shown]) == text


@requires_held_out
def test_committed_review_notes_name_no_held_out_word() -> None:
    """Every judge's note in a committed review file is sealed (`scripts/review/seal_notes.py`)."""
    words = load_words(read_catalog())
    held_out = [word for word in words if word.held_out]
    listing = subprocess.run(
        ["git", "ls-files", "data/review"], capture_output=True, text=True, check=True
    )
    named: set[str] = set()
    for name in listing.stdout.split():
        if name.endswith(".json"):
            for note in _notes(json.loads(Path(name).read_text(encoding="utf-8"))):
                named |= {w.lemma for w in held_out if seal_held_out(note, [w]) != note}
    fail_sealed(sorted(named), "named in committed review notes")


def _notes(value: object) -> list[str]:
    if isinstance(value, dict):
        own = [value["note"]] if isinstance(value.get("note"), str) else []
        return own + [note for inner in value.values() for note in _notes(inner)]
    if isinstance(value, list):
        return [note for inner in value for note in _notes(inner)]
    return []
