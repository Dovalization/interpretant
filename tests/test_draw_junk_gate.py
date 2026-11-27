"""Drawing the junk gate's sets from a small catalog: the files, their order and their content."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from interpretant.catalog.abstracts import save_abstracts
from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.junk_gate import (
    RUN3_RECORDS_COLUMNS,
    JunkLabel,
    JunkLabels,
    save_junk_labels,
    save_run3_records,
)
from interpretant.catalog.static_view import STATIC_VIEW_SCHEMA, save_static_view
from interpretant.catalog.usages import USAGES_COLUMNS, save_usages
from interpretant.pipelines import draw_junk_gate as pipeline
from interpretant.pipelines.draw_junk_gate import draw_junk_gate
from interpretant.usages.identifiers import build_content_key, build_draw_key
from tests.abstracts_tables import abstract_row, abstracts_table

SEALED = "held-out:0123456789ab"
OPENING = "Abstract: We study attention in network models."
MIDDLE = "Attention improves recall in every network we tried."
CLOSING = "Copyright 1995 Elsevier."


def write_words(catalog: CatalogConfig) -> None:
    catalog.words.path.parent.mkdir(parents=True, exist_ok=True)
    catalog.words.path.write_text("lemma\tpos\tmatch\nattention\tNOUN|PROPN\tlemma\n")
    catalog.held_out_words.path.write_text(
        f"lemma\tpos\tmatch\tsealed_name\nnetwork\tNOUN\tlemma\t{SEALED}\n"
    )


def work_id(work: int) -> str:
    return f"https://openalex.org/W{work}"


def build_abstract(work: int, year: int, **changes: Any) -> dict[str, Any]:
    """A three-sentence abstract whose label and copyright line are removed."""
    row = abstract_row(work, [OPENING, MIDDLE, CLOSING], year=year, **changes)
    closing = row["text"].index(CLOSING)
    row["removals"] = [
        {"start": 0, "end": 9, "reason": "abstract_label"},
        {"start": closing, "end": len(row["text"]), "reason": "copyright"},
    ]
    return row


def build_usage(abstract: dict[str, Any], word: str, *, list_like: bool = False) -> dict[str, Any]:
    """The usage of ``word`` in the abstract's middle sentence, drawn."""
    text, sentence = abstract["text"], abstract["sentences"][1]
    start = text.index(word.capitalize() if word == "attention" else word, sentence["start"])
    end = start + len(word)
    content_key = build_content_key(
        text[sentence["start"] : sentence["end"]],
        start - sentence["start"],
        end - sentence["start"],
    )
    return {
        "work_id": abstract["work_id"],
        "word": word,
        "content_key": content_key,
        "draw_key": build_draw_key(1, abstract["work_id"], word, content_key),
        "sentence_number": 1,
        "sentence_start": sentence["start"],
        "sentence_end": sentence["end"],
        "target_start": start,
        "target_end": end,
        "compound": None,
        "position": None,
        "pos": "NOUN",
        "tag": "NN",
        "morph": "Number=Sing",
        "lemma": word,
        "entity": None,
        "static_token": word,
        "sentence_word_count": 8,
        "sentence_language": "en",
        "sentence_confidence": 0.99,
        "sentence_exclusion": None,
        "inside_name": False,
        "in_scope": True,
        "drawn": True,
        "model": "en_core_web_sm",
        "windowed": False,
        "role": "head",
        "capitalised": False,
        "mention": False,
        "list_like": list_like,
        "copy_of": None,
    }


def build_static_entry(abstract: dict[str, Any], *, has_finite_verb: bool) -> dict[str, Any]:
    """The static view's tokens of an abstract: its words, with a finite verb in its middle."""
    text = abstract["text"]
    tokens, starts, position = [], [], 0
    for token in text.split(" "):
        tokens.append(token.lower())
        starts.append(text.index(token, position))
        position = starts[-1] + len(token)
    finite_verbs = [has_finite_verb and token == "improves" for token in tokens]
    return {
        "work_id": abstract["work_id"],
        "tokens": tokens,
        "starts": starts,
        "finite_verbs": finite_verbs,
    }


def write_part(catalog: CatalogConfig, year: int, works: list[dict[str, Any]]) -> None:
    """One part's abstracts, static view and usages; each work's options say what it holds."""
    part = f"ai_{year - year % 10}"
    abstracts, entries, usages = [], [], []
    for work in works:
        flags = ["repository_only"] if work.get("repository_only") else []
        abstract = build_abstract(work["number"], year, flags=flags)
        abstracts.append(abstract)
        entries.append(build_static_entry(abstract, has_finite_verb=work.get("finite_verb", True)))
        usages.append(build_usage(abstract, "attention", list_like=work.get("list_like", False)))
        if work.get("network"):
            usages.append(build_usage(abstract, "network"))
    save_abstracts(catalog, abstracts_table(abstracts), part)
    save_static_view(catalog, pl.DataFrame(entries, schema=STATIC_VIEW_SCHEMA), part)
    save_usages(catalog, pl.DataFrame(usages, schema=USAGES_COLUMNS), part)


def write_corpus(catalog: CatalogConfig) -> None:
    write_words(catalog)
    write_part(
        catalog,
        1995,
        [
            {"number": 1, "network": True},
            {"number": 2, "finite_verb": False},
            {"number": 3, "repository_only": True},
            {"number": 4, "list_like": True},
            {"number": 5},
            {"number": 6, "finite_verb": False},
        ],
    )
    write_part(
        catalog,
        2005,
        [
            {"number": 7},
            {"number": 8, "list_like": True},
            {"number": 9, "repository_only": True},
            {"number": 10, "network": True},
        ],
    )


@pytest.fixture
def small_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sizes small enough that L leaves records for E."""
    monkeypatch.setattr(pipeline, "LABELLED_PER_DECADE", 2)
    monkeypatch.setattr(pipeline, "ENRICHMENT_PER_STRATUM", 1)


def read_folder(folder: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(folder.iterdir())}


def read_items(folder: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for path in sorted(folder.glob("batch_*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


@pytest.mark.usefixtures("small_sizes")
def test_the_sample_is_drawn_into_the_same_files_each_time(
    catalog: CatalogConfig, tmp_path: Path
) -> None:
    write_corpus(catalog)

    draw_junk_gate(catalog, tmp_path / "first", draw_set="sample", process_count=1)
    draw_junk_gate(catalog, tmp_path / "second", draw_set="sample", process_count=1)

    first = read_folder(tmp_path / "first")
    assert set(first) == {"batch_00.jsonl", "key.json", "counts.json"}
    assert first == read_folder(tmp_path / "second")


@pytest.mark.usefixtures("small_sizes")
def test_the_samples_items_show_only_the_kept_text_with_the_sentence_marked(
    catalog: CatalogConfig, tmp_path: Path
) -> None:
    write_corpus(catalog)

    draw = draw_junk_gate(catalog, tmp_path, draw_set="sample", process_count=1)

    items = read_items(tmp_path)
    key = json.loads((tmp_path / "key.json").read_text())
    assert [item["id"] for item in items] == list(key) == [f"S{n:04d}" for n in range(1, 8)]
    assert draw.items_by_set == {"labelled": 4, "enrichment": 3}
    for item in items:
        assert set(item) == {"id", "text"}
        assert item["text"] == f"We study attention in network models. ⟦{MIDDLE}⟧"
    assert {entry["stratum"] for entry in key.values() if entry["set"] == "enrichment"} == {
        "no_finite_verb",
        "list_like",
        "short_repository_only",
    }
    assert {entry["word"] for entry in key.values()} <= {"attention", SEALED}
    labelled = [entry for entry in key.values() if entry["set"] == "labelled"]
    assert sorted(entry["decade"] for entry in labelled) == [1990, 1990, 2000, 2000]
    assert set(labelled[0]) == {"set", "work_id", "part", "decade", "word", "content_key"}


@pytest.mark.usefixtures("small_sizes")
def test_the_samples_counts_hold_decades_words_and_strata(
    catalog: CatalogConfig, tmp_path: Path
) -> None:
    write_corpus(catalog)

    draw_junk_gate(catalog, tmp_path, draw_set="sample", process_count=1)

    counts = json.loads((tmp_path / "counts.json").read_text())
    assert counts["drawn_usages_by_decade"] == {"1990": 7, "2000": 5}
    assert counts["records_by_decade"] == {"1990": 6, "2000": 4}
    assert counts["usages_by_word_and_decade"][SEALED] == {
        "1990": {"matched": 1, "in_scope": 1, "drawn": 1},
        "2000": {"matched": 1, "in_scope": 1, "drawn": 1},
    }
    assert counts["enrichment_strata"] == {
        "no_finite_verb": 2,
        "list_like": 2,
        "short_repository_only": 2,
    }
    assert counts["cells_with_at_least_100_drawn"] == 0


@pytest.mark.usefixtures("small_sizes")
def test_adding_a_record_to_the_corpus_changes_no_other_records_selection(
    catalog: CatalogConfig, tmp_path: Path
) -> None:
    write_corpus(catalog)
    draw_junk_gate(catalog, tmp_path / "before", draw_set="sample", process_count=1)
    write_part(catalog, 2015, [{"number": 11}])

    draw_junk_gate(catalog, tmp_path / "after", draw_set="sample", process_count=1)

    def selection(folder: Path) -> dict[tuple[str, str, str], tuple[str, str | None]]:
        key = json.loads((folder / "key.json").read_text())
        return {
            (entry["work_id"], entry["word"], entry["content_key"]): (
                entry["set"],
                entry.get("stratum"),
            )
            for entry in key.values()
        }

    before, after = selection(tmp_path / "before"), selection(tmp_path / "after")
    added = {usage: chosen for usage, chosen in after.items() if usage[0] == work_id(11)}
    assert list(added.values()) == [("labelled", None)]
    assert {usage: chosen for usage, chosen in after.items() if usage not in added} == before


def test_run3s_records_are_drawn_under_run3s_ids(catalog: CatalogConfig, tmp_path: Path) -> None:
    abstract = build_abstract(1, 1995)
    save_run3_records(
        catalog,
        pl.DataFrame(
            [{key: abstract[key] for key in RUN3_RECORDS_COLUMNS}], schema=RUN3_RECORDS_COLUMNS
        ),
    )
    verdicts = [
        {"id": "b000", "part": "sentence_breaks", "work_id": work_id(2)},
        {"id": "r000", "part": "record", "work_id": work_id(1), "decade": 1990}
        | {"field": "ai", "verdict": "abstract"},
    ]
    catalog.run3_records.path.with_name("verdicts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in verdicts)
    )

    draw_junk_gate(catalog, tmp_path, draw_set="run3", process_count=1)

    assert read_items(tmp_path) == [
        {"id": "r000", "text": f"We study attention in network models. {MIDDLE}"}
    ]
    assert json.loads((tmp_path / "key.json").read_text()) == {
        "r000": {"set": "run3", "work_id": work_id(1), "decade": 1990, "field": "ai"}
    }
    assert json.loads((tmp_path / "counts.json").read_text()) == {"items": 1}


def test_the_confirmation_reuses_the_labels_already_made(
    catalog: CatalogConfig, tmp_path: Path
) -> None:
    write_corpus(catalog)
    catalog.junk_cells.path.parent.mkdir(parents=True)
    pl.DataFrame(
        {"word": ["attention", "attention"], "decade": [1990, 2000], "flagged": [True, False]}
    ).write_parquet(catalog.junk_cells.path)
    labelled_usage = pl.read_parquet(catalog.usages.path / "ai_1990.parquet").row(0, named=True)
    label = JunkLabel(
        set="labelled",
        stratum=None,
        work_id=labelled_usage["work_id"],
        part="ai_1990",
        decade=1990,
        word="attention",
        content_key=labelled_usage["content_key"],
        verdict="abstract",
        kind="none",
        sentence="own",
        damaged="no",
    )
    save_junk_labels(catalog, JunkLabels(codebook_sha256="a" * 64, items={"S0003": label}))

    draw = draw_junk_gate(catalog, tmp_path, draw_set="confirmation", process_count=1)

    key = json.loads((tmp_path / "key.json").read_text())
    [cell] = key["cells"]
    assert (cell["word"], cell["decade"]) == ("attention", 1990)
    assert len(cell["items"]) == 6
    assert "S0003" in cell["items"]
    assert (
        sorted(key["items"])
        == [f"C{n:04d}" for n in range(1, 6)]
        == sorted(item["id"] for item in read_items(tmp_path))
    )
    assert draw.items_by_set == {"confirmation": 5}
    counts = json.loads((tmp_path / "counts.json").read_text())
    assert counts == {"items": 5, "flagged_cells": 1, "usages": 6, "already_labelled": 1}
