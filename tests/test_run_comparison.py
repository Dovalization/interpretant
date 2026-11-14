"""The comparison of two builds of the abstracts table, and its stage."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from interpretant.catalog import CatalogConfig, save_abstracts
from interpretant.pipelines.compare_corpus import compare_corpus
from interpretant.review.run_comparison import compare_runs, render_comparison
from tests.abstracts_tables import abstract_row, abstracts_table

GERMAN = "Die Aufmerksamkeit ist für die Übersetzung von großer Bedeutung."
LINK = "See https://example.org now."


def previous_build() -> pl.DataFrame:
    return abstracts_table(
        [
            abstract_row(
                1,
                ["Models learn senses.", LINK, GERMAN],
                removals=[{"start": 25, "end": 44, "reason": "url"}],
            ),
            abstract_row(2, ["Short."], dropped_reason="too_short"),
            abstract_row(3, ["Attention helps models."]),
        ]
    )


def current_build() -> pl.DataFrame:
    row = abstract_row(1, ["Models learn senses.", LINK, GERMAN])
    row["sentences"][2]["language"] = "de"
    return abstracts_table(
        [
            row,
            abstract_row(2, ["Short."]),
            abstract_row(3, ["Attention helps models."], dropped_reason="not_english"),
            abstract_row(4, ["A new work."]),
        ]
    )


def test_compare_runs_counts_every_change_of_outcome() -> None:
    outcomes = compare_runs(previous_build(), current_build()).outcomes
    assert set(outcomes.rows()) == {
        ("kept", "kept", 1),
        ("too_short", "kept", 1),
        ("kept", "not_english", 1),
        ("absent", "kept", 1),
    }


def test_compare_runs_samples_each_changed_outcome() -> None:
    samples = compare_runs(previous_build(), current_build()).outcome_samples
    headings = {group.heading for group in samples}
    assert "too_short → kept (1 works)" in headings
    (not_english,) = [group for group in samples if group.heading.startswith("kept → not_english")]
    assert not_english.lines == ["`https://openalex.org/W3` 2010 ai: Attention helps models."]


def test_compare_runs_matches_removals_and_languages_by_their_text() -> None:
    comparison = compare_runs(previous_build(), current_build())
    assert comparison.removals.rows() == [("url", 1, 0, 0, 1)]
    assert comparison.languages.rows() == [("en", "de", 1)]
    assert comparison.sentences["count"].to_list() == [0, 0]


def test_render_comparison_writes_tables_then_samples() -> None:
    text = render_comparison(compare_runs(previous_build(), current_build()), title="Run 3")
    assert text.startswith("# Run 3\n")
    assert "## Outcomes" in text
    assert "### too_short → kept (1 works)" in text
    assert "- `https://openalex.org/W2` 2010 ai: Short." in text


def test_compare_corpus_writes_the_comparison(catalog: CatalogConfig, tmp_path: Path) -> None:
    save_abstracts(catalog, current_build(), part="ai_2010")
    previous = tmp_path / "abstracts_previous"
    previous.mkdir()
    previous_build().write_parquet(previous / "ai_2010.parquet")
    written = compare_corpus(catalog, previous)
    assert written == catalog.corpus_comparison.path
    assert "kept → not_english" in written.read_text()
