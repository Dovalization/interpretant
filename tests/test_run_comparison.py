"""The comparison of two builds: abstracts, final text and decisions, and its command."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import polars as pl

from interpretant.catalog import (
    CatalogConfig,
    load_abstracts_at,
    load_decision_tables_at,
    save_abstracts,
)
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


def test_compare_runs_counts_the_works_whose_final_text_differs() -> None:
    comparison = compare_runs(previous_build(), current_build())
    assert comparison.kept_texts.rows() == [(1, 1)]
    (group,) = comparison.kept_text_samples
    assert group.heading == "final text differs (1 works)"
    before, now = group.lines[0].split(" → ")
    assert before == "`https://openalex.org/W1`: …Models learn senses. **See** now. " + GERMAN + "…"
    assert now == "…Models learn senses. **See https://example.org** now. " + GERMAN + "…"


def decisions(glued: list[str], joined: list[tuple[str, str]]) -> dict[str, pl.DataFrame]:
    return {
        "glued_words": pl.DataFrame({"word": glued, "split": [True] * len(glued)}),
        "split_words": pl.DataFrame(
            {
                "word": [word for word, _ in joined],
                "left": [left for _, left in joined],
                "join": [True] * len(joined),
                "join_hyphenated": [False] * len(joined),
            }
        ),
    }


def test_compare_runs_counts_the_decisions_each_table_changed() -> None:
    comparison = compare_runs(
        previous_build(),
        current_build(),
        previous_decisions=decisions(["inthe", "ofthe"], [("information", "infor")]),
        current_decisions=decisions(["andthe", "ofthe"], [("information", "infor")]),
    )
    assert comparison.decisions.rows() == [
        ("glued_words", "split", 2, 2, 1, 1),
        ("split_words", "join", 1, 1, 0, 0),
        ("split_words", "join_hyphenated", 0, 0, 0, 0),
    ]
    headings = [group.heading for group in comparison.decision_samples]
    assert headings == ["glued_words split, new (1)", "glued_words split, no longer (1)"]
    assert comparison.decision_samples[0].lines == ["`andthe`"]


def test_render_comparison_leaves_out_decisions_when_a_build_has_none() -> None:
    text = render_comparison(compare_runs(previous_build(), current_build()), title="Run 3")
    assert "## Final text of works kept in both" in text
    assert "## Word repair decisions" not in text


def record_build(repository: Path) -> None:
    """Commit the repository's current files, the abstracts table tracked by DVC."""
    git = ["git", "-c", "user.name=Test", "-c", "user.email=test@example.org"]
    git += ["-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null"]
    environment = {**os.environ, "DVC_NO_ANALYTICS": "1"}
    for command in (
        ["git", "init", "--quiet"],
        [sys.executable, "-m", "dvc", "init", "--quiet"],
        [sys.executable, "-m", "dvc", "add", "--quiet", "data/abstracts"],
        ["git", "add", "--all"],
        [*git, "commit", "--quiet", "-m", "record a build"],
    ):
        subprocess.run(command, cwd=repository, env=environment, check=True)


def test_compare_corpus_compares_with_the_build_recorded_at_a_revision(
    catalog: CatalogConfig, tmp_path: Path
) -> None:
    save_abstracts(catalog, previous_build(), part="ai_2010")
    record_build(tmp_path)
    save_abstracts(catalog, current_build(), part="ai_2010")
    written = compare_corpus(catalog, tmp_path)
    assert written == catalog.corpus_comparison.path
    text = written.read_text()
    assert text.startswith("# What changed since the build recorded at HEAD\n")
    assert "kept → not_english" in text
    assert "final text differs (1 works)" in text


def test_load_abstracts_at_reads_the_recorded_build_and_no_decision_tables(
    catalog: CatalogConfig, tmp_path: Path
) -> None:
    save_abstracts(catalog, previous_build(), part="ai_2010")
    record_build(tmp_path)
    save_abstracts(catalog, current_build(), part="ai_2010")
    recorded = load_abstracts_at(catalog, tmp_path, "HEAD")
    assert recorded["work_id"].to_list() == previous_build()["work_id"].to_list()
    assert load_decision_tables_at(catalog, tmp_path, "HEAD") == {}
