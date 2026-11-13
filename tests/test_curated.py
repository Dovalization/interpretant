"""The curated inputs: the cleaning rules and patterns and their validation."""

from __future__ import annotations

from typing import Any

import pydantic
import pytest

from interpretant.catalog import CleaningPatterns, load_cleaning_patterns, read_catalog


def cleaning_patterns(**changes: Any) -> dict[str, Any]:
    content: dict[str, Any] = {
        "mojibake_signatures": ["â€"],
        "labels": {"url": "tokens", "abstract_label": "tokens", "footer": "sentence"},
        "patterns": [
            {"label": "url", "pattern": [{"LIKE_URL": True}]},
            {"label": "abstract_label", "pattern": [{"LOWER": "abstract"}], "id": "start"},
            {"label": "footer", "pattern": [{"LOWER": "click"}, {"LOWER": "here"}]},
        ],
    }
    content.update(changes)
    return content


def test_cleaning_patterns_accept_a_well_formed_set() -> None:
    patterns = CleaningPatterns.model_validate(cleaning_patterns())
    assert [pattern.id for pattern in patterns.patterns] == [None, "start", None]


def test_span_ruler_patterns_leave_out_unset_ids() -> None:
    ruler_patterns = CleaningPatterns.model_validate(cleaning_patterns()).span_ruler_patterns()
    assert ruler_patterns[0] == {"label": "url", "pattern": [{"LIKE_URL": True}]}
    assert ruler_patterns[1]["id"] == "start"


@pytest.mark.parametrize(
    "changes",
    [
        {"patterns": [{"label": "spam", "pattern": [{"LOWER": "dumps"}]}]},
        {"labels": {"url": "paragraph"}},
        {"mojibake_signatures": ["Ã["]},
        {"patterns": [{"label": "url", "pattern": []}]},
        {"patterns": [{"label": "url", "pattern": [{"LIKE_URL": True}], "id": "end"}]},
        {"extra_section": []},
    ],
    ids=[
        "undeclared label",
        "unknown scope",
        "invalid signature",
        "empty pattern",
        "unknown pattern id",
        "unknown key",
    ],
)
def test_cleaning_patterns_reject_a_malformed_set(changes: dict[str, Any]) -> None:
    with pytest.raises(pydantic.ValidationError):
        CleaningPatterns.model_validate(cleaning_patterns(**changes))


def test_the_repository_cleaning_rules_and_patterns_are_valid() -> None:
    patterns = load_cleaning_patterns(read_catalog())
    assert {"tokens", "sentence", "abstract"} == set(patterns.labels.values())
    assert any(pattern.id == "start" for pattern in patterns.patterns)
