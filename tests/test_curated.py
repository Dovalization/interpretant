"""The curated inputs: the cleaning patterns file and its validation."""

from __future__ import annotations

from typing import Any

import pydantic
import pytest

from interpretant.catalog import CleaningPatterns, load_cleaning_patterns, read_catalog


def patterns_file(**changes: Any) -> dict[str, Any]:
    content: dict[str, Any] = {
        "mojibake_signatures": ["â€"],
        "labels": {"url": "tokens", "footer": "sentence"},
        "patterns": [
            {"label": "url", "pattern": [{"LIKE_URL": True}]},
            {"label": "footer", "pattern": [{"LOWER": "click"}, {"LOWER": "here"}]},
        ],
    }
    content.update(changes)
    return content


def test_cleaning_patterns_accept_a_well_formed_file() -> None:
    patterns = CleaningPatterns.model_validate(patterns_file())
    assert [pattern.label for pattern in patterns.patterns] == ["url", "footer"]
    assert patterns.patterns[0].at_start is False


@pytest.mark.parametrize(
    "changes",
    [
        {"patterns": [{"label": "spam", "pattern": [{"LOWER": "dumps"}]}]},
        {"labels": {"url": "paragraph"}},
        {"mojibake_signatures": ["Ã["]},
        {"patterns": [{"label": "url", "pattern": []}]},
        {"extra_section": []},
    ],
    ids=["undeclared label", "unknown scope", "invalid signature", "empty pattern", "unknown key"],
)
def test_cleaning_patterns_reject_a_malformed_file(changes: dict[str, Any]) -> None:
    with pytest.raises(pydantic.ValidationError):
        CleaningPatterns.model_validate(patterns_file(**changes))


def test_the_repository_cleaning_patterns_file_is_valid() -> None:
    patterns = load_cleaning_patterns(read_catalog())
    assert {"tokens", "sentence", "abstract"} == set(patterns.labels.values())
