"""The curated inputs: the cleaning rules and patterns and their validation."""

from __future__ import annotations

from typing import Any

import pydantic
import pytest
from spacy.lang.lex_attrs import _tlds

from interpretant.catalog import (
    CatalogConfig,
    CleaningPatterns,
    load_cleaning_patterns,
    load_doi_corrections,
    read_catalog,
)

# The url pattern for a last dotted part from spaCy's list of domain endings, around the list.
DOMAIN_PREFIX = r"\.(?:"
DOMAIN_SUFFIX = r")(?::[^.]*)?$"


def cleaning_patterns(**changes: Any) -> dict[str, Any]:
    content: dict[str, Any] = {
        "mojibake_signatures": ["â€"],
        "labels": {"url": "tokens", "abstract_label": "tokens", "footer": "rest_of_sentence"},
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
    assert {"tokens", "rest_of_sentence", "abstract"} == set(patterns.labels.values())
    assert any(pattern.id == "start" for pattern in patterns.patterns)


def test_the_url_domain_endings_are_spacys() -> None:
    patterns = load_cleaning_patterns(read_catalog()).patterns
    regexes = [
        token["TEXT"]["REGEX"]
        for pattern in patterns
        if pattern.label == "url"
        for token in pattern.pattern
        if "TEXT" in token
    ]
    (domain,) = [regex for regex in regexes if regex.startswith(DOMAIN_PREFIX)]
    copied = domain.removeprefix(DOMAIN_PREFIX).removesuffix(DOMAIN_SUFFIX).split("|")
    assert sorted(copied) == sorted(_tlds)


def test_load_doi_corrections_reads_the_repository_file_by_work() -> None:
    corrections = load_doi_corrections(read_catalog())
    typo = corrections["https://openalex.org/W2410668642"]
    assert typo.openalex_doi == "https://doi.org/10.153/scnp00400195"
    assert typo.doi == "https://doi.org/10.1053/scnp00400195"
    assert corrections["https://openalex.org/W15233091"].doi is None


@pytest.mark.parametrize(
    ("doi", "work_ids", "message"),
    [
        ("https://doi.org/10.153/x", ["W1"], "String should match pattern"),
        ("https://doi.org/10.1053/x", ["W1", "W1"], "corrected more than once"),
    ],
    ids=["a corrected DOI without Crossref's form", "a work corrected twice"],
)
def test_load_doi_corrections_refuses_bad_entries(
    catalog: CatalogConfig, doi: str, work_ids: list[str], message: str
) -> None:
    entries = "".join(
        f"  - work_id: https://openalex.org/{work}\n    openalex_doi: x\n"
        f"    doi: {doi}\n    evidence: seen\n"
        for work in work_ids
    )
    catalog.doi_corrections.path.parent.mkdir(parents=True, exist_ok=True)
    catalog.doi_corrections.path.write_text("corrections:\n" + entries)
    with pytest.raises(pydantic.ValidationError, match=message):
        load_doi_corrections(catalog)
