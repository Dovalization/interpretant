"""The curated inputs the author reviews, read and validated for the pipelines."""

from __future__ import annotations

import re
from typing import Any, Literal

import srsly
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from interpretant.catalog.config import CatalogConfig

Scope = Literal["tokens", "rest_of_sentence", "abstract"]
# The pattern id that makes a match count only when it begins the abstract.
START_ID = "start"


class CleaningPattern(BaseModel):
    """One line of a spaCy pattern file: a label, a token pattern, and an optional id."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    pattern: list[dict[str, Any]] = Field(min_length=1)
    id: Literal["start"] | None = None


class CleaningPatterns(BaseModel):
    """The cleaning rules and patterns: mojibake signatures, label scopes, token patterns."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mojibake_signatures: list[str] = Field(min_length=1)
    labels: dict[str, Scope] = Field(min_length=1)
    patterns: list[CleaningPattern] = Field(min_length=1)

    @field_validator("mojibake_signatures")
    @classmethod
    def _signatures_are_regular_expressions(cls, signatures: list[str]) -> list[str]:
        for signature in signatures:
            try:
                re.compile(signature)
            except re.error as exc:
                raise ValueError(
                    f"mojibake signature {signature!r} is not a valid pattern"
                ) from exc
        return signatures

    @model_validator(mode="after")
    def _every_pattern_has_a_declared_label(self) -> CleaningPatterns:
        undeclared = sorted({pattern.label for pattern in self.patterns} - set(self.labels))
        if undeclared:
            raise ValueError(f"patterns use labels with no scope declared: {undeclared}")
        return self

    def span_ruler_patterns(self) -> list[dict[str, Any]]:
        """The patterns as spaCy's SpanRuler takes them, without unset ids."""
        return [pattern.model_dump(exclude_none=True) for pattern in self.patterns]


def load_cleaning_patterns(catalog: CatalogConfig) -> CleaningPatterns:
    """The cleaning rules file and the pattern file, read together and validated.

    Raises:
        FileNotFoundError: if either file does not exist.
        pydantic.ValidationError: if a pattern uses an undeclared label, a scope is unknown or a
            signature is not a valid regular expression.
    """
    rules = yaml.safe_load(catalog.cleaning_rules.path.read_text(encoding="utf-8"))
    patterns = list(srsly.read_jsonl(catalog.cleaning_patterns.path))
    return CleaningPatterns.model_validate({**rules, "patterns": patterns})


class DOICorrection(BaseModel):
    """A work whose DOI OpenAlex gives wrongly: OpenAlex's value, the right one, and why."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    work_id: str = Field(pattern=r"^https://openalex\.org/W\d+$")
    openalex_doi: str
    doi: str | None = Field(pattern=r"^https://doi\.org/10\.\d{4,}(?:\.\d+)*/\S+$")
    evidence: str = Field(min_length=1)


class DOICorrections(BaseModel):
    """The corrections file: one entry per work, each work at most once."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    corrections: list[DOICorrection]

    @model_validator(mode="after")
    def _each_work_once(self) -> DOICorrections:
        works = [correction.work_id for correction in self.corrections]
        repeated = sorted({work for work in works if works.count(work) > 1})
        if repeated:
            raise ValueError(f"works corrected more than once: {repeated}")
        return self


def load_doi_corrections(catalog: CatalogConfig) -> dict[str, DOICorrection]:
    """The DOI corrections by work id.

    Raises:
        FileNotFoundError: if the corrections file does not exist.
        pydantic.ValidationError: if an entry is malformed, gives a DOI without Crossref's form or
            corrects a work twice.
    """
    document = yaml.safe_load(catalog.doi_corrections.path.read_text(encoding="utf-8"))
    corrections = DOICorrections.model_validate(document).corrections
    return {correction.work_id: correction for correction in corrections}
