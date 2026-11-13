"""The curated inputs the author reviews, read and validated for the pipelines."""

from __future__ import annotations

import re
from typing import Any, Literal

import srsly
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from interpretant.catalog.config import CatalogConfig

Scope = Literal["tokens", "sentence", "abstract"]
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
