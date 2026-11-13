"""The curated inputs the author reviews, read and validated for the pipelines."""

from __future__ import annotations

import re
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from interpretant.catalog.config import CatalogConfig

Scope = Literal["tokens", "sentence", "abstract"]


class CleaningPattern(BaseModel):
    """One spaCy SpanRuler token pattern and the label its matches carry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    pattern: list[dict[str, Any]] = Field(min_length=1)
    at_start: bool = False


class CleaningPatterns(BaseModel):
    """The cleaning patterns file: mojibake signatures, what each label removes, the patterns."""

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


def load_cleaning_patterns(catalog: CatalogConfig) -> CleaningPatterns:
    """The cleaning patterns file, validated.

    Raises:
        FileNotFoundError: if the file does not exist.
        pydantic.ValidationError: if a pattern uses an undeclared label, a scope is unknown or a
            signature is not a valid regular expression.
    """
    text = catalog.cleaning_patterns.path.read_text(encoding="utf-8")
    return CleaningPatterns.model_validate(yaml.safe_load(text))
