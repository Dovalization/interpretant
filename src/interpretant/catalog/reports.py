"""Reports for the author: Markdown documents the pipelines write whole each time."""

from __future__ import annotations

from pathlib import Path

from interpretant.catalog.config import CatalogConfig


def save_corpus_report(catalog: CatalogConfig, text: str) -> Path:
    """Write the corpus report, replacing the last one; the file is written whole or not at all."""
    target = catalog.corpus_report.path
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(f"{target.suffix}.partial")
    staging.write_text(text, encoding="utf-8")
    staging.replace(target)
    return target
