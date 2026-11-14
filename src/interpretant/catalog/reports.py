"""What the pipelines report: the corpus report, its metrics, and the log of runs.

The report is a Markdown document written whole each time. The metrics are a small JSON file
kept in git, which DVC compares between commits (``dvc metrics diff``). The run log is a JSON
Lines file kept in git, one record appended per run: the track record of when each run took
place, from which commit and how long each stage took (decision agenda, step 9).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from interpretant.catalog.config import CatalogConfig


def save_corpus_report(catalog: CatalogConfig, text: str) -> Path:
    """Write the corpus report, replacing the last one; the file is written whole or not at all."""
    return _write_whole(catalog.corpus_report.path, text)


def load_corpus_metrics(catalog: CatalogConfig) -> dict[str, object]:
    """The corpus metrics last written, or no metrics before the first report."""
    target = catalog.corpus_metrics.path
    if not target.exists():
        return {}
    metrics: dict[str, object] = json.loads(target.read_text(encoding="utf-8"))
    return metrics


def save_corpus_comparison(catalog: CatalogConfig, text: str) -> Path:
    """Write the comparison with an earlier build, replacing the last one."""
    return _write_whole(catalog.corpus_comparison.path, text)


def save_corpus_metrics(catalog: CatalogConfig, metrics: Mapping[str, object]) -> Path:
    """Write the corpus metrics as sorted, indented JSON, so their diffs read line by line."""
    text = json.dumps(metrics, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return _write_whole(catalog.corpus_metrics.path, text)


def append_run_record(catalog: CatalogConfig, record: Mapping[str, object]) -> Path:
    """Append one run's record to the run log, as one line of JSON."""
    target = catalog.run_log.path
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as log:
        log.write(json.dumps(record, ensure_ascii=False) + "\n")
    return target


def _write_whole(target: Path, text: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(f"{target.suffix}.partial")
    staging.write_text(text, encoding="utf-8")
    staging.replace(target)
    return target
