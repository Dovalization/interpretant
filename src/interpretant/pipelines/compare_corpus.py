"""The comparison of this build with an earlier one, for the review of each run (4b).

Not a DVC stage: an earlier build is not one of this build's inputs. Run it after each build,
before its record is committed: by default it compares with the build recorded at ``HEAD``, read
from the DVC cache, so the earlier build needs no copy set aside.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import polars as pl

from interpretant.catalog import (
    CatalogConfig,
    load_abstracts,
    load_abstracts_at,
    load_decision_tables_at,
    load_glued_words,
    load_lower_glued_words,
    load_split_words,
    save_corpus_comparison,
)
from interpretant.pipelines.progress import SILENT, Progress
from interpretant.review.run_comparison import compare_runs, render_comparison

_CURRENT_DECISIONS: dict[str, Callable[[CatalogConfig], pl.DataFrame]] = {
    "glued_words": load_glued_words,
    "split_words": load_split_words,
    "lower_glued_words": load_lower_glued_words,
}


def compare_corpus(
    catalog: CatalogConfig,
    repository: Path,
    revision: str = "HEAD",
    *,
    progress: Progress = SILENT,
) -> Path:
    """Write the comparison of this build with the one recorded at ``revision``.

    ``repository`` is the project's root, the folder of the catalog file. The decision tables are
    compared when both builds have them.

    Raises:
        FileNotFoundError: if either build has no part of the abstracts table.
    """
    progress.stage("loading both builds")
    current = load_abstracts(catalog).collect()
    earlier = load_abstracts_at(catalog, repository, revision)
    earlier_decisions = load_decision_tables_at(catalog, repository, revision)
    current_decisions = {
        name: load(catalog)
        for name, load in _CURRENT_DECISIONS.items()
        if getattr(catalog, name).path.exists()
    }
    progress.stage("comparing the builds")
    comparison = compare_runs(
        earlier,
        current,
        previous_decisions=earlier_decisions,
        current_decisions=current_decisions,
    )
    title = f"What changed since the build recorded at {revision}"
    return save_corpus_comparison(catalog, render_comparison(comparison, title=title))
