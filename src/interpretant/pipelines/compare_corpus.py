"""The comparison of the abstracts table with an earlier build, for the review of each run (4b).

Not a DVC stage: an earlier build is not one of this build's inputs. Run it by hand after each
build, with the earlier table set aside in a folder of its own.
"""

from __future__ import annotations

from pathlib import Path

from interpretant.catalog import (
    CatalogConfig,
    load_abstracts,
    load_abstracts_from,
    save_corpus_comparison,
)
from interpretant.pipelines.progress import SILENT, Progress
from interpretant.review.run_comparison import compare_runs, render_comparison


def compare_corpus(catalog: CatalogConfig, previous: Path, *, progress: Progress = SILENT) -> Path:
    """Write the comparison of the current abstracts table with the one in ``previous``.

    Raises:
        FileNotFoundError: if either table has no part.
    """
    progress.stage("loading both abstracts tables")
    current = load_abstracts(catalog).collect()
    earlier = load_abstracts_from(previous).collect()
    progress.stage("comparing the builds")
    comparison = compare_runs(earlier, current)
    title = f"What changed since the build in {previous.name}"
    return save_corpus_comparison(catalog, render_comparison(comparison, title=title))
