"""Cell text files of one stage of the old text-file pipeline.

Every stage (``data/processed``, ``data/lemmatised``, ``data/phrased``) keeps the same layout,
``<stage>/<source>/texts/<cell>.txt`` with one document per line. The commands that wrote the
stages retired on 2026-11-17; the files stay on disk for the grid and the evidence scripts.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def source_texts(stage_dir: Path, sources: Iterable[str]) -> list[Path]:
    """Return every cell file of the given sources under ``stage_dir``, sorted."""
    return sorted(path for source in sources for path in stage_dir.glob(f"{source}/texts/*.txt"))
