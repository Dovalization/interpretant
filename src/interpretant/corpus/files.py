"""Cell text files of one pipeline stage: find them, and rewrite them into the next stage.

Every stage (``data/processed``, ``data/lemmatised``, ``data/phrased``) keeps the same layout,
``<stage>/<source>/texts/<cell>.txt`` with one document per line, so a stage is produced by
rewriting the previous one line by line under the same relative paths.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from tqdm import tqdm


def source_texts(stage_dir: Path, sources: Iterable[str]) -> list[Path]:
    """Return every cell file of the given sources under ``stage_dir``, sorted."""
    return sorted(path for source in sources for path in stage_dir.glob(f"{source}/texts/*.txt"))


def rewrite_texts(
    paths: Iterable[Path], stage_dir: Path, output_dir: Path, transform: Callable[[str], str]
) -> list[Path]:
    """Write every line of each file through ``transform`` to the same path under
    ``output_dir``; return the files written."""
    written = []
    for path in tqdm(list(paths), unit="file"):
        target = output_dir / path.relative_to(stage_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        with path.open(encoding="utf-8") as source, target.open("w", encoding="utf-8") as out:
            for line in source:
                out.write(transform(line) + "\n")
        written.append(target)
    return written
