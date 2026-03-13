"""Semantic Scholar Open Research Corpus (S2ORC) source."""

from collections.abc import Iterator
from pathlib import Path

from interpretant.corpus.base import CorpusSource


class S2ORCSource(CorpusSource):
    """Loads documents from a local S2ORC snapshot.

    The S2ORC dataset is available from https://allenai.org/data/s2orc.
    Download the metadata and full-text shards into ``raw_dir`` before use.
    """

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir

    def iter_documents(self) -> Iterator[str]:
        raise NotImplementedError(
            "S2ORC loading is not yet implemented. "
            "Download the corpus and implement shard parsing here."
        )

    def iter_decade_slices(
        self,
        start: int,
        end: int,
        step: int,
        min_tokens: int = 0,
        workers: int = 1,
    ) -> Iterator[tuple[int, list[str]]]:
        raise NotImplementedError

    def document_count(self) -> int:
        raise NotImplementedError
