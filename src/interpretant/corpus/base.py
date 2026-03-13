"""Abstract base class for corpus sources."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path


class CorpusSource(ABC):
    """Base interface for loading and iterating over a text corpus."""

    @abstractmethod
    def __init__(self, raw_dir: Path) -> None:
        """Initialise with path to raw corpus directory."""

    @abstractmethod
    def iter_documents(self) -> Iterator[str]:
        """Yield raw text documents one at a time."""

    @abstractmethod
    def iter_decade_slices(
        self,
        start: int,
        end: int,
        step: int,
        min_tokens: int = 0,
        workers: int = 1,
    ) -> Iterator[tuple[int, list[str]]]:
        """Yield (decade_start, documents) tuples for each decade in [start, end)."""

    @abstractmethod
    def document_count(self) -> int:
        """Return the total number of available documents."""
