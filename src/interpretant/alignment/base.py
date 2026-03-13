"""Abstract base class for temporal alignment of embedding spaces."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class Aligner(ABC):
    """Base interface for aligning embedding spaces across decades."""

    @abstractmethod
    def fit(self, model_paths: dict[int, Path]) -> None:
        """Learn alignment transforms from a mapping of decade → model path.

        Args:
            model_paths: Dict mapping decade integer to saved model path.
        """

    @abstractmethod
    def align(self, output_dir: Path) -> dict[int, Path]:
        """Write aligned models to ``output_dir`` and return decade → path mapping."""

    @abstractmethod
    def get_vector(self, word: str, decade: int) -> np.ndarray:
        """Return the aligned vector for ``word`` at ``decade``."""

    @abstractmethod
    def shared_vocabulary(self) -> list[str]:
        """Return words present in every decade's model."""

    @abstractmethod
    def decades(self) -> list[int]:
        """Return sorted list of decades covered by this aligner."""

    @abstractmethod
    def load_aligned(self, model_dir: Path) -> None:
        """Load previously aligned models from ``model_dir``."""

    def get_vectors(self, word: str) -> dict[int, Any]:
        """Convenience: return {decade: vector} for all decades."""
        return {decade: self.get_vector(word, decade) for decade in self.decades()}
