"""Abstract base class for embedding trainers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class EmbeddingTrainer(ABC):
    """Base interface for training decade-sliced word embeddings."""

    @abstractmethod
    def __init__(self, vector_size: int, window: int, min_count: int, **kwargs: Any) -> None:
        """Initialise with core hyperparameters."""

    @abstractmethod
    def train(self, sentences: list[list[str]], decade: int) -> None:
        """Train an embedding model on tokenised sentences for one decade.

        Args:
            sentences: List of tokenised documents (each a list of strings).
            decade: The decade label (e.g. 1990).
        """

    @abstractmethod
    def save(self, output_dir: Path, decade: int) -> Path:
        """Persist the trained model for ``decade`` and return the saved path."""

    @abstractmethod
    def load(self, model_path: Path) -> None:
        """Load a previously saved model from ``model_path``."""

    @abstractmethod
    def get_vector(self, word: str) -> Any:
        """Return the embedding vector for ``word``."""

    @abstractmethod
    def vocabulary(self) -> list[str]:
        """Return the full vocabulary of the loaded model."""
