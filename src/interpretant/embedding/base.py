"""Abstract base class for embedding trainers."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# gensim's compiled training loop silently truncates a sentence after this many tokens.
MAX_SENTENCE_TOKENS = 10_000


def training_sentences(line: str) -> Iterator[list[str]]:
    """Tokenise one document line into sentences gensim trains on in full.

    Books are stored one per line (about 120,000 tokens on average); passed whole, gensim
    would train on the first 10,000 tokens of each book and ignore the rest.
    """
    tokens = line.split()
    for start in range(0, len(tokens), MAX_SENTENCE_TOKENS):
        yield tokens[start : start + MAX_SENTENCE_TOKENS]


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
