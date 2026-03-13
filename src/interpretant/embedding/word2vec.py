"""Word2Vec embedding trainer backed by gensim."""

from pathlib import Path
from typing import Any

import numpy as np
from gensim.models import Word2Vec

from interpretant.embedding.base import EmbeddingTrainer


class Word2VecTrainer(EmbeddingTrainer):
    """Trains skip-gram Word2Vec models using gensim."""

    def __init__(
        self,
        vector_size: int = 300,
        window: int = 10,
        min_count: int = 10,
        workers: int = 4,
        epochs: int = 5,
        sg: int = 1,
        negative: int = 10,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.sg = sg
        self.negative = negative
        self.seed = seed
        self._model: Word2Vec | None = None

    def train(self, sentences: list[list[str]], decade: int) -> None:
        """Train a Word2Vec model on ``sentences`` for ``decade``."""
        self._model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            sg=self.sg,
            negative=self.negative,
            seed=self.seed,
        )

    def save(self, output_dir: Path, decade: int) -> Path:
        """Save the trained model and return the path."""
        if self._model is None:
            raise RuntimeError("No model trained yet. Call train() first.")
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{decade}.model"
        self._model.save(str(path))
        return path

    def load(self, model_path: Path) -> None:
        """Load a Word2Vec model from disk."""
        self._model = Word2Vec.load(str(model_path))

    def get_vector(self, word: str) -> np.ndarray:
        """Return the embedding vector for ``word``."""
        if self._model is None:
            raise RuntimeError("No model loaded.")
        return self._model.wv[word]  # type: ignore[no-any-return]

    def vocabulary(self) -> list[str]:
        """Return the full model vocabulary."""
        if self._model is None:
            raise RuntimeError("No model loaded.")
        return list(self._model.wv.key_to_index.keys())
