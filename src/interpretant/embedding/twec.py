"""Temporal Word Embeddings with a Compass (TWEC).

Trains a single "compass" model on all decades combined, then trains each
decade model initialised from the compass with context vectors (syn1neg)
frozen after every epoch. Because all decade models share the same context
matrix, their target vectors are already in the same coordinate system —
no post-hoc Procrustes alignment required.

Reference: Di Carlo et al. (2019) "Training Temporal Word Embeddings with
a Compass". https://arxiv.org/abs/1906.02376
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from gensim.models import KeyedVectors, Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

logger = logging.getLogger(__name__)


class _EpochProgressCallback(CallbackAny2Vec):
    """Calls bar.update(1) after every training epoch.

    Designed to work with any object that has an ``update(int)`` method,
    so callers can pass a tqdm bar without creating a hard tqdm dependency here.
    """

    def __init__(self, bar: Any) -> None:
        self._bar = bar

    def on_epoch_end(self, model: Word2Vec) -> None:
        self._bar.update(1)


class _FileSentences:
    """Re-iterable sentence source backed by one or more text files.

    Each file is expected to have one tokenised document per line, with tokens
    separated by whitespace — the format written by ``corpus preprocess``.
    gensim calls ``__iter__`` once for vocabulary building and once per training
    epoch, so the class must support multiple iterations without consuming the
    underlying data.

    Args:
        paths: Text files to read from.
        max_docs: If set, stride-sample at most this many lines per file using
            a uniform stride (every Nth line), so the sample covers the full
            temporal range of the file rather than just the first N lines.
    """

    def __init__(self, paths: list[Path], max_docs: int | None = None) -> None:
        self._paths = paths
        self._max_docs = max_docs
        # Pre-compute stride per file: one line-count pass at construction time.
        # stride=1 means take every line; stride=k means take every k-th line.
        self._strides: dict[Path, int] = {}
        if max_docs is not None:
            for path in paths:
                with path.open(encoding="utf-8") as fh:
                    total = sum(1 for _ in fh)
                self._strides[path] = max(1, total // max_docs)

    def __iter__(self) -> Iterator[list[str]]:
        for path in self._paths:
            stride = self._strides.get(path, 1)
            count = 0
            with path.open(encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if i % stride != 0:
                        continue
                    tokens = line.split()
                    if tokens:
                        yield tokens
                        count += 1
                        if self._max_docs is not None and count >= self._max_docs:
                            break


class _FreezeContextCallback(CallbackAny2Vec):
    """Restores compass context vectors (syn1neg) after every training epoch.

    This implements the TWEC "frozen compass" constraint without requiring a
    custom gensim fork: within each epoch the context vectors drift slightly,
    but they are reset at the epoch boundary so the net effect is that the
    decade model's context matrix remains equal to the compass.
    """

    def __init__(self, compass: Word2Vec, decade_model: Word2Vec) -> None:
        # Pre-build index mapping: decade_model word index → compass word index
        # Only for words shared between the two models.
        self._pairs: list[tuple[int, int]] = []
        for word, d_idx in decade_model.wv.key_to_index.items():
            c_idx = compass.wv.key_to_index.get(word)
            if c_idx is not None:
                self._pairs.append((d_idx, c_idx))
        # Snapshot of compass syn1neg at construction time.
        self._compass_syn1neg = compass.syn1neg.copy()

    def on_epoch_end(self, model: Word2Vec) -> None:
        for d_idx, c_idx in self._pairs:
            model.syn1neg[d_idx] = self._compass_syn1neg[c_idx]


class TWECTrainer:
    """Trains temporally aligned word embeddings using the compass method.

    Usage::

        trainer = TWECTrainer(vector_size=300, window=10, min_count=5, epochs=10)
        trainer.train_compass([Path("data/processed/books/texts/books_1960.txt")])
        kv = trainer.train_decade([Path("data/processed/books/texts/books_1960.txt")], decade=1960)
        kv.save("models/twec/1960.kv")

    The compass must be trained before any decade model.
    All training methods accept lists of file paths and stream from disk so
    that arbitrarily large corpora can be processed without loading everything
    into RAM.
    """

    def __init__(
        self,
        vector_size: int = 300,
        window: int = 10,
        min_count: int = 5,
        workers: int = 4,
        epochs: int = 10,
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
        self._compass: Word2Vec | None = None

    # ------------------------------------------------------------------
    # Compass
    # ------------------------------------------------------------------

    def train_compass(
        self,
        corpus_paths: list[Path],
        max_docs: int | None = None,
        extra_callbacks: list[Any] | None = None,
    ) -> None:
        """Train the compass on all decades' text, streaming from disk.

        The compass establishes the universal coordinate system. It must be
        called before training any decade model.

        Args:
            corpus_paths: Paths to preprocessed text files (one doc per line).
                All files are read in order on each pass; gensim makes multiple
                passes for vocabulary building and per-epoch training.
            max_docs: If set, cap the number of lines read per file per pass.
            extra_callbacks: Additional ``CallbackAny2Vec`` instances (e.g. a
                tqdm progress bar callback) to pass to gensim's train loop.
        """
        sentences = _FileSentences(corpus_paths, max_docs=max_docs)
        logger.info("Training compass on %d file(s)", len(corpus_paths))
        self._compass = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            sg=self.sg,
            negative=self.negative,
            seed=self.seed,
            callbacks=extra_callbacks or [],
        )

    def save_compass(self, output_dir: Path) -> Path:
        """Save the compass model and return its path."""
        if self._compass is None:
            raise RuntimeError("No compass trained yet. Call train_compass() first.")
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "compass.model"
        self._compass.save(str(path))
        return path

    def load_compass(self, path: Path) -> None:
        """Load a previously saved compass model."""
        self._compass = Word2Vec.load(str(path))

    # ------------------------------------------------------------------
    # Decade models
    # ------------------------------------------------------------------

    def train_decade(
        self,
        decade_paths: list[Path],
        decade: int,
        max_docs: int | None = None,
        extra_callbacks: list[Any] | None = None,
    ) -> KeyedVectors:
        """Train a decade model with compass context vectors frozen.

        Args:
            decade_paths: Paths to preprocessed text files for this decade.
            decade: The decade label (used only for logging).
            max_docs: If set, cap the number of lines read per file per pass.
            extra_callbacks: Additional ``CallbackAny2Vec`` instances passed to
                gensim's train loop alongside the freeze-context callback.

        Returns:
            KeyedVectors with target vectors trained on this decade's text
            while context vectors were held fixed to the compass values.
        """
        if self._compass is None:
            raise RuntimeError("No compass available. Call train_compass() first.")

        compass = self._compass
        sentences = _FileSentences(decade_paths, max_docs=max_docs)

        # Build the decade model's vocabulary from the decade sentences.
        # min_count=1 so all words that appear at least once are included;
        # the compass already established the global vocabulary threshold.
        model = Word2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=1,
            workers=self.workers,
            epochs=self.epochs,
            sg=self.sg,
            negative=self.negative,
            seed=self.seed,
        )
        model.build_vocab(sentences)

        # Initialise target and context vectors from compass for shared words.
        # Words absent from the compass start from gensim's random init.
        self._init_from_compass(model, compass)

        # Train with frozen context callback (plus any caller-supplied callbacks).
        freeze_cb = _FreezeContextCallback(compass, model)
        all_callbacks: list[Any] = [freeze_cb] + (extra_callbacks or [])
        model.train(
            sentences,
            total_examples=model.corpus_count,
            epochs=model.epochs,
            callbacks=all_callbacks,
        )

        logger.info(
            "Decade %d: vocab=%d  shared_with_compass=%d",
            decade,
            len(model.wv),
            len(freeze_cb._pairs),
        )

        return model.wv

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_from_compass(self, model: Word2Vec, compass: Word2Vec) -> None:
        """Copy wv and syn1neg from compass into model for shared words."""
        for word, d_idx in model.wv.key_to_index.items():
            c_idx = compass.wv.key_to_index.get(word)
            if c_idx is None:
                continue
            model.wv.vectors[d_idx] = compass.wv.vectors[c_idx]
            model.syn1neg[d_idx] = compass.syn1neg[c_idx]

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def save_decade_kv(kv: KeyedVectors, output_dir: Path, decade: int) -> Path:
        """Save a decade's KeyedVectors and return the path."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{decade}.kv"
        kv.save(str(path))
        return path

    @property
    def compass_vocab_size(self) -> int:
        """Number of words in the compass vocabulary."""
        if self._compass is None:
            raise RuntimeError("No compass trained yet.")
        return len(self._compass.wv)

    @property
    def compass(self) -> Word2Vec:
        """The trained compass model."""
        if self._compass is None:
            raise RuntimeError("No compass trained yet.")
        return self._compass
