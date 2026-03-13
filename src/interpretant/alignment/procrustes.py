"""Orthogonal Procrustes alignment for temporal word embeddings."""

from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from scipy.linalg import orthogonal_procrustes

from interpretant.alignment.base import Aligner


class ProcrustesAligner(Aligner):
    """Aligns each decade's embedding space to a reference decade via Procrustes rotation.

    All decade matrices are rotated onto the reference embedding space so that
    vectors are directly comparable across time.
    """

    def __init__(self, reference_decade: int = 2010) -> None:
        self.reference_decade = reference_decade
        self._keyed_vectors: dict[int, KeyedVectors] = {}
        self._shared_vocab: list[str] = []

    def _compute_shared_vocab(self, keyed_vectors: dict[int, KeyedVectors]) -> list[str]:
        """Return sorted vocabulary present in every decade's model."""
        vocab_sets = [set(kv.key_to_index.keys()) for kv in keyed_vectors.values()]
        return sorted(set.intersection(*vocab_sets))

    def _rotate_onto_reference(
        self, source_kv: KeyedVectors, reference_kv: KeyedVectors
    ) -> KeyedVectors:
        """Return a new KeyedVectors with source rotated onto the reference space."""
        source_matrix = np.stack([source_kv[word] for word in self._shared_vocab])
        target_matrix = np.stack([reference_kv[word] for word in self._shared_vocab])
        rotation, _ = orthogonal_procrustes(source_matrix, target_matrix)
        aligned_vectors = source_kv.vectors @ rotation
        aligned_kv = KeyedVectors(source_kv.vector_size)
        aligned_kv.add_vectors(list(source_kv.key_to_index.keys()), aligned_vectors)
        return aligned_kv

    def fit(self, model_paths: dict[int, Path]) -> None:
        """Load models and compute Procrustes rotation matrices.

        Args:
            model_paths: Dict mapping decade → saved gensim model path.
        """
        if not model_paths:
            raise ValueError("model_paths must contain at least one entry.")

        raw: dict[int, KeyedVectors] = {
            decade: Word2Vec.load(str(path)).wv
            for decade, path in model_paths.items()
        }

        self._shared_vocab = self._compute_shared_vocab(raw)
        if not self._shared_vocab:
            raise ValueError("No shared vocabulary found across all decade models.")

        reference_kv = raw[self.reference_decade]

        for decade, kv in raw.items():
            if decade == self.reference_decade:
                self._keyed_vectors[decade] = kv
            else:
                self._keyed_vectors[decade] = self._rotate_onto_reference(kv, reference_kv)

    def align(self, output_dir: Path) -> dict[int, Path]:
        """Save aligned KeyedVectors to disk and return decade → path mapping."""
        if not self._keyed_vectors:
            raise RuntimeError("Call fit() before align().")
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[int, Path] = {}
        for decade, kv in self._keyed_vectors.items():
            path = output_dir / f"{decade}.kv"
            kv.save(str(path))
            paths[decade] = path
        return paths

    def get_vector(self, word: str, decade: int) -> np.ndarray:
        """Return the aligned vector for ``word`` at ``decade``."""
        return self._keyed_vectors[decade][word]  # type: ignore[no-any-return]

    def shared_vocabulary(self) -> list[str]:
        """Return words present in every decade's model."""
        return list(self._shared_vocab)

    def decades(self) -> list[int]:
        """Return sorted list of aligned decades."""
        return sorted(self._keyed_vectors.keys())

    def load_aligned(self, model_dir: Path) -> None:
        """Load previously saved aligned KeyedVectors from ``model_dir``."""
        for path in sorted(model_dir.glob("*.kv")):
            decade = int(path.stem)
            self._keyed_vectors[decade] = KeyedVectors.load(str(path))

        if self._keyed_vectors:
            self._shared_vocab = self._compute_shared_vocab(self._keyed_vectors)
