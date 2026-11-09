"""Tests for alignment utilities."""

import numpy as np
import pytest
from gensim.models import KeyedVectors

from interpretant.alignment.procrustes import ProcrustesAligner
from interpretant.alignment.twec import TWECAligner


def _make_keyed_vectors(vocab: list[str], dim: int = 4, seed: int = 0) -> KeyedVectors:
    rng = np.random.default_rng(seed)
    kv = KeyedVectors(dim)
    vectors = rng.standard_normal((len(vocab), dim)).astype(np.float32)
    kv.add_vectors(vocab, vectors)
    return kv


class TestProcrustesAligner:
    def test_instantiation(self) -> None:
        aligner = ProcrustesAligner(reference_decade=2010)
        assert aligner.reference_decade == 2010

    def test_default_reference_decade(self) -> None:
        aligner = ProcrustesAligner()
        assert aligner.reference_decade == 2010

    def test_empty_aligner_has_no_decades(self) -> None:
        aligner = ProcrustesAligner()
        assert aligner.decades() == []

    def test_empty_aligner_has_empty_shared_vocab(self) -> None:
        aligner = ProcrustesAligner()
        assert aligner.shared_vocabulary() == []

    def test_fit_raises_without_models(self, tmp_path: object) -> None:
        """fit() with no real model files should raise an error."""
        aligner = ProcrustesAligner()
        with pytest.raises(ValueError):
            aligner.fit({})

    def test_vocabulary_for_pair(self) -> None:
        """vocabulary_for_pair returns only words present in both decades."""
        aligner = ProcrustesAligner()
        aligner._keyed_vectors[2000] = _make_keyed_vectors(["cat", "dog", "fish"])
        aligner._keyed_vectors[2010] = _make_keyed_vectors(["dog", "fish", "bird"])

        result = aligner.vocabulary_for_pair(2000, 2010)

        assert sorted(result) == ["dog", "fish"]

    def test_word_freqs_returns_none_when_no_expandos(self) -> None:
        aligner = ProcrustesAligner()
        aligner._keyed_vectors[2000] = _make_keyed_vectors(["cat", "dog"])
        assert aligner.word_freqs(2000) is None

    def test_vectors_for_decade(self) -> None:
        """vectors_for_decade returns all words with correct vector shapes."""
        vocab = ["cat", "dog", "fish"]
        kv = _make_keyed_vectors(vocab, dim=4)
        aligner = ProcrustesAligner()
        aligner._keyed_vectors[2000] = kv

        result = aligner.vectors_for_decade(2000)

        assert set(result.keys()) == set(vocab)
        for word in vocab:
            assert result[word].shape == (4,)
            np.testing.assert_array_equal(result[word], kv[word])


class TestTWECAligner:
    def test_fit_raises_not_implemented(self) -> None:
        aligner = TWECAligner()
        with pytest.raises(NotImplementedError):
            aligner.fit({})

    def test_align_raises_not_implemented(self, tmp_path: object) -> None:
        aligner = TWECAligner()
        with pytest.raises(NotImplementedError):
            import pathlib

            aligner.align(pathlib.Path(str(tmp_path)))
