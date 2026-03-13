"""Tests for embedding trainer instantiation."""

import pytest

from interpretant.embedding.fasttext import FastTextTrainer
from interpretant.embedding.word2vec import Word2VecTrainer


class TestWord2VecTrainer:
    def test_instantiation_with_defaults(self) -> None:
        trainer = Word2VecTrainer()
        assert trainer.vector_size == 300
        assert trainer.window == 10
        assert trainer.min_count == 10

    def test_instantiation_with_custom_params(self) -> None:
        trainer = Word2VecTrainer(vector_size=100, window=5, min_count=2)
        assert trainer.vector_size == 100
        assert trainer.window == 5
        assert trainer.min_count == 2

    def test_no_model_before_training(self) -> None:
        trainer = Word2VecTrainer()
        assert trainer._model is None

    def test_vocabulary_raises_before_training(self) -> None:
        trainer = Word2VecTrainer()
        with pytest.raises(RuntimeError):
            trainer.vocabulary()

    def test_get_vector_raises_before_training(self) -> None:
        trainer = Word2VecTrainer()
        with pytest.raises(RuntimeError):
            trainer.get_vector("word")


class TestFastTextTrainer:
    def test_instantiation_with_defaults(self) -> None:
        trainer = FastTextTrainer()
        assert trainer.vector_size == 300
        assert trainer.min_n == 3
        assert trainer.max_n == 6

    def test_instantiation_with_custom_params(self) -> None:
        trainer = FastTextTrainer(vector_size=50, min_n=2, max_n=4)
        assert trainer.vector_size == 50
        assert trainer.min_n == 2
        assert trainer.max_n == 4

    def test_no_model_before_training(self) -> None:
        trainer = FastTextTrainer()
        assert trainer._model is None

    def test_vocabulary_raises_before_training(self) -> None:
        trainer = FastTextTrainer()
        with pytest.raises(RuntimeError):
            trainer.vocabulary()
