"""Embedding trainers."""

from interpretant.embedding.base import EmbeddingTrainer
from interpretant.embedding.fasttext import FastTextTrainer
from interpretant.embedding.word2vec import Word2VecTrainer

__all__ = ["EmbeddingTrainer", "Word2VecTrainer", "FastTextTrainer"]
