"""Tests for cross-perplexity between fields."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from interpretant.drift.perplexity import (
    END,
    UNKNOWN,
    KneserNeyBigram,
    add_excess,
    decade_cross_entropy,
    training_sample,
)

PHILOSOPHY = "the mind perceives the world through embodied experience and intentional acts"
COMPUTING = "the model learns embeddings from data through gradient descent and attention layers"


def test_probabilities_sum_to_one_for_every_context() -> None:
    documents = [PHILOSOPHY, COMPUTING, "the world model learns"]
    vocabulary = {word for document in documents for word in document.split()}
    model = KneserNeyBigram(vocabulary).fit(documents)

    for previous in ["the", "learns", "never_seen"]:
        total = sum(model.probability(word, previous) for word in model.vocabulary)
        assert total == pytest.approx(1.0)


def test_own_domain_text_is_cheaper_than_other_domain_text() -> None:
    vocabulary = set(PHILOSOPHY.split()) | set(COMPUTING.split())
    model = KneserNeyBigram(vocabulary).fit([PHILOSOPHY] * 50)

    assert model.cross_entropy([PHILOSOPHY]) < model.cross_entropy([COMPUTING])


def test_unknown_words_map_to_one_token() -> None:
    model = KneserNeyBigram({"known"}).fit(["known word"])

    assert model._tokens("known mystery") == ["<s>", "known", UNKNOWN, END]


def test_training_sample_stops_at_the_token_budget() -> None:
    pool = ["a b c"] * 10

    assert len(training_sample(pool, 7)) == 3


def test_decade_cross_entropy_scores_every_pair_with_equal_training(tmp_path: Path) -> None:
    for field, sentence in (("philosophy", PHILOSOPHY), ("compsci", COMPUTING)):
        (tmp_path / f"{field}_2010.txt").write_text("\n".join([sentence] * 30_000))

    table = decade_cross_entropy(sorted(tmp_path.glob("*.txt")))
    table.insert(0, "decade", 2010)
    scored = add_excess(table).set_index(["model_field", "text_field"])

    assert len(scored) == 4
    assert scored.training_tokens.nunique() == 1
    assert scored.loc[("philosophy", "philosophy"), "excess"] == 0
    assert scored.loc[("compsci", "philosophy"), "excess"] > 0
    assert isinstance(scored, pd.DataFrame)
