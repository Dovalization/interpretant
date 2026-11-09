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


def test_remove_shared_drops_abstracts_found_in_more_than_one_cell() -> None:
    from interpretant.drift.perplexity import remove_shared

    cells = {"compsci": ["shared paper", "cs only"], "ai": ["shared paper", "ai only"]}

    assert remove_shared(cells) == {"compsci": ["cs only"], "ai": ["ai only"]}


def test_fixed_budget_trains_everyone_equally_and_drops_small_cells(tmp_path: Path) -> None:
    (tmp_path / "philosophy_1990.txt").write_text("\n".join([PHILOSOPHY] * 30_000))
    (tmp_path / "compsci_1990.txt").write_text("\n".join([COMPUTING] * 30_000))
    (tmp_path / "hci_1990.txt").write_text("\n".join([COMPUTING + " user"] * 1_100))

    table = decade_cross_entropy(sorted(tmp_path.glob("*.txt")), fixed_budget=50_000)

    assert set(table.text_field) == {"philosophy", "compsci"}
    assert set(table.training_tokens) == {50_000}


def test_add_excess_compares_within_each_seed() -> None:
    table = pd.DataFrame(
        {
            "seed": [1, 1, 2, 2],
            "decade": [2010] * 4,
            "model_field": ["a", "b", "a", "b"],
            "text_field": ["a", "a", "a", "a"],
            "cross_entropy": [5.0, 6.0, 4.0, 7.0],
        }
    )

    assert list(add_excess(table).excess) == [0.0, 1.0, 0.0, 3.0]


def test_word_bits_add_up_to_the_cross_entropy() -> None:
    documents = [PHILOSOPHY, COMPUTING]
    model = KneserNeyBigram(set(PHILOSOPHY.split())).fit([PHILOSOPHY] * 10)

    bits, occurrences = model.word_bits(documents)

    assert sum(bits.values()) / sum(occurrences.values()) == pytest.approx(
        model.cross_entropy(documents)
    )


def test_document_cross_entropy_is_the_length_weighted_piece_of_the_corpus_value() -> None:
    documents = [PHILOSOPHY, COMPUTING, "the world model learns"]
    model = KneserNeyBigram(set(PHILOSOPHY.split())).fit([PHILOSOPHY] * 10)

    per_document = [model.document_cross_entropy(document) for document in documents]
    lengths = [len(document.split()) + 1 for document in documents]
    weighted = sum(h * n for h, n in zip(per_document, lengths, strict=True)) / sum(lengths)

    assert model.document_cross_entropy(PHILOSOPHY) == pytest.approx(
        model.cross_entropy([PHILOSOPHY])
    )
    assert weighted == pytest.approx(model.cross_entropy(documents))
    assert per_document[0] < per_document[1]
