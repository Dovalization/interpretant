"""Tests for cross-perplexity between fields."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from interpretant.drift.perplexity import (
    END,
    UNKNOWN,
    DecadeModels,
    KneserNeyBigram,
    UnigramModel,
    add_excess,
    cells_by_decade,
    cross_entropy_runs,
    decade_cross_entropy,
    decompose_fall,
    decompose_runs,
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


def test_known_bits_per_token_score_vocabulary_words_and_leave_unknown_ones_out() -> None:
    model = KneserNeyBigram(set(PHILOSOPHY.split())).fit([PHILOSOPHY] * 10)

    bits, occurrences = model.word_bits([COMPUTING])
    known = sum(occurrences.values()) - occurrences[UNKNOWN]

    assert model.known_bits_per_token(COMPUTING) == pytest.approx(
        (sum(bits.values()) - bits[UNKNOWN]) / known
    )
    assert model.known_bits_per_token(PHILOSOPHY) == pytest.approx(
        model.document_cross_entropy(PHILOSOPHY)
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


def test_unigram_control_sums_to_one_and_prefers_its_own_words() -> None:
    vocabulary = set(PHILOSOPHY.split()) | set(COMPUTING.split())
    model = UnigramModel(vocabulary).fit([PHILOSOPHY] * 20)

    assert sum(model.probability(word) for word in model.vocabulary) == pytest.approx(1.0)
    assert model.cross_entropy([PHILOSOPHY]) < model.cross_entropy([COMPUTING])
    bits, occurrences = model.word_bits([COMPUTING])
    assert sum(bits.values()) / sum(occurrences.values()) == pytest.approx(
        model.cross_entropy([COMPUTING])
    )


def test_fall_decomposes_into_mix_and_rate_and_pure_mix_changes_leave_rate_at_zero() -> None:
    vocabulary = set(PHILOSOPHY.split()) | set(COMPUTING.split())
    model_a = UnigramModel(vocabulary).fit([PHILOSOPHY] * 20)
    model_b = UnigramModel(vocabulary).fit([COMPUTING] * 20)
    # Same models in both decades, held-out text changing from computing-heavy to mixed:
    # every change in excess is a change of word mix.
    early = DecadeModels({"a": model_a, "b": model_b}, {"b": [COMPUTING] * 4}, 1)
    late = DecadeModels({"a": model_a, "b": model_b}, {"b": [COMPUTING, PHILOSOPHY]}, 1)

    decomposition = decompose_fall(early, late, "a", "b")

    assert decomposition.fall == pytest.approx(decomposition.mix + decomposition.rate)
    assert decomposition.rate == pytest.approx(0.0)
    assert decomposition.fall > 0


def write_grid(root: Path, decades: tuple[int, ...]) -> None:
    for decade in decades:
        for field, sentence in (("philosophy", PHILOSOPHY), ("compsci", COMPUTING)):
            (root / f"{field}_{decade}.txt").write_text("\n".join([sentence] * 6_000))


def test_cells_by_decade_groups_cells_in_decade_order(tmp_path: Path) -> None:
    write_grid(tmp_path, (2020, 1990))

    grouped = cells_by_decade(tmp_path)

    assert list(grouped) == [1990, 2020]
    assert [path.name for path in grouped[1990]] == ["compsci_1990.txt", "philosophy_1990.txt"]


def test_cross_entropy_runs_scores_every_seed_and_decade_with_excess(tmp_path: Path) -> None:
    write_grid(tmp_path, (1990, 2020))
    logged: list[str] = []

    table = cross_entropy_runs(tmp_path, seeds=(1, 2), fixed_budget=50_000, log=logged.append)

    assert len(table) == 2 * 2 * 4
    assert set(table.seed) == {1, 2}
    assert set(table.decade) == {1990, 2020}
    own = table[table.model_field == table.text_field]
    assert (own.excess == 0).all()
    assert len(logged) == 4


def test_decompose_runs_splits_each_seed_fall_into_mix_and_rate(tmp_path: Path) -> None:
    write_grid(tmp_path, (1990, 2020))

    table = decompose_runs(tmp_path, 1990, 2020, seeds=(1, 2), fixed_budget=50_000)

    assert len(table) == 2 * 2
    assert set(table.early) == {1990}
    assert list(table.fall) == pytest.approx(list(table.mix + table.rate))


def test_decompose_runs_needs_two_fields_in_both_decades(tmp_path: Path) -> None:
    write_grid(tmp_path, (1990,))

    with pytest.raises(ValueError, match="fewer than two qualifying fields"):
        decompose_runs(tmp_path, 1990, 2020, seeds=(1,), fixed_budget=50_000)
