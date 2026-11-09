"""Tests for phrase detection."""

from __future__ import annotations

from pathlib import Path

from interpretant.corpus.phrases import PhraseModel, learn_phrases, sample_chunks, split_targets

TRAINING = (
    [["the", "new", "york", "times", "reported", "it"]] * 40
    + [["a", "network", "analysis", "of", "trade"]] * 40
    + [["the", "theory", "of", "mind", "debate"]] * 40
    + [["markets", "and", "prices", "move"]] * 40
    + [["the", "lan", "guage", "of", "o", "f", "texts"]] * 40
    + [["a", "language", "game"]] * 40
    + [["it", "may", "be", "argued"]] * 60
    + [["maybe", "not"]] * 10
)


def test_split_targets_separates_phrase_targets() -> None:
    assert split_targets(["network", "neural_network", "mind"]) == (
        {"network", "mind"},
        {"neural_network"},
    )


def test_learned_phrases_merge_but_never_absorb_a_protected_target() -> None:
    model = learn_phrases(TRAINING, protected={"network", "mind"}, forced=set(), min_count=5)

    assert model.apply("the new york times") == "the new_york times"
    assert model.apply("a network analysis") == "a network analysis"
    assert model.apply("theory of mind") == "theory of mind"


def test_forced_phrase_targets_merge_even_when_never_seen() -> None:
    model = learn_phrases(TRAINING, protected={"network"}, forced={"neural_network"}, min_count=5)

    assert model.apply("a neural network learns") == "a neural_network learns"


def test_sample_chunks_respects_the_token_budget(tmp_path: Path) -> None:
    path = tmp_path / "books_1900.txt"
    path.write_text(" ".join(["word"] * 2_500) + "\n" + " ".join(["other"] * 500) + "\n")

    chunks = sample_chunks(path, token_budget=1_200)

    assert all(len(chunk) <= 1_000 for chunk in chunks)
    assert 1_200 <= sum(len(chunk) for chunk in chunks) < 2_200


def test_broken_words_are_repaired_and_one_letter_pairs_dropped() -> None:
    model = learn_phrases(TRAINING, protected=set(), forced=set(), min_count=5)

    assert model.apply("the lan guage of o f texts") == "the language of o f texts"
    assert "maybe" not in model.apply("it may be argued")


def test_phrase_model_round_trips_through_save_and_load(tmp_path: Path) -> None:
    model = learn_phrases(TRAINING, protected={"network"}, forced={"neural_network"}, min_count=5)
    model.save(tmp_path / "phrases.model")

    loaded = PhraseModel.load(tmp_path / "phrases.model")

    line = "the lan guage of a neural network in new york"
    assert loaded.apply(line) == model.apply(line) == "the language of a neural_network in new_york"
