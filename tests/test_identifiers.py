"""The usages' identifiers: content keys follow the text, draw keys are uniform and independent."""

from __future__ import annotations

from scipy.stats import chisquare

from interpretant.usages.identifiers import build_content_key, build_draw_key

SENTENCE = "Attention is a limited resource."


def test_build_content_key_is_twenty_hex_characters_of_the_text_and_offsets() -> None:
    key = build_content_key(SENTENCE, 0, 9)
    assert len(key) == 20
    assert int(key, 16) >= 0
    assert key == build_content_key(SENTENCE, 0, 9)


def test_build_content_key_changes_with_the_offsets_and_the_text() -> None:
    key = build_content_key(SENTENCE, 0, 9)
    assert key != build_content_key(SENTENCE, 1, 9)
    assert key != build_content_key(SENTENCE + " ", 0, 9)


def test_build_draw_key_changes_with_each_part() -> None:
    key = build_draw_key(1, "https://openalex.org/W1", "attention", "a" * 20)
    assert key != build_draw_key(2, "https://openalex.org/W1", "attention", "a" * 20)
    assert key != build_draw_key(1, "https://openalex.org/W2", "attention", "a" * 20)
    assert key != build_draw_key(1, "https://openalex.org/W1", "memory", "a" * 20)
    assert key != build_draw_key(1, "https://openalex.org/W1", "attention", "b" * 20)


def test_draw_keys_are_uniform_over_their_first_hex_character() -> None:
    counts = [0] * 16
    for work in range(8000):
        key = build_draw_key(20261301, f"https://openalex.org/W{work}", "network", "c" * 20)
        counts[int(key[0], 16)] += 1
    assert chisquare(counts).pvalue > 0.001


def test_draw_keys_of_two_words_are_not_ordered_alike() -> None:
    works = [f"https://openalex.org/W{work}" for work in range(2000)]
    first = sorted(works, key=lambda w: build_draw_key(7, w, "network", "d" * 20))
    second = sorted(works, key=lambda w: build_draw_key(7, w, "memory", "d" * 20))
    shared = len(set(first[:200]) & set(second[:200]))
    assert shared < 60
