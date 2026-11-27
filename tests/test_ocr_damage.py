"""The junk gate's OCR damage score: the words read, which are known, and its check."""

from __future__ import annotations

import numpy as np
import pytest

from interpretant.review.ocr_damage import (
    build_corpus_words,
    compute_ocr_check,
    count_document_frequencies,
    count_unknown_words,
    find_lower_case_words,
)


def test_lower_case_words_skip_capitals_short_words_digits_and_left_out_spans() -> None:
    text = "The model ran 3 tbe-qnick tests in UNESCO data, removed here."
    removed = (text.index("removed"), len(text))

    words = find_lower_case_words(text, min_letters=3, left_out=[removed])

    assert words == ["model", "ran", "tbe", "qnick", "tests", "data"]


def test_a_sentences_words_leave_out_the_target_and_the_text_around() -> None:
    text = "Before. The network learns every network weight. After."
    sentence = (text.index("The"), text.index(" After"))
    target = (text.index("network"), text.index("network") + len("network"))

    words = find_lower_case_words(text, min_letters=3, within=sentence, left_out=[target])

    assert words == ["learns", "every", "network", "weight"]


def test_document_frequencies_count_each_text_once_per_word() -> None:
    frequencies = count_document_frequencies(
        ["the cat sat on the cat", "a cat ran", "Cat"], min_letters=3
    )

    assert frequencies.rows() == [("cat", 2), ("ran", 1), ("sat", 1), ("the", 1)]


def test_corpus_words_are_those_in_enough_works_over_every_part() -> None:
    first = count_document_frequencies(["glial cells", "glial"], min_letters=3)
    second = count_document_frequencies(["glial", "cells"], min_letters=3)

    assert build_corpus_words([first, second], min_works=3) == {"glial"}
    assert build_corpus_words([first, second], min_works=2) == {"glial", "cells"}
    assert build_corpus_words([], min_works=1) == frozenset()


def test_a_word_is_unknown_when_neither_the_corpus_nor_the_dictionary_knows_it() -> None:
    words = ["model", "tbe", "connectome", "qnick", "children"]

    assert count_unknown_words(words, frozenset()) == 3
    assert count_unknown_words(words, frozenset({"connectome"})) == 2


def test_the_ocr_check_reads_the_auc_and_the_cut_that_separates_best() -> None:
    scores = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 0.9])
    damaged = np.array([False, False, False, True, True, False, True, True])

    check = compute_ocr_check(scores, damaged)

    assert check.auc == pytest.approx(14 / 16)
    assert check.best_cut == pytest.approx(0.3)
    assert (check.sensitivity, check.specificity) == (1.0, 0.75)
    assert check.meets_bar
    assert (check.sentences, check.damaged) == (8, 4)


def test_the_ocr_check_needs_both_damaged_and_undamaged_sentences() -> None:
    with pytest.raises(ValueError, match="both"):
        compute_ocr_check(np.array([0.1, 0.2]), np.array([False, False]))
