"""The cue-word classifier on synthetic inputs: cues, example sentences and classification."""

from __future__ import annotations

import numpy as np
import pytest

from interpretant.drift.classifier import (
    classify,
    cue_first,
    cue_sense,
    example_usage,
    out_of_scope,
    sqrt_balanced,
)
from interpretant.drift.pairs import Usage

CUES = {"aircraft": ["pilot", "landed"], "surface": ["inclined", "horizontal"]}


def usage(text: str, word: str = "plane") -> Usage:
    start = text.index(word)
    return Usage(text, start, start + len(word))


def test_a_cue_near_the_target_names_its_sense() -> None:
    assert cue_sense(usage("the pilot said the plane was late"), CUES) == "aircraft"


def test_cues_of_two_senses_or_none_give_no_seed() -> None:
    assert cue_sense(usage("the pilot drew a horizontal plane"), CUES) is None
    assert cue_sense(usage("a plane appeared"), CUES) is None


def test_cues_beyond_the_window_do_not_count() -> None:
    text = "pilot one two three four five six the plane"
    assert cue_sense(usage(text), CUES, window=5) is None


def test_the_example_sentence_marks_the_first_word_starting_with_the_lemma() -> None:
    found = example_usage("plane", "Two Planes landed in Chicago.")
    assert found is not None
    assert found.text[found.start : found.end] == "Planes"
    assert example_usage("plane", "No such word here.") is None


def test_the_classifier_follows_the_seeds() -> None:
    seeds = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    labels = classify(seeds, ["a", "a", "b", "b"], np.array([[0.95, 0.05], [0.05, 0.95]]))
    assert labels == ["a", "b"]


def test_seeds_of_one_sense_label_everything_with_it() -> None:
    assert classify(np.ones((2, 2)), ["a", "a"], np.zeros((3, 2))) == ["a", "a", "a"]


def test_a_cue_can_be_held_to_one_side_of_the_target() -> None:
    cues = {"perfect": [">obtained"], "possessing": ["images"]}
    assert (
        cue_sense(usage("the data was obtained from a source having many images", "having"), cues)
        == "possessing"
    )
    assert cue_sense(usage("having obtained the data we fit a model", "having"), cues) == "perfect"
    assert (
        cue_sense(usage("without having to retrain", "having"), {"obligation": ["<without"]})
        == "obligation"
    )
    assert (
        cue_sense(usage("having to retrain without data", "having"), {"obligation": ["<without"]})
        is None
    )


def test_a_cue_can_be_held_to_the_word_right_next_to_the_target() -> None:
    cues = {"obligation": [">>to"]}
    assert cue_sense(usage("without having to use resources", "having"), cues) == "obligation"
    assert cue_sense(usage("reservoirs having short memory to project", "having"), cues) is None
    assert cue_sense(usage("we use prompt engineering here", "prompt"), {"llm": ["<<we"]}) is None
    assert cue_sense(usage("a novel prompt tuning method", "prompt"), {"llm": ["<<novel"]}) == "llm"


def test_the_confusion_matrix_is_normalised_per_true_label() -> None:
    from interpretant.drift.classifier import confusion

    matrix = confusion(["a", "a", "a", "a", "b"], ["a", "a", "a", "b", "b"], ["a", "b", "c"])
    assert matrix.tolist() == [[0.75, 0.25, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def test_adjusted_shares_undo_a_known_confusion() -> None:
    from interpretant.drift.classifier import adjusted_shares

    # true shares 0.2 a / 0.8 b; the classifier calls 30% of b "a" and 10% of a "b"
    matrix = np.array([[0.9, 0.1], [0.3, 0.7]])
    predicted = ["a"] * 42 + ["b"] * 58  # 0.2*0.9 + 0.8*0.3 = 0.42 observed as a
    assert adjusted_shares(predicted, matrix, ["a", "b"]).tolist() == pytest.approx([0.2, 0.8])


def test_adjusted_shares_never_go_negative() -> None:
    from interpretant.drift.classifier import adjusted_shares

    shares = adjusted_shares(["a"] * 5 + ["b"] * 95, np.array([[0.9, 0.1], [0.3, 0.7]]), ["a", "b"])
    assert (shares >= 0).all() and shares.sum() == pytest.approx(1.0)


def test_tidy_usage_cleans_artefacts_and_rebuilds_the_span() -> None:
    from interpretant.drift.classifier import tidy_usage

    text = (
        "the quan\u00ad titative \\nrequired  \u00e2\u20ac\u0153components\u00e2\u20ac\u009d of\tit"
    )
    start = text.index("components")
    tidied = tidy_usage(Usage(text, start, start + len("components")))
    assert tidied.text == "the quantitative required \u201ccomponents\u201d of it"
    assert tidied.text[tidied.start : tidied.end] == "components"


def test_sentences_repeated_across_abstracts_are_boilerplate() -> None:
    from interpretant.drift.classifier import boilerplate

    texts = ["Published by Frontiers."] * 5 + ["A real sentence."] * 2
    assert boilerplate(texts) == {"Published by Frontiers."}


def test_an_exclusion_cue_silences_its_sense_only() -> None:
    cues = {"structure": ["station", "!work"], "base": ["computer"]}
    assert cue_sense(usage("trains leave the station platform", "platform"), cues) == "structure"
    assert cue_sense(usage("a work station class platform", "platform"), cues) is None
    assert cue_sense(usage("work station and computer platform", "platform"), cues) == "base"


def test_a_scope_word_near_the_target_puts_the_usage_out_of_scope() -> None:
    words = ["ergative", "accusative"]
    text = "the language has ergative alignment in its case marking"
    assert out_of_scope(usage(text, "alignment"), words)
    assert not out_of_scope(usage("word alignment between two languages", "alignment"), words)
    far = "ergative one two three four five six alignment"
    assert not out_of_scope(usage(far, "alignment"), words)


def test_square_root_weights_favour_thin_senses_less_than_balanced_ones() -> None:
    weights = sqrt_balanced(["common"] * 9 + ["rare"])
    assert weights["rare"] == pytest.approx(5**0.5)
    assert weights["common"] == pytest.approx((10 / 18) ** 0.5)
    assert weights["rare"] / weights["common"] == pytest.approx(3.0)


def test_cues_decide_where_they_fire_and_the_classifier_elsewhere() -> None:
    usages = [
        usage("The pilot landed the plane safely."),
        usage("The plane was inclined at ten degrees."),
        usage("We saw the plane from the hill."),
    ]
    labels = cue_first(usages, CUES, ["surface", "aircraft", "aircraft"])
    assert labels == ["aircraft", "surface", "aircraft"]
