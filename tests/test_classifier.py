"""The cue-word classifier on synthetic inputs: cues, example sentences and classification."""

from __future__ import annotations

import numpy as np

from interpretant.drift.classifier import classify, cue_sense, example_usage
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
