"""Sense induction on synthetic inputs: the substitute pattern, cleaning, communities, placement,
neighbours and dictionary readers. No language model is loaded."""

from __future__ import annotations

import json
from pathlib import Path

from interpretant.drift.induction import (
    Community,
    cleaned,
    communities,
    kept_communities,
    neighbours,
    patterned,
    placed,
    wiktionary_entries,
)
from interpretant.drift.pairs import Usage

STOPS = frozenset({"the", "a", "of", "in"})


def test_the_pattern_follows_the_target_inside_its_context() -> None:
    usage = Usage("pay close attention to the road", 10, 19)
    assert patterned(usage, "<mask>") == "pay close attention (or even <mask>) to the road"


def test_fillers_lose_non_words_stop_words_target_forms_and_repeats() -> None:
    fillers = [" Focus", "the", "attentions", "Attention", "3", "x", "focus", " awareness"]
    assert cleaned(fillers, "attention", "Attention", STOPS) == ["focus", "awareness"]


def found_in_two_groups() -> list[list[str]]:
    plane = ["aircraft", "jet", "airplane"]
    surface = ["surface", "line", "level"]
    return [plane] * 6 + [surface] * 4


def test_substitutes_that_appear_together_form_communities_largest_first() -> None:
    groups = communities(found_in_two_groups(), min_usages=3)
    assert [g.name for g in groups] == ["C1", "C2"]
    assert groups[0].members == {"aircraft", "jet", "airplane"}
    assert groups[1].members == {"surface", "line", "level"}


def test_a_usage_goes_to_the_community_holding_most_of_its_substitutes() -> None:
    groups = communities(found_in_two_groups(), min_usages=3)
    assert placed(["jet", "line", "aircraft"], groups) == "C1"
    assert placed(["tree"], groups) is None


def test_communities_with_too_few_usages_are_dropped() -> None:
    groups = [
        Community("C1", frozenset({"a1", "a2"}), ("a1", "a2")),
        Community("C2", frozenset({"b1"}), ("b1",)),
    ]
    substitutes = [["a1"]] * 19 + [["b1"]]
    assert [g.name for g in kept_communities(substitutes, groups, min_share=0.1)] == ["C1"]


def test_neighbours_count_the_content_words_on_either_side() -> None:
    usages = [
        Usage("the jet plane landed", 8, 13),
        Usage("a jet plane took off", 6, 11),
        Usage("the plane of the table", 4, 9),
    ]
    assert neighbours(usages, STOPS) == [("before jet", 2), ("after landed", 1), ("after took", 1)]


def test_wiktionary_entries_keep_the_part_of_speech_and_its_topics(tmp_path: Path) -> None:
    path = tmp_path / "plane.jsonl"
    records = [
        {"pos": "noun", "senses": [{"glosses": ["An airplane."]}]},
        {"pos": "noun", "senses": [{"glosses": ["A flat surface."], "topics": ["geometry"]}]},
        {"pos": "verb", "senses": [{"glosses": ["To smooth with a plane."]}]},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    assert wiktionary_entries(path, "noun") == ["An airplane.", "(geometry) A flat surface."]
