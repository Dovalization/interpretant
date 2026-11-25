"""The matching audit's draws: shares, the recall search, the items, the key and the counts."""

from __future__ import annotations

from typing import Any

import polars as pl

from interpretant.catalog.words import Word
from interpretant.review.usages_audit import (
    KEPT_PER_WORD,
    LEFT_OUT_PER_WORD,
    MISSED_PER_WORD,
    WIDTH,
    build_forms,
    build_items,
    build_search_pattern,
    compute_shares,
    count_usages,
    draw_calibration,
    draw_kept,
    draw_left_out,
    draw_missed,
    format_left_out_reason,
    render_marked_sentence,
)

ATTENTION = Word(lemma="attention", pos=("NOUN",), match="lemma", held_out=False)
MEMORY = Word(lemma="memory", pos=("NOUN",), match="lemma", held_out=True)
NEURAL_NETWORK = Word(lemma="neural network", pos=("NOUN",), match="lemma", held_out=False)


def usage(work: int, word: str, decade: int, *, in_scope: bool = True) -> dict[str, Any]:
    return {
        "work_id": f"W{work}",
        "word": word,
        "content_key": f"{work}:{word}",
        "decade": decade,
        "in_scope": in_scope,
        "drawn": True,
        "part": f"ai_{decade}",
    }


def test_compute_shares_spreads_what_a_short_decade_cannot_give() -> None:
    assert compute_shares({1990: 2, 2000: 50, 2010: 50}, 20) == {1990: 2, 2000: 9, 2010: 9}
    assert compute_shares({2000: 3, 2010: 4}, 20) == {2000: 3, 2010: 4}


def test_build_forms_adds_spellings_and_plurals_longest_first() -> None:
    forms = build_forms(MEMORY, {"memorie": "memory"})
    assert set(forms) == {"memory", "memorys", "memoryes", "memories", "memorie", "memoriees"}
    assert [len(form) for form in forms] == sorted((len(form) for form in forms), reverse=True)
    assert build_forms(Word(lemma="gan", pos=("NOUN",), match="form", held_out=False), {}) == [
        "gan"
    ]


def test_build_search_pattern_finds_the_word_alone_or_as_a_hyphen_part() -> None:
    pattern = build_search_pattern(ATTENTION, {})
    text = "Attention-based models; self-attentions; attentional; attention2."
    assert [hit.group() for hit in pattern.finditer(text)] == ["Attention", "attentions"]


def test_build_search_pattern_lets_a_phrase_join_by_space_or_hyphen() -> None:
    pattern = build_search_pattern(NEURAL_NETWORK, {})
    text = "Neural networks and neural-network models, but not neural  nets."
    assert [hit.group() for hit in pattern.finditer(text)] == ["Neural networks", "neural-network"]


def test_render_marked_sentence_marks_the_target() -> None:
    row = {"sentence_start": 4, "sentence_end": 22, "target_start": 7, "target_end": 16}
    assert render_marked_sentence("Hi. We attention here.", row) == "We ⟨t⟩attention⟨/t⟩ here."


def test_render_marked_sentence_cuts_a_long_sentence_to_a_window_around_the_target() -> None:
    text = "a " * 400 + "attention" + " b" * 400
    start = text.index("attention")
    row = {
        "sentence_start": 0,
        "sentence_end": len(text),
        "target_start": start,
        "target_end": start + 9,
    }
    shown = render_marked_sentence(text, row)
    assert shown.startswith("…") and shown.endswith("…")
    assert len(shown) == WIDTH + 2
    assert "⟨t⟩attention⟨/t⟩" in shown


def test_format_left_out_reason_names_the_name_or_the_part_of_speech() -> None:
    assert format_left_out_reason({"inside_name": True, "entity": "ORG", "pos": "PROPN"}) == (
        "inside a name (ORG)"
    )
    assert format_left_out_reason({"inside_name": True, "entity": None, "pos": "NOUN"}) == (
        "inside a name (a named instrument)"
    )
    assert format_left_out_reason({"inside_name": False, "entity": None, "pos": "VERB"}) == (
        "part of speech VERB"
    )


def usages_table() -> pl.DataFrame:
    rows = [usage(work, "attention", 2000 + 10 * (work % 2)) for work in range(60)]
    rows += [usage(100 + work, "attention", 2000, in_scope=False) for work in range(15)]
    rows += [usage(200 + work, "memory", 2010) for work in range(5)]
    return pl.DataFrame(rows)


def test_draw_kept_takes_each_words_share_per_decade_and_is_seeded() -> None:
    kept = draw_kept(usages_table(), 7)
    per_word = dict(kept.group_by("word").len().iter_rows())
    assert per_word == {"attention": KEPT_PER_WORD, "memory": 5}
    attention = kept.filter(pl.col("word") == "attention")
    assert dict(attention.group_by("decade").len().iter_rows()) == {2000: 10, 2010: 10}
    assert (
        kept.sort("key")["key"].to_list()
        == draw_kept(usages_table(), 7).sort("key")["key"].to_list()
    )
    assert set(kept["key"]) != set(draw_kept(usages_table(), 8)["key"])


def test_draw_left_out_takes_the_smallest_keys_per_word() -> None:
    left_out = draw_left_out(usages_table(), 7)
    assert left_out.height == LEFT_OUT_PER_WORD
    assert not left_out["in_scope"].any()
    assert left_out["key"].to_list() == sorted(left_out["key"].to_list())


def test_draw_missed_keeps_the_smallest_keys_over_all_parts() -> None:
    first = {"attention": [{"key": f"{n:02d}"} for n in range(0, 40, 2)]}
    second = {"attention": [{"key": f"{n:02d}"} for n in range(1, 40, 2)]}
    missed = draw_missed([first, second])
    assert [hit["key"] for hit in missed] == [f"{n:02d}" for n in range(MISSED_PER_WORD)]


def sample_row(work: int, word: str) -> dict[str, Any]:
    return {
        "work_id": f"W{work}",
        "word": word,
        "sentence_start": 0,
        "sentence_end": 22,
        "target_start": 3,
        "target_end": 12,
        "part": "ai_2010",
        "inside_name": False,
        "entity": None,
        "pos": "VERB",
    }


def test_build_items_shuffles_with_the_seed_and_keys_each_item() -> None:
    samples = [("kept", sample_row(1, "attention")), ("left_out", sample_row(2, "memory"))]
    texts = {"W1": "We attention a lot of.", "W2": "We memory_ a lot of it."}
    words = {"attention": ATTENTION, "memory": MEMORY}
    items, key = build_items(samples, texts, words, 3)
    assert items == build_items(samples, texts, words, 3)[0]
    assert [item["id"] for item in items] == ["U0000", "U0001"]
    by_kind = {item["kind"]: item for item in items}
    assert by_kind["left_out"]["reason"] == "part of speech VERB"
    assert "reason" not in by_kind["kept"]
    assert by_kind["kept"]["sentence"] == "We ⟨t⟩attention⟨/t⟩ a lot of."
    left_out_id = by_kind["left_out"]["id"]
    assert key[left_out_id]["word"] == MEMORY.reported_name != "memory"


def test_draw_calibration_takes_up_to_each_kinds_size() -> None:
    key = {f"U{n:04d}": {"kind": kind} for n, kind in enumerate(["kept"] * 40 + ["missed"] * 5)}
    calibration = draw_calibration(key)
    assert calibration == sorted(calibration) == draw_calibration(key)
    kinds = [key[item_id]["kind"] for item_id in calibration]
    assert kinds.count("kept") == 30
    assert kinds.count("missed") == 5


def test_count_usages_names_words_by_their_reported_names() -> None:
    words = {"attention": ATTENTION, "memory": MEMORY}
    key = {"U0000": {"kind": "kept"}, "U0001": {"kind": "missed"}}
    counts = count_usages(usages_table(), {"attention": 4}, key, words)
    assert counts["in_scope_usages"] == {"attention": 60, MEMORY.reported_name: 5}
    assert counts["left_out_usages"] == {"attention": 15}
    assert counts["in_scope_by_decade"] == {"2000": 30, "2010": 35}
    assert counts["left_out_by_decade"] == {"2000": 15}
    assert counts["candidate_misses"] == {"attention": 4}
    assert counts["items"] == {"kept": 1, "missed": 1}
    assert counts["held_out_words"] == 1
