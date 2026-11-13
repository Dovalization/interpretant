"""Sense induction on synthetic occurrences with planted senses and a fake encoder.

The fake encoder places a context by its cue words: the attention mechanism's (*transformer,
layer, weights*) on one axis, selective attention's (*visual, task, participants*) on another.
AI's 2020s abstracts use the mechanism sense, AI's 1990s abstracts and psychology's use the
other, so the clustering must find two senses and place the change in AI.
"""

from __future__ import annotations

import gzip
import json
import random
from collections.abc import Sequence
from pathlib import Path

import pytest

from interpretant.drift.senses import (
    Sampling,
    cased_abstracts,
    context_around,
    drop_shared,
    foreign_share,
    sample_many,
    sample_occurrences,
    word_pattern,
)

MECHANISM = ("transformer", "layer", "weights", "encoder")
SELECTIVE = ("visual", "task", "participants", "selective")


def abstract(rng: random.Random, cues: Sequence[str], number: int) -> tuple[str, str]:
    filler = " ".join(rng.choices(["the", "study", "results", "show", "we", "data"], k=8))
    sentence = f"We report {' '.join(rng.sample(list(cues), 3))} attention in {filler}."
    return f"W{number}", f"Background matters here. {sentence} Further work follows."


@pytest.fixture(scope="module")
def abstracts() -> dict[tuple[str, int], list[tuple[str, str]]]:
    rng = random.Random(1)
    plan = {
        ("ai", 1990): SELECTIVE,
        ("ai", 2020): MECHANISM,
        ("psychology", 1990): SELECTIVE,
        ("psychology", 2020): SELECTIVE,
    }
    cells: dict[tuple[str, int], list[tuple[str, str]]] = {}
    for index, (cell, cues) in enumerate(plan.items()):
        documents = [abstract(rng, cues, index * 1_000 + n) for n in range(60)]
        documents += [(f"X{index}{n}", "No target word in this one at all.") for n in range(20)]
        cells[cell] = documents
    return cells


def test_the_word_matches_in_any_case_and_inside_compounds_only_as_a_whole_word() -> None:
    pattern = word_pattern("attention")

    assert pattern.search("Self-attention layers")
    assert pattern.search("ATTENTION was paid")
    assert not pattern.search("attentional control")


def test_the_context_is_the_sentence_and_keeps_the_occurrence_offsets() -> None:
    text = "First sentence here. Selective attention (SA) matters! Last one."
    start = text.index("attention")

    context, begin, end = context_around(text, start, start + len("attention"))

    assert context == "Selective attention (SA) matters!"
    assert context[begin:end] == "attention"


def test_a_long_sentence_is_cut_to_a_window_around_the_occurrence() -> None:
    words = [f"w{n}" for n in range(300)]
    words[200] = "attention"
    text = " ".join(words) + "."
    start = text.index("attention")

    context, begin, end = context_around(text, start, start + 9, max_words=20)

    assert len(context.split()) == 20
    assert context[begin:end] == "attention"


def test_sampling_takes_one_occurrence_per_abstract_up_to_the_cap_and_repeats_by_seed(
    abstracts: dict[tuple[str, int], list[tuple[str, str]]],
) -> None:
    first = sample_occurrences(abstracts, "attention", seed=1, sampling=Sampling(cap=25))
    again = sample_occurrences(abstracts, "attention", seed=1, sampling=Sampling(cap=25))
    other = sample_occurrences(abstracts, "attention", seed=2, sampling=Sampling(cap=25))

    assert first == again
    assert {occurrence.work_id for occurrence in first} != {o.work_id for o in other}
    assert len(first) == 4 * 25
    assert len({(o.field, o.decade, o.work_id) for o in first}) == len(first)
    assert all(o.context[o.start : o.end].lower() == "attention" for o in first)


def test_cased_abstracts_keep_case_and_punctuation_and_drop_the_label(tmp_path: Path) -> None:
    text = (
        "Abstract: We studied how People direct their attention. The results, which were "
        "clear, show that attention follows the task and that the effect is large."
    )
    records = [{"id": "W1", "year": 1995, "type": "article", "authors": [], "abstract": text}]
    (tmp_path / "psychology").mkdir()
    with gzip.open(tmp_path / "psychology" / "1990.jsonl.gz", "wt", encoding="utf-8") as cell:
        cell.write("\n".join(json.dumps(record) for record in records))

    found = cased_abstracts(tmp_path, "psychology", 1990)

    assert found == [("W1", text.removeprefix("Abstract: "))]


def test_an_abstract_in_two_fields_of_one_decade_is_dropped_from_both() -> None:
    abstracts = {
        ("ai", 2020): [("W1", "a"), ("W2", "b")],
        ("psychology", 2020): [("W2", "b"), ("W3", "c")],
        ("psychology", 1990): [("W1", "a")],
    }

    kept = drop_shared(abstracts)

    assert kept == {
        ("ai", 2020): [("W1", "a")],
        ("psychology", 2020): [("W3", "c")],
        ("psychology", 1990): [("W1", "a")],
    }


def test_drawing_with_replacement_varies_a_small_cell_between_seeds() -> None:
    cell = {("ai", 2020): [(f"W{n}", f"Some attention here, number {n}.") for n in range(20)]}
    sampling = Sampling(cap=500, replace=True)

    first = sample_occurrences(cell, "attention", seed=1, sampling=sampling)
    second = sample_occurrences(cell, "attention", seed=2, sampling=sampling)

    assert len(first) == len(second) == 20
    assert sorted(o.work_id for o in first) != sorted(o.work_id for o in second)
    assert len({o.work_id for o in first}) < 20


def test_the_plural_counts_only_when_asked() -> None:
    cell = {("psychology", 2000): [("W1", "Patients reported auditory hallucinations at night.")]}

    assert sample_occurrences(cell, "hallucination", seed=1) == []
    plural = sample_occurrences(cell, "hallucination", 1, Sampling(plural=True))
    assert plural[0].context[plural[0].start : plural[0].end] == "hallucinations"


def test_the_quality_rules_skip_foreign_script_and_drop_urls() -> None:
    cell = {
        ("ai", 2020): [
            ("W1", "目的 在图像分类领域小样本学习旨在利用知识 attention 处理任务。"),
            ("W2", "See https://example.org/x for attention maps in vision models."),
        ]
    }

    found = sample_occurrences(cell, "attention", 1, Sampling(quality=True))

    assert foreign_share(cell[("ai", 2020)][0][1]) > 0.2
    assert [o.work_id for o in found] == ["W2"]
    assert "https" not in found[0].context
    assert found[0].context[found[0].start : found[0].end] == "attention"


def test_sampling_many_words_in_one_pass_matches_sampling_each_word() -> None:
    texts = [
        "The attention mechanism weights tokens. See https://example.org for attention maps.",
        "Self-attention and Attention heads differ; networks of attention.",
        "A network of neurons, and two networks of roads, with no attentions here.",
        "Network effects shape platforms; the platform grows. Ε λ λ η ν ι κ ά κείμενα πολλά.",
        "Prompts and a prompt: the model prompted users to answer the prompt quickly.",
        "Nothing relevant in this abstract at all, only filler words and more filler.",
    ] * 4
    cells = {
        ("ai", 2010): [(f"w{i}", text) for i, text in enumerate(texts)],
        ("hci", 2020): [(f"h{i}", text) for i, text in enumerate(reversed(texts))],
    }
    plan = [
        ("attention", Sampling(cap=3, plural=True, quality=True)),
        ("network", Sampling(cap=5, plural=True, quality=True)),
        ("platform", Sampling(cap=2, plural=False, quality=False)),
        ("prompt", Sampling(cap=10, plural=True, quality=False)),
        ("absent", Sampling(cap=3, plural=True, quality=True)),
    ]
    many = sample_many(cells, plan, seed=7)
    for word, sampling in plan:
        assert many[word] == sample_occurrences(cells, word, seed=7, sampling=sampling)
