"""Sense induction on synthetic occurrences with planted senses and a fake encoder.

The fake encoder places a context by its cue words: the attention mechanism's (*transformer,
layer, weights*) on one axis, selective attention's (*visual, task, participants*) on another.
AI's 2020s abstracts use the mechanism sense, AI's 1990s abstracts and psychology's use the
other, so the clustering must find two senses and the measures must place the change in AI.
"""

from __future__ import annotations

import gzip
import json
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from interpretant.drift.senses import (
    Occurrence,
    Sampling,
    cased_abstracts,
    context_around,
    context_words,
    distribution,
    drop_shared,
    examples,
    foreign_share,
    induce_senses,
    jensen_shannon,
    narrowed,
    sample_many,
    sample_occurrences,
    sense_runs,
    sense_shares,
    span_mask,
    word_pattern,
)

MECHANISM = ("transformer", "layer", "weights", "encoder")
SELECTIVE = ("visual", "task", "participants", "selective")


class CueEncoder:
    """A context's vector counts its mechanism and selective-attention cues, plus noise."""

    def __init__(self) -> None:
        self.calls = 0
        self.rng = np.random.default_rng(0)

    def encode(self, occurrences: Sequence[Occurrence]) -> np.ndarray:
        self.calls += 1
        rows = []
        for occurrence in occurrences:
            words = occurrence.context.lower().split()
            rows.append(
                [
                    sum(word in MECHANISM for word in words),
                    sum(word in SELECTIVE for word in words),
                    0.1,
                ]
            )
        return np.array(rows, dtype=float) + self.rng.normal(0, 0.05, (len(rows), 3))


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


def test_the_span_mask_keeps_overlapping_tokens_and_drops_special_ones() -> None:
    offsets = np.array([[0, 0], [0, 4], [4, 13], [13, 14], [15, 20], [0, 0]])

    assert span_mask(offsets, 4, 13).tolist() == [False, False, True, False, False, False]
    assert span_mask(offsets, 2, 14).tolist() == [False, True, True, True, False, False]


def test_the_silhouette_picks_the_number_of_planted_clusters() -> None:
    rng = np.random.default_rng(3)
    centres = np.eye(3) * 5
    vectors = np.vstack([centre + rng.normal(0, 0.3, (40, 3)) for centre in centres])

    labels, found = induce_senses(vectors, seed=1)

    assert len(found) == 3
    assert all(len(set(labels[n * 40 : (n + 1) * 40])) == 1 for n in range(3))


def test_the_new_sense_is_found_in_ai_and_psychology_does_not_change(
    abstracts: dict[tuple[str, int], list[tuple[str, str]]],
) -> None:
    encoder = CueEncoder()

    table = sense_runs(
        abstracts, ["attention"], [1, 2], encoder, sampling=Sampling(cap=30), log=lambda _: None
    )

    rows = table[table.seed == 1]
    shares = sense_shares(rows)
    assert set(table.senses) == {2}
    assert jensen_shannon(distribution(shares, "ai", 1990), distribution(shares, "ai", 2020)) > 0.9
    psychology = [distribution(shares, "psychology", decade) for decade in (1990, 2020)]
    assert jensen_shannon(*psychology) < 0.01
    mechanism = int(rows[(rows.field == "ai") & (rows.decade == 2020)].sense.mode()[0])
    assert set(context_words(rows, mechanism, top=4)) == set(MECHANISM)
    assert all(
        any(cue in context for cue in MECHANISM) for context in examples(rows, mechanism, top=5)
    )


def test_an_occurrence_drawn_in_several_seeds_is_encoded_once(
    abstracts: dict[tuple[str, int], list[tuple[str, str]]],
) -> None:
    encoder = CueEncoder()

    table = sense_runs(
        abstracts, ["attention"], [1, 2, 3], encoder, sampling=Sampling(cap=60), log=lambda _: None
    )

    assert encoder.calls == 1
    assert len(table) == 3 * 4 * 60


def test_shares_fill_unused_senses_with_zero_and_sum_to_one() -> None:
    occurrences = [("ai", 1990, 0), ("ai", 1990, 0), ("ai", 2020, 1), ("ai", 2020, 2)]
    rows = pd.DataFrame(
        [{"field": f, "decade": d, "sense": s, "senses": 3} for f, d, s in occurrences]
    )

    shares = sense_shares(rows)

    assert distribution(shares, "ai", 1990).tolist() == [1.0, 0.0, 0.0]
    assert distribution(shares, "ai", 2020).tolist() == [0.0, 0.5, 0.5]
    assert jensen_shannon(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)


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


def test_a_narrowed_context_keeps_the_occurrence_and_its_offsets() -> None:
    context = "字" * 700 + "intelligence" + "字" * 300
    occurrence = Occurrence("ai", 2020, "W1", context, 700, 712)

    narrow = narrowed(occurrence, chars=50)

    assert len(narrow.context) == 50 + 12 + 50
    assert narrow.context[narrow.start : narrow.end] == "intelligence"
    assert narrowed(Occurrence("ai", 2020, "W1", "intelligence here", 0, 12), 50).start == 0


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


def test_a_single_number_of_senses_is_used_as_given() -> None:
    rng = np.random.default_rng(3)
    vectors = np.vstack([centre + rng.normal(0, 0.3, (40, 3)) for centre in np.eye(3) * 5])

    _, centres = induce_senses(vectors, seed=1, sense_range=(5,))

    assert len(centres) == 5


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
