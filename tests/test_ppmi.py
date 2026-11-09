"""Tests for PPMI-SVD cell embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from click.testing import CliRunner
from gensim.models import KeyedVectors

from interpretant.__main__ import cli
from interpretant.embedding.ppmi import cooccurrence, ppmi, train_cell, vocabulary


def test_cooccurrence_counts_pairs_within_the_window_symmetrically() -> None:
    index = {"a": 0, "b": 1, "c": 2}

    counts = cooccurrence(["a b c", "a b"], index, window=1).toarray()

    assert counts[0, 1] == counts[1, 0] == 2
    assert counts[1, 2] == counts[2, 1] == 1
    assert counts[0, 2] == 0


def test_cooccurrence_drops_unknown_words_before_windowing_like_word2vec() -> None:
    counts = cooccurrence(["a x c"], {"a": 0, "c": 1}, window=1).toarray()

    assert counts[0, 1] == counts[1, 0] == 1


def test_ppmi_keeps_only_positive_associations() -> None:
    index = {"cat": 0, "dog": 1, "purr": 2, "bark": 3}
    lines = ["cat purr"] * 20 + ["dog bark"] * 20 + ["cat dog"] * 2

    matrix = ppmi(cooccurrence(lines, index, window=1)).toarray()

    assert matrix[0, 2] > 0
    assert matrix[0, 3] == 0
    assert (matrix >= 0).all()


def test_train_cell_groups_words_used_alike(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    animals, tools = ["cat", "dog", "horse", "cow"], ["hammer", "saw", "drill", "wrench"]
    lines = []
    for _ in range(600):
        lines.append(f"the {rng.choice(animals)} eats grass and sleeps in the barn")
        lines.append(f"the {rng.choice(tools)} cuts wood and sits in the shed")
    path = tmp_path / "toy_1990.txt"
    path.write_text("\n".join(lines))

    vectors = train_cell(path, window=2, dimensions=4, min_count=5)

    assert vectors.get_vecattr("cat", "count") > 100
    nearest = [word for word, _ in vectors.most_similar("cat", topn=3)]
    assert set(nearest) <= set(animals)


def test_ppmi_grid_builds_every_cell_and_resumes(tmp_path: Path) -> None:
    texts = tmp_path / "phrased" / "openalex" / "texts"
    texts.mkdir(parents=True)
    for stem in ("hci_2000", "hci_2010"):
        (texts / f"{stem}.txt").write_text("\n".join(["the user sees the interface"] * 300))
    args = ["ppmi", "grid", "--input-dir", str(tmp_path / "phrased"), "--layer", "openalex"]
    args += ["--output-dir", str(tmp_path / "m"), "--dimensions", "2", "--min-count", "1"]

    first = CliRunner().invoke(cli, args)
    second = CliRunner().invoke(cli, args)

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    loaded = KeyedVectors.load(str(tmp_path / "m" / "openalex" / "hci_2000.kv"))
    assert loaded.get_vecattr("interface", "count") == 300


def test_vocabulary_applies_the_count_threshold() -> None:
    assert vocabulary(["a a b", "a c"], min_count=2) == {"a": 3}
