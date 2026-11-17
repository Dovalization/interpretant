"""Tests for PPMI-SVD cell embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from gensim.models import KeyedVectors
from scipy import sparse

from interpretant.embedding.grid import build_ppmi_grid, discover_cells
from interpretant.embedding.ppmi import cooccurrence, ppmi, train_cell, train_lines, vocabulary


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


def test_train_lines_gives_the_vectors_train_cell_gives_for_the_same_text(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    words = ["cat", "dog", "horse", "hammer", "saw", "drill", "grass", "wood", "barn", "shed"]
    lines = [" ".join(rng.choice(words, size=12)) for _ in range(400)]
    path = tmp_path / "toy_1990.txt"
    path.write_text("\n".join(lines))

    from_file = train_cell(path, window=2, dimensions=4, min_count=5)
    from_lines = train_lines(lines, window=2, dimensions=4, min_count=5)

    assert from_file.index_to_key == from_lines.index_to_key
    assert np.array_equal(from_file.vectors, from_lines.vectors)


def test_cooccurrence_never_counts_a_pair_across_lines() -> None:
    index = {"a": 0, "b": 1}

    assert cooccurrence(["x a", "b x"], index, window=3).toarray()[0, 1] == 0


def test_ppmi_grid_builds_every_cell_and_resumes(tmp_path: Path) -> None:
    texts = tmp_path / "phrased" / "openalex" / "texts"
    texts.mkdir(parents=True)
    for stem in ("hci_2000", "hci_2010"):
        (texts / f"{stem}.txt").write_text("\n".join(["the user sees the interface"] * 300))
    cells = discover_cells(tmp_path / "phrased", ("openalex",))
    output = tmp_path / "m"

    first = build_ppmi_grid(cells, output, window=5, dimensions=2, min_count=1)
    second = build_ppmi_grid(cells, output, window=5, dimensions=2, min_count=1)

    assert [path.name for path in first] == ["hci_2000.kv", "hci_2010.kv"]
    assert second == []
    loaded = KeyedVectors.load(str(output / "openalex" / "hci_2000.kv"))
    assert loaded.get_vecattr("interface", "count") == 300


def test_vocabulary_applies_the_count_threshold() -> None:
    assert vocabulary(["a a b", "a c"], min_count=2) == {"a": 3}


def test_train_lines_gives_no_vectors_to_a_cell_without_two_counted_words() -> None:
    assert len(train_lines(["a b c"], min_count=20)) == 0


def batch_cooccurrence(lines: list[str], index: dict[str, int], window: int) -> sparse.csr_matrix:
    """The counting the module did before, by adding a matrix per line: the reference."""
    size = len(index)
    total = sparse.csr_matrix((size, size), dtype=np.float64)
    for line in lines:
        ids = np.array([index[word] for word in line.split() if word in index], dtype=np.int64)
        rows: list[np.ndarray] = []
        columns: list[np.ndarray] = []
        for offset in range(1, window + 1):
            if len(ids) > offset:
                rows += [ids[:-offset], ids[offset:]]
                columns += [ids[offset:], ids[:-offset]]
        if rows:
            row, column = np.concatenate(rows), np.concatenate(columns)
            ones = np.ones(len(row), dtype=np.float64)
            total = total + sparse.coo_matrix((ones, (row, column)), shape=(size, size)).tocsr()
    return total


def test_cooccurrence_counts_exactly_as_adding_matrices_per_line_did() -> None:
    rng = np.random.default_rng(3)
    words = [f"w{n}" for n in range(40)]
    lines = [" ".join(rng.choice([*words, "oov"], size=rng.integers(0, 30))) for _ in range(300)]
    index = {word: position for position, word in enumerate(words)}

    counted = cooccurrence(lines, index, window=5)
    reference = batch_cooccurrence(lines, index, window=5)

    assert counted.has_canonical_format
    assert np.array_equal(counted.indptr, reference.indptr)
    assert np.array_equal(counted.indices, reference.indices)
    assert np.array_equal(counted.data, reference.data)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_the_cuda_svd_finds_the_neighbours_arpack_finds() -> None:
    rng = np.random.default_rng(4)
    animals, tools = ["cat", "dog", "horse", "cow"], ["hammer", "saw", "drill", "wrench"]
    lines = []
    for _ in range(600):
        lines.append(f"the {rng.choice(animals)} eats grass and sleeps in the barn")
        lines.append(f"the {rng.choice(tools)} cuts wood and sits in the shed")

    on_cpu = train_lines(lines, window=2, dimensions=4, min_count=5)
    on_gpu = train_lines(lines, window=2, dimensions=4, min_count=5, svd_backend="cuda")

    assert on_cpu.index_to_key == on_gpu.index_to_key
    for word in ("cat", "hammer"):
        cpu_neighbours = {neighbour for neighbour, _ in on_cpu.most_similar(word, topn=3)}
        gpu_neighbours = {neighbour for neighbour, _ in on_gpu.most_similar(word, topn=3)}
        assert cpu_neighbours == gpu_neighbours
