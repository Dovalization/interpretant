"""A sense labeller with no generative model (docs/predictions_labels.md, round E).

Humans draw the lines: a word's senses come from a list someone approved. Each sense also has cue
words that signal it near the target. Usages whose neighbourhood holds the cues of exactly one
sense become training examples for it, together with the sense's own example sentence; a logistic
regression on XL-LEXEME's vectors of those examples then labels every other usage.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np

from interpretant.drift.pairs import Usage

TOKEN = re.compile(r"[a-z']+")


def cue_sense(usage: Usage, cues: Mapping[str, Sequence[str]], window: int = 5) -> str | None:
    """The one sense whose cues appear within ``window`` words of the target, or None when no
    sense's cues do or more than one sense's do. A cue written ``<word`` counts only before the
    target and ``>word`` only after it; ``<<word`` only as the word just before and ``>>word``
    only as the word just after; a plain cue counts on either side."""
    left = TOKEN.findall(usage.text[: usage.start].lower())[-window:]
    right = TOKEN.findall(usage.text[usage.end :].lower())[:window]
    before, after = set(left), set(right)

    def present(cue: str) -> bool:
        if cue.startswith("<<"):
            return bool(left) and left[-1] == cue[2:]
        if cue.startswith(">>"):
            return bool(right) and right[0] == cue[2:]
        if cue.startswith("<"):
            return cue[1:] in before
        if cue.startswith(">"):
            return cue[1:] in after
        return cue in before or cue in after

    found = [sense for sense, words in cues.items() if any(present(c) for c in words)]
    return found[0] if len(found) == 1 else None


def example_usage(word: str, example: str) -> Usage | None:
    """The sense's example sentence as a usage: the first word that starts with ``word``."""
    match = re.search(rf"\b{re.escape(word)}\w*", example, flags=re.IGNORECASE)
    return Usage(example, match.start(), match.end()) if match else None


def normalised(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.asarray(vectors / np.where(norms == 0, 1, norms))


def classify(
    seed_vectors: np.ndarray, seed_senses: Sequence[str], vectors: np.ndarray
) -> list[str]:
    """Each row of ``vectors`` labelled by a logistic regression trained on the seeds; with seeds
    of a single sense, every row takes that sense."""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    if len(set(seed_senses)) == 1:
        return [seed_senses[0]] * len(vectors)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    model.fit(normalised(seed_vectors), list(seed_senses))
    return [str(label) for label in model.predict(normalised(vectors))]


def confusion(truth: Sequence[str], predicted: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    """Row-normalised confusion matrix: entry (i, j) is the share of items whose true label is
    ``labels[i]`` that were predicted as ``labels[j]``. A label never seen as true gets an
    identity row (assumed always predicted correctly)."""
    index = {label: i for i, label in enumerate(labels)}
    counts = np.zeros((len(labels), len(labels)))
    for true, guess in zip(truth, predicted, strict=True):
        counts[index[true], index[guess]] += 1
    rows = counts.sum(axis=1, keepdims=True)
    matrix = np.divide(counts, rows, out=np.zeros_like(counts), where=rows > 0)
    for i in range(len(labels)):
        if rows[i, 0] == 0:
            matrix[i, i] = 1.0
    return np.asarray(matrix)


def adjusted_shares(
    predicted: Sequence[str], matrix: np.ndarray, labels: Sequence[str]
) -> np.ndarray:
    """Adjusted classify and count: the true label shares that, passed through the confusion
    ``matrix`` (see :func:`confusion`), best reproduce the observed shares of ``predicted``.
    Solved by non-negative least squares and normalised to sum to 1; with no predictions, zeros."""
    from scipy.optimize import nnls  # noqa: PLC0415

    observed = np.array([sum(p == label for p in predicted) for label in labels], dtype=float)
    if observed.sum() == 0:
        return observed
    observed /= observed.sum()
    solution, _ = nnls(matrix.T, observed)
    total = solution.sum()
    return np.asarray(solution / total) if total > 0 else observed
