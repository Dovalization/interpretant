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
    only as the word just after; a plain cue counts on either side. A cue written ``!word`` is an
    exclusion: when ``word`` appears within the window on either side, that sense's cues do not
    count, so the usage seeds no sense through them."""
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

    def fires(words: Sequence[str]) -> bool:
        if any(w[1:] in before or w[1:] in after for w in words if w.startswith("!")):
            return False
        return any(present(c) for c in words if not c.startswith("!"))

    found = [sense for sense, words in cues.items() if fires(words)]
    return found[0] if len(found) == 1 else None


def out_of_scope(usage: Usage, words: Sequence[str], window: int = 5) -> bool:
    """Whether one of ``words`` appears within ``window`` words of the target on either side: a
    usage outside its sense list's scope, left out of training, labelling and counts."""
    near = TOKEN.findall(usage.text[: usage.start].lower())[-window:]
    near += TOKEN.findall(usage.text[usage.end :].lower())[:window]
    return not set(words).isdisjoint(near)


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
    from sklearn.linear_model import LogisticRegression

    if len(set(seed_senses)) == 1:
        return [seed_senses[0]] * len(vectors)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    model.fit(normalised(seed_vectors), list(seed_senses))
    return [str(label) for label in model.predict(normalised(vectors))]


def sqrt_balanced(senses: Sequence[str]) -> dict[str, float]:
    """Class weights halfway, on a log scale, between none and balanced: the square root of the
    balanced weight. Thin senses still count for more, but uncertain usages are pulled to them
    less than under balanced weights (round U)."""
    counts: dict[str, int] = {}
    for sense in senses:
        counts[sense] = counts.get(sense, 0) + 1
    total, kinds = len(senses), len(counts)
    return {sense: float((total / (kinds * n)) ** 0.5) for sense, n in counts.items()}


def cue_first(
    usages: Sequence[Usage], cues: Mapping[str, Sequence[str]], predicted: Sequence[str]
) -> list[str]:
    """Each usage's label: the sense its cues name, where they name exactly one, and otherwise
    the classifier's prediction (round U)."""
    return [cue_sense(usage, cues) or guess for usage, guess in zip(usages, predicted, strict=True)]


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
    from scipy.optimize import nnls

    observed = np.array([sum(p == label for p in predicted) for label in labels], dtype=float)
    if observed.sum() == 0:
        return observed
    observed /= observed.sum()
    solution, _ = nnls(matrix.T, observed)
    total = solution.sum()
    return np.asarray(solution / total) if total > 0 else observed


MOJIBAKE = {
    "â€œ": "\u201c",
    "â€\x9d": "\u201d",
    "â€•": "\u201d",
    "â€™": "\u2019",
    "â€˜": "\u2018",
    "â€“": "\u2013",
    "â€”": "\u2014",
    "â€¦": "\u2026",
    "Ã©": "\u00e9",
    "Ã¨": "\u00e8",
    "Ã¶": "\u00f6",
    "Ã¼": "\u00fc",
    "Ã¤": "\u00e4",
    "Â ": " ",
}
BREAKS = re.compile(
    r"\\n|[\n\r\t\x00-\x08\x0b\x0c\x0e-\x1f\u00a0\u200b\u200c\u200d\ufeff\u2060\ufffd]"
)
SPACES = re.compile(r" {2,}")


def tidy_text(text: str) -> str:
    """Abstract text without extraction artefacts: literal ``\\n`` sequences, line breaks, tabs,
    control and invisible characters and the replacement character become spaces, soft hyphens
    rejoin the word they split, common mojibake is repaired, and runs of spaces collapse."""
    for broken, fixed in MOJIBAKE.items():
        text = text.replace(broken, fixed)
    text = re.sub(r"\u00ad\s?", "", text)
    return SPACES.sub(" ", BREAKS.sub(" ", text))


def tidy_usage(usage: Usage) -> Usage:
    """The usage with :func:`tidy_text` applied before and after the target, its span rebuilt."""
    left = tidy_text(usage.text[: usage.start])
    target = usage.text[usage.start : usage.end]
    right = tidy_text(usage.text[usage.end :])
    return Usage(left + target + right, len(left), len(left) + len(target))


def boilerplate(texts: Sequence[str], min_abstracts: int = 5) -> set[str]:
    """Sentences found in at least ``min_abstracts`` usages of a sample: publisher notices,
    affiliations and other text that is not an abstract's own."""
    from collections import Counter

    counts = Counter(texts)
    return {text for text, count in counts.items() if count >= min_abstracts}
