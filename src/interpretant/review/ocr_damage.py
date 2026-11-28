"""The junk gate's OCR damage score (`docs/junk_gate.md`, "OCR damage and the flags").

A sentence's score is the share of its lower-case words of 3 or more letters, the target left out,
that are unknown. A word is known when spaCy's dictionary knows it (``corpus.dictionary``) or when
it appears in the kept text of at least 20 kept works, so technical terms are not read as damage;
a dictionary share is the usual estimate of OCR quality without a reference text (van Strien et al.
2020, "Assessing the impact of OCR quality on downstream NLP tasks"). A sentence with fewer than 5
such words has no score. The same share over a record's whole kept text is one of the classifier's
surface features. Its check: how well the score separates the sentences the judges marked damaged
(the area under the ROC curve), and the score that best separates them.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score, roc_curve

from interpretant.corpus.dictionary import is_dictionary_word

# A word is a run of letters: digits, underscores and punctuation split words
WORD = re.compile(r"[^\W\d_]+")
# The score must separate the damaged sentences at least this well (area under the ROC curve)
AUC_BAR = 0.75
FREQUENCIES_SCHEMA = pl.Schema({"word": pl.String(), "works": pl.UInt32()})
# A character range of a text: its start and end
Span = tuple[int, int]


@dataclass(frozen=True)
class OcrCheck:
    """How the score separates the damaged sentences, and the cut that separates them best.

    ``best_cut`` maximises sensitivity plus specificity, a sentence at or above it called
    damaged, the highest such score on a tie; ``sensitivity`` and ``specificity`` are read at it.
    """

    sentences: int
    damaged: int
    auc: float
    best_cut: float
    sensitivity: float
    specificity: float

    @property
    def meets_bar(self) -> bool:
        """Whether the area under the ROC curve is at least 0.75."""
        return self.auc >= AUC_BAR


def find_lower_case_words(
    text: str,
    *,
    min_letters: int,
    within: Span | None = None,
    left_out: Sequence[Span] = (),
) -> list[str]:
    """The text's lower-case words of ``min_letters`` letters or more, in order.

    A word is a run of letters (``WORD``). Only the words inside ``within`` are
    read when it is given, and a word overlapping a span of ``left_out`` (a removed range, the
    target) is skipped.
    """
    start, end = within if within is not None else (0, len(text))
    words = []
    for match in WORD.finditer(text, start, end):
        word = match.group()
        if len(word) < min_letters or not word.islower():
            continue
        if any(low < match.end() and match.start() < high for low, high in left_out):
            continue
        words.append(word)
    return words


def count_document_frequencies(texts: Iterable[str], *, min_letters: int) -> pl.DataFrame:
    """For each lower-case word of ``min_letters`` letters or more, the texts it appears in.

    One row per word (``FREQUENCIES_SCHEMA``), sorted by word.
    """
    distinct = [
        word for text in texts for word in set(find_lower_case_words(text, min_letters=min_letters))
    ]
    counts = pl.Series("word", distinct, dtype=pl.String()).value_counts(name="works", sort=False)
    return counts.cast(FREQUENCIES_SCHEMA).sort("word")


def build_corpus_words(frequencies: Iterable[pl.DataFrame], *, min_works: int) -> frozenset[str]:
    """The words appearing in at least ``min_works`` texts, over every part's frequencies."""
    combined = pl.concat([pl.DataFrame(schema=FREQUENCIES_SCHEMA), *frequencies])
    totals = combined.group_by("word").agg(pl.col("works").sum())
    return frozenset(totals.filter(pl.col("works") >= min_works)["word"])


def count_unknown_words(words: Iterable[str], corpus_words: Collection[str]) -> int:
    """The words neither the corpus (``corpus_words``) nor spaCy's dictionary knows."""
    return sum(word not in corpus_words and not is_dictionary_word(word) for word in words)


def compute_ocr_check(scores: NDArray[np.float64], damaged: NDArray[np.bool_]) -> OcrCheck:
    """The score's area under the ROC curve for the damaged sentences, and its best cut.

    Raises:
        ValueError: if the sentences are all damaged or none is, where the curve is undefined.
    """
    if damaged.all() or not damaged.any():
        raise ValueError("the check needs both damaged and undamaged sentences")
    false_positive_rates, true_positive_rates, thresholds = roc_curve(
        damaged, scores, drop_intermediate=False
    )
    # The first threshold calls nothing damaged; the best cut is among the scores themselves
    best = 1 + int(np.argmax(true_positive_rates[1:] - false_positive_rates[1:]))
    return OcrCheck(
        sentences=int(damaged.size),
        damaged=int(damaged.sum()),
        auc=float(roc_auc_score(damaged, scores)),
        best_cut=float(thresholds[best]),
        sensitivity=float(true_positive_rates[best]),
        specificity=float(1 - false_positive_rates[best]),
    )
