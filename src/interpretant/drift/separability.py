"""Field separability: how well a classifier tells two fields' abstracts apart, per decade.

A second instrument for the questions the cross-perplexity rounds asked, sharing nothing with
the language models but the corpus. In each decade, abstracts found in more than one cell are
removed and every field contributes the same number of randomly drawn abstracts. A TF-IDF and
logistic-regression classifier then gives two readings:

- pairwise separability, the 5-fold cross-validated accuracy of telling two fields apart
  (0.5 is chance); falling accuracy between two fields means their language converged;
- assignments, the share of each field's held-out abstracts that a classifier over all fields
  sends to every field, within a decade or from one decade's classifier to another decade's
  abstracts, which shows which field moved.

The method and predictions are fixed in docs/predictions_cross_perplexity.md (round S).
"""

from __future__ import annotations

import itertools
import random
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from tqdm import tqdm

from interpretant.drift.perplexity import read_cell, remove_shared

PER_FIELD = 2_000
TRAIN_SHARE = 0.8
FOLDS = 5


def decade_samples(
    texts_dir: Path, decade: int, seed: int, per_field: int = PER_FIELD
) -> dict[str, list[str]]:
    """Draw ``per_field`` abstracts from every field of a decade, shared abstracts removed.

    Fields with fewer abstracts are left out. The draw depends only on the seed and the decade,
    and its order is random, so its first part can train and its last part can test.
    """
    cells = sorted(texts_dir.glob(f"*_{decade}.txt"))
    contents = remove_shared(dict(read_cell(path) for path in cells))
    rng = random.Random(f"{seed}:{decade}")
    return {
        field: rng.sample(lines, per_field)
        for field, lines in sorted(contents.items())
        if len(lines) >= per_field
    }


def strip_terms(text: str, terms: frozenset[str]) -> str:
    """Drop every token that is one of ``terms`` or a phrase with one as a component."""
    return " ".join(token for token in text.split() if terms.isdisjoint(token.split("_")))


def classifier() -> Pipeline:
    """TF-IDF over whitespace tokens (in at least 2 training abstracts), then L2 logistic
    regression with C = 1."""
    return make_pipeline(
        TfidfVectorizer(analyzer=str.split, min_df=2, sublinear_tf=True),
        LogisticRegression(C=1.0, max_iter=1_000),
    )


def pair_accuracy(texts_a: list[str], texts_b: list[str], seed: int, folds: int = FOLDS) -> float:
    """Mean stratified ``folds``-fold cross-validated accuracy of telling the two sets apart."""
    texts = texts_a + texts_b
    labels = [0] * len(texts_a) + [1] * len(texts_b)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(classifier(), texts, labels, cv=splitter, n_jobs=folds)
    return float(scores.mean())


def split(
    samples: dict[str, list[str]], train_share: float = TRAIN_SHARE
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Split every field's sample into its first ``train_share`` (train) and the rest (test)."""
    cut = {field: round(len(texts) * train_share) for field, texts in samples.items()}
    return (
        {field: texts[: cut[field]] for field, texts in samples.items()},
        {field: texts[cut[field] :] for field, texts in samples.items()},
    )


class FieldAssigner(Protocol):
    """Anything that assigns abstracts to fields: a fitted classifier, or the nearest language
    model of :mod:`interpretant.drift.direction`."""

    @property
    def classes_(self) -> Sequence[Any]: ...

    def predict(self, texts: list[str]) -> Sequence[Any]: ...


def fit_fields(train: dict[str, list[str]]) -> Pipeline:
    """Fit one classifier over every field in ``train``."""
    texts = [text for field_texts in train.values() for text in field_texts]
    labels = [field for field, field_texts in train.items() for _ in field_texts]
    return classifier().fit(texts, labels)


def assignment_shares(model: FieldAssigner, test: dict[str, list[str]]) -> list[dict[str, object]]:
    """For every field in ``test``, the share of its abstracts the model assigns to each field."""
    rows: list[dict[str, object]] = []
    for text_field, texts in test.items():
        assigned = Counter(str(label) for label in model.predict(texts))
        rows.extend(
            {
                "text_field": text_field,
                "assigned_field": str(field),
                "share": assigned[str(field)] / len(texts),
            }
            for field in model.classes_
        )
    return rows


def _ignore(message: str) -> None:
    del message


def separability_runs(  # noqa: PLR0913
    texts_dir: Path,
    decades: Iterable[int],
    seeds: Iterable[int],
    cross: tuple[int, int] = (1990, 2020),
    per_field: int = PER_FIELD,
    removed_terms: frozenset[str] = frozenset(),
    log: Callable[[str], None] = _ignore,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise accuracies and assignment shares for every seed and decade.

    With ``removed_terms``, the same samples are drawn and those terms are stripped from them
    (see :func:`strip_terms`) before any classifier sees them.

    Returns ``(pairs, assignments)``. ``pairs`` has one row per seed, decade and field pair.
    ``assignments`` has one row per seed, training decade, test decade, text field and
    assigned field: every decade's classifier on its own held-out abstracts, plus the
    ``cross[0]`` classifier on the ``cross[1]`` held-out abstracts.
    """
    decades = list(decades)
    pair_rows: list[dict[str, object]] = []
    assignment_rows: list[dict[str, object]] = []
    for seed in seeds:
        held_out: dict[int, dict[str, list[str]]] = {}
        models: dict[int, Pipeline] = {}
        for decade in decades:
            samples = {
                field: [strip_terms(text, removed_terms) for text in texts]
                for field, texts in decade_samples(texts_dir, decade, seed, per_field).items()
            }
            pairs = list(itertools.combinations(samples, 2))
            for field_a, field_b in tqdm(pairs, desc=f"  seed {seed} {decade}s", unit="pair"):
                accuracy = pair_accuracy(samples[field_a], samples[field_b], seed)
                pair_rows.append(
                    {
                        "seed": seed,
                        "decade": decade,
                        "field_a": field_a,
                        "field_b": field_b,
                        "accuracy": accuracy,
                    }
                )
            train, held_out[decade] = split(samples)
            models[decade] = fit_fields(train)
            assignment_rows.extend(
                {"seed": seed, "train_decade": decade, "test_decade": decade, **row}
                for row in assignment_shares(models[decade], held_out[decade])
            )
            log(f"  seed {seed} {decade}s: {len(samples)} fields, {len(pairs)} pairs")
        early, late = cross
        if early in models and late in held_out:
            assignment_rows.extend(
                {"seed": seed, "train_decade": early, "test_decade": late, **row}
                for row in assignment_shares(models[early], held_out[late])
            )
    return pd.DataFrame(pair_rows), pd.DataFrame(assignment_rows)
