"""Cross-perplexity between fields: how predictable one field's language is from another's.

Each cell gets a bigram language model with interpolated Kneser-Ney smoothing (discount
0.75); each model then scores held-out abstracts from every other field in the same decade.
The excess cross-entropy E(A -> B) is B's held-out cross-entropy under A's model minus under
B's own model, in bits per token: low means B's language is predictable from A's (Vilhena et
al. 2014). It is asymmetric, so E(A -> B) and E(B -> A) can differ.

Comparisons are kept fair within a decade: every qualifying field trains on the same number
of tokens, all models share one vocabulary, and held-out abstracts are never trained on. The
method and predictions are fixed in docs/predictions_cross_perplexity.md.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DISCOUNT = 0.75
HELD_OUT = 1_000
MIN_TRAINING_TOKENS = 200_000
MAX_TRAINING_TOKENS = 1_000_000
MIN_VOCABULARY_COUNT = 5
START, END, UNKNOWN = "<s>", "</s>", "<unk>"


class KneserNeyBigram:
    """Interpolated Kneser-Ney bigram model over a fixed vocabulary.

    P(w | v) = max(c(v, w) - d, 0) / c(v) + d * N(v .) / c(v) * P_cont(w), where N(v .) is the
    number of distinct words following v and P_cont(w) is the share of distinct bigram types
    ending in w. One is added to every continuation count so that no vocabulary word gets
    zero probability.
    """

    def __init__(self, vocabulary: set[str], discount: float = DISCOUNT) -> None:
        self.vocabulary = vocabulary | {END, UNKNOWN}
        self.discount = discount
        self.bigrams: dict[str, Counter[str]] = defaultdict(Counter)
        self.context_counts: Counter[str] = Counter()
        self.continuations: Counter[str] = Counter()
        self.bigram_types = 0

    def _tokens(self, document: str) -> list[str]:
        return [START] + [w if w in self.vocabulary else UNKNOWN for w in document.split()] + [END]

    def fit(self, documents: Iterable[str]) -> KneserNeyBigram:
        for document in documents:
            tokens = self._tokens(document)
            for previous, word in zip(tokens, tokens[1:], strict=False):
                self.bigrams[previous][word] += 1
        for previous, followers in self.bigrams.items():
            self.context_counts[previous] = sum(followers.values())
            for word in followers:
                self.continuations[word] += 1
        self.bigram_types = sum(self.continuations.values())
        return self

    def continuation_probability(self, word: str) -> float:
        return (self.continuations[word] + 1) / (self.bigram_types + len(self.vocabulary))

    def probability(self, word: str, previous: str) -> float:
        context = self.context_counts[previous]
        if context == 0:
            return self.continuation_probability(word)
        followers = self.bigrams[previous]
        seen = max(followers[word] - self.discount, 0) / context
        backoff = self.discount * len(followers) / context
        return seen + backoff * self.continuation_probability(word)

    def cross_entropy(self, documents: Iterable[str]) -> float:
        """Average bits per token over the documents, end-of-document tokens included."""
        bits, occurrences = self.word_bits(documents)
        total = sum(occurrences.values())
        return sum(bits.values()) / total if total else float("nan")

    def word_bits(self, documents: Iterable[str]) -> tuple[Counter[str], Counter[str]]:
        """Total bits and occurrences per predicted word (unknown words pooled as <unk>)."""
        bits: Counter[str] = Counter()
        occurrences: Counter[str] = Counter()
        for document in documents:
            tokens = self._tokens(document)
            for previous, word in zip(tokens, tokens[1:], strict=False):
                bits[word] -= math.log2(self.probability(word, previous))
                occurrences[word] += 1
        return bits, occurrences


@dataclass
class FieldText:
    field: str
    training_pool: list[str]
    held_out: list[str]

    @property
    def pool_tokens(self) -> int:
        return sum(len(line.split()) for line in self.training_pool)


def read_cell(path: Path) -> tuple[str, list[str]]:
    """Return a cell's field name and its non-empty abstracts."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return path.stem.rsplit("_", 1)[0], [line for line in lines if line]


def remove_shared(cells: dict[str, list[str]]) -> dict[str, list[str]]:
    """Drop every abstract that appears in more than one cell, from all of them.

    OpenAlex files subfields under their parent field (AI under computer science), so the
    same abstract can sit in two cells; a model trained on one cell would then have seen the
    other cell's held-out text.
    """
    presence: Counter[str] = Counter()
    for lines in cells.values():
        presence.update(set(lines))
    return {
        field: [line for line in lines if presence[line] == 1] for field, lines in cells.items()
    }


def split_cell(field: str, lines: list[str], held_out: int = HELD_OUT, seed: int = 42) -> FieldText:
    """Set aside ``held_out`` abstracts with a fixed seed; the rest is the training pool."""
    shuffled = list(lines)
    random.Random(seed).shuffle(shuffled)
    return FieldText(field, shuffled[held_out:], shuffled[:held_out])


def training_sample(pool: list[str], tokens: int) -> list[str]:
    """Take abstracts from a shuffled pool until ``tokens`` tokens are reached."""
    sample, total = [], 0
    for line in pool:
        if total >= tokens:
            break
        sample.append(line)
        total += len(line.split())
    return sample


def decade_cross_entropy(
    cells: list[Path],
    seed: int = 42,
    fixed_budget: int | None = None,
    drop_shared: bool = False,
) -> pd.DataFrame:
    """Score every qualifying field's held-out text with every qualifying field's model.

    By default every qualifying field trains on the smallest qualifying pool, capped at
    ``MAX_TRAINING_TOKENS``. With ``fixed_budget`` every field trains on exactly that many
    tokens and qualifies only if its pool is at least that large; with ``drop_shared``
    abstracts appearing in more than one cell are removed first.
    """
    columns = ["model_field", "text_field", "cross_entropy", "training_tokens"]
    decade = build_decade(cells, seed, fixed_budget, drop_shared)
    if decade is None:
        return pd.DataFrame(columns=columns)
    rows = [
        {
            "model_field": model_field,
            "text_field": field,
            "cross_entropy": model.cross_entropy(held_out),
            "training_tokens": decade.budget,
        }
        for model_field, model in decade.models.items()
        for field, held_out in decade.held_out.items()
    ]
    return pd.DataFrame(rows, columns=columns)


@dataclass
class DecadeModels:
    models: dict[str, KneserNeyBigram]
    held_out: dict[str, list[str]]
    budget: int


def build_decade(
    cells: list[Path],
    seed: int = 42,
    fixed_budget: int | None = None,
    drop_shared: bool = False,
) -> DecadeModels | None:
    """Train every qualifying field's model and keep its held-out abstracts (None if < 2)."""
    contents = dict(read_cell(path) for path in cells)
    if drop_shared:
        contents = remove_shared(contents)
    texts = [split_cell(field, lines, seed=seed) for field, lines in contents.items()]
    floor = fixed_budget or MIN_TRAINING_TOKENS
    qualifying = [text for text in texts if text.pool_tokens >= floor]
    if len(qualifying) < 2:
        return None
    budget = fixed_budget or min(MAX_TRAINING_TOKENS, min(t.pool_tokens for t in qualifying))
    samples = {text.field: training_sample(text.training_pool, budget) for text in qualifying}
    counts: Counter[str] = Counter()
    for sample in samples.values():
        for line in sample:
            counts.update(line.split())
    vocabulary = {word for word, count in counts.items() if count >= MIN_VOCABULARY_COUNT}
    return DecadeModels(
        models={
            field: KneserNeyBigram(vocabulary).fit(sample) for field, sample in samples.items()
        },
        held_out={text.field: text.held_out for text in qualifying},
        budget=budget,
    )


def add_excess(table: pd.DataFrame) -> pd.DataFrame:
    """Add each row's excess over the text field's own model, per decade (and seed)."""
    keys = ["seed", "decade", "text_field"] if "seed" in table else ["decade", "text_field"]
    own = table[table.model_field == table.text_field].set_index(keys)["cross_entropy"]
    result = table.copy()
    result["own_cross_entropy"] = own.reindex(pd.MultiIndex.from_frame(result[keys])).to_numpy()
    result["excess"] = result.cross_entropy - result.own_cross_entropy
    return result
