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
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from tqdm import tqdm

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

    def document_cross_entropy(self, document: str) -> float:
        """Bits per token of one document, end token included: the per-abstract excess's unit."""
        bits, occurrences = self.word_bits([document])
        return sum(bits.values()) / sum(occurrences.values())

    def known_bits_per_token(self, document: str) -> float:
        """Mean bits per vocabulary word and end token of one document. Unknown words are not
        scored, but still serve as the next word's context."""
        tokens = self._tokens(document)
        bits = [
            -math.log2(self.probability(word, previous))
            for previous, word in zip(tokens, tokens[1:], strict=False)
            if word != UNKNOWN
        ]
        return sum(bits) / len(bits)

    def word_bits(self, documents: Iterable[str]) -> tuple[dict[str, float], Counter[str]]:
        """Total bits and occurrences per predicted word (unknown words pooled as <unk>)."""
        bits: defaultdict[str, float] = defaultdict(float)
        occurrences: Counter[str] = Counter()
        for document in documents:
            tokens = self._tokens(document)
            for previous, word in zip(tokens, tokens[1:], strict=False):
                bits[word] -= math.log2(self.probability(word, previous))
                occurrences[word] += 1
        return dict(bits), occurrences


class UnigramModel:
    """Add-one unigram model over the same fixed vocabulary: the lexical-composition control.

    A fall in excess cross-entropy that a unigram model reproduces is a change in which words
    a text uses; what the bigram model adds beyond it is phrase structure.
    """

    def __init__(self, vocabulary: set[str]) -> None:
        self.vocabulary = vocabulary | {END, UNKNOWN}
        self.counts: Counter[str] = Counter()
        self.total = 0

    def _tokens(self, document: str) -> list[str]:
        return [w if w in self.vocabulary else UNKNOWN for w in document.split()] + [END]

    def fit(self, documents: Iterable[str]) -> UnigramModel:
        for document in documents:
            self.counts.update(self._tokens(document))
        self.total = sum(self.counts.values())
        return self

    def probability(self, word: str) -> float:
        return (self.counts[word] + 1) / (self.total + len(self.vocabulary))

    def word_bits(self, documents: Iterable[str]) -> tuple[dict[str, float], Counter[str]]:
        bits: defaultdict[str, float] = defaultdict(float)
        occurrences: Counter[str] = Counter()
        for document in documents:
            for word in self._tokens(document):
                bits[word] -= math.log2(self.probability(word))
                occurrences[word] += 1
        return dict(bits), occurrences

    def cross_entropy(self, documents: Iterable[str]) -> float:
        bits, occurrences = self.word_bits(documents)
        total = sum(occurrences.values())
        return sum(bits.values()) / total if total else float("nan")

    def document_cross_entropy(self, document: str) -> float:
        bits, occurrences = self.word_bits([document])
        return sum(bits.values()) / sum(occurrences.values())


LanguageModel = KneserNeyBigram | UnigramModel


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
    order: int = 2,
) -> pl.DataFrame:
    """Score every qualifying field's held-out text with every qualifying field's model.

    By default every qualifying field trains on the smallest qualifying pool, capped at
    ``MAX_TRAINING_TOKENS``. With ``fixed_budget`` every field trains on exactly that many
    tokens and qualifies only if its pool is at least that large; with ``drop_shared``
    abstracts appearing in more than one cell are removed first.
    """
    decade = build_decade(cells, seed, fixed_budget, drop_shared, order)
    if decade is None:
        return pl.DataFrame(schema=CROSS_ENTROPY_SCHEMA)
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
    return pl.DataFrame(rows, schema=CROSS_ENTROPY_SCHEMA)


CROSS_ENTROPY_SCHEMA = {
    "model_field": pl.String,
    "text_field": pl.String,
    "cross_entropy": pl.Float64,
    "training_tokens": pl.Int64,
}
FALL_SCHEMA = {
    "model_field": pl.String,
    "text_field": pl.String,
    "fall": pl.Float64,
    "mix": pl.Float64,
    "rate": pl.Float64,
}


@dataclass
class DecadeModels:
    models: dict[str, LanguageModel]
    held_out: dict[str, list[str]]
    budget: int


def build_decade(
    cells: list[Path],
    seed: int = 42,
    fixed_budget: int | None = None,
    drop_shared: bool = False,
    order: int = 2,
) -> DecadeModels | None:
    """Train every qualifying field's model and keep its held-out abstracts (None if < 2).

    ``order`` 2 is the Kneser-Ney bigram model; 1 the add-one unigram control.
    """
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

    def model(sample: list[str]) -> LanguageModel:
        return (KneserNeyBigram(vocabulary) if order == 2 else UnigramModel(vocabulary)).fit(sample)

    return DecadeModels(
        models={field: model(sample) for field, sample in samples.items()},
        held_out={text.field: text.held_out for text in qualifying},
        budget=budget,
    )


def add_excess(table: pl.DataFrame) -> pl.DataFrame:
    """Add each row's excess over the text field's own model, per decade (and seed)."""
    keys = ["seed", "decade", "text_field"] if "seed" in table.columns else ["decade", "text_field"]
    own = table.filter(pl.col("model_field") == pl.col("text_field")).select(
        *keys, own_cross_entropy=pl.col("cross_entropy")
    )
    return table.join(own, on=keys, how="left", maintain_order="left").with_columns(
        excess=pl.col("cross_entropy") - pl.col("own_cross_entropy")
    )


@dataclass(frozen=True)
class FallDecomposition:
    """A fall in excess cross-entropy between two decades, split into two effects.

    ``mix`` is the part due to the text's word frequencies changing (the earlier per-word
    excess rates applied to the later frequencies); ``rate`` is the part due to the model's
    per-word excess rates changing at the later frequencies. ``fall = mix + rate``.
    """

    fall: float
    mix: float
    rate: float


def per_word_excess(
    model: LanguageModel, own: LanguageModel, held_out: list[str]
) -> tuple[dict[str, float], dict[str, int]]:
    """Per-word excess rate (bits per occurrence over the text's own model) and occurrences."""
    other_bits, counts = model.word_bits(held_out)
    own_bits, _ = own.word_bits(held_out)
    return {w: (other_bits[w] - own_bits[w]) / counts[w] for w in counts}, dict(counts)


def decompose_fall(
    early: DecadeModels, late: DecadeModels, model_field: str, text_field: str
) -> FallDecomposition:
    """Split the fall of E(model_field -> text_field) from ``early`` to ``late``.

    Words with no earlier rate (new in the later text) keep their later rate, so they count
    towards the mix effect at their actual cost.
    """
    rates_early, counts_early = per_word_excess(
        early.models[model_field], early.models[text_field], early.held_out[text_field]
    )
    rates_late, counts_late = per_word_excess(
        late.models[model_field], late.models[text_field], late.held_out[text_field]
    )
    n_early, n_late = sum(counts_early.values()), sum(counts_late.values())
    mean_early = sum(rates_early[w] * c for w, c in counts_early.items()) / n_early
    mean_late = sum(rates_late[w] * c for w, c in counts_late.items()) / n_late
    counterfactual = (
        sum(rates_early.get(w, rates_late[w]) * c for w, c in counts_late.items()) / n_late
    )
    return FallDecomposition(
        fall=mean_early - mean_late,
        mix=mean_early - counterfactual,
        rate=counterfactual - mean_late,
    )


def decompose_decades(
    early: DecadeModels, late: DecadeModels, text_fields: Iterable[str] | None = None
) -> pl.DataFrame:
    """Decompose the fall for every model x text pair qualifying in both decades."""
    fields = sorted(set(early.models) & set(late.models))
    texts = [t for t in fields if text_fields is None or t in set(text_fields)]
    rows = [
        {"model_field": model_field, "text_field": text_field, **vars(d)}
        for text_field in texts
        for model_field in fields
        if model_field != text_field
        for d in [decompose_fall(early, late, model_field, text_field)]
    ]
    return pl.DataFrame(rows, schema=FALL_SCHEMA)


def cells_by_decade(texts_dir: Path) -> dict[int, list[Path]]:
    """Group the ``<field>_<decade>.txt`` cells in ``texts_dir`` by decade, in order."""
    by_decade: dict[int, list[Path]] = {}
    for path in sorted(texts_dir.glob("*_*.txt")):
        by_decade.setdefault(int(path.stem.rsplit("_", 1)[1]), []).append(path)
    return dict(sorted(by_decade.items()))


def _ignore(message: str) -> None:
    del message


def _labelled(table: pl.DataFrame, **labels: int) -> pl.DataFrame:
    """Prepend integer label columns (seed, decade, ...) in the order given."""
    columns = [pl.lit(value, dtype=pl.Int64).alias(name) for name, value in labels.items()]
    return table.select(*columns, pl.all())


def cross_entropy_runs(  # noqa: PLR0913
    texts_dir: Path,
    seeds: Iterable[int],
    fixed_budget: int | None = None,
    drop_shared: bool = False,
    order: int = 2,
    log: Callable[[str], None] = _ignore,
) -> pl.DataFrame:
    """Score every decade's fields against each other once per seed, with excess added.

    One row per seed, decade, model field and text field (see :func:`decade_cross_entropy`).
    """
    frames = []
    for seed in seeds:
        for decade, cells in tqdm(cells_by_decade(texts_dir).items(), desc=f"  seed {seed}"):
            table = decade_cross_entropy(cells, seed, fixed_budget, drop_shared, order)
            frames.append(_labelled(table, seed=seed, decade=decade))
            if len(table):
                log(
                    f"  seed {seed} {decade}s: {table['text_field'].n_unique()} fields, "
                    f"{table.select(pl.col('training_tokens').max()).item():,} tokens each"
                )
    return add_excess(pl.concat(frames))


def decompose_runs(  # noqa: PLR0913
    texts_dir: Path,
    early: int,
    late: int,
    seeds: Iterable[int],
    fixed_budget: int,
    order: int = 2,
    text_fields: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Decompose every pair's fall from ``early`` to ``late`` once per seed, with shared
    abstracts removed. Raises ValueError if a decade has fewer than two qualifying fields."""
    frames = []
    for seed in tqdm(list(seeds), desc="  seeds", unit="seed"):
        early_models, late_models = (
            build_decade(sorted(texts_dir.glob(f"*_{decade}.txt")), seed, fixed_budget, True, order)
            for decade in (early, late)
        )
        if early_models is None or late_models is None:
            raise ValueError("a decade has fewer than two qualifying fields")
        table = decompose_decades(early_models, late_models, text_fields)
        frames.append(_labelled(table, seed=seed, early=early, late=late))
    return pl.concat(frames)
