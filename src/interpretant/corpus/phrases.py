"""Phrase detection — merge multi-word terms into single tokens before training.

One phrase model is learned from a balanced sample (the same token budget from every
processed file, so the large modern cells do not decide the phrases of the 1850s) and
applied to every file, so a token means the same thing in every slice.

Two rules protect the targets in ``data/words_of_interest.txt``:

- a learned phrase that contains a single-word target (``network_analysis``) is dropped,
  so phrase merging cannot quietly take occurrences away from the target;
- phrase targets (``neural_network``) are always merged, even in cells where they are too
  rare to be learned.

Learning also repairs words broken across lines in OCR'd books: a learned pair whose halves
join into a frequent word (``lan guage``) is written as that word (``language``).
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS, FrozenPhrases, Phrases

CHUNK_TOKENS = 1_000


def split_targets(targets: Iterable[str]) -> tuple[set[str], set[str]]:
    """Split target words into (single words, phrase targets joined with ``_``)."""
    words = set(targets)
    phrases = {word for word in words if "_" in word}
    return words - phrases, phrases


def sample_chunks(path: Path, token_budget: int, seed: int = 42) -> list[list[str]]:
    """Return random chunks of ``CHUNK_TOKENS`` tokens from a file, up to ``token_budget``.

    Chunking makes books (one book per line) and abstracts (one abstract per line) comparable
    units, so every file contributes the same amount of text.
    """
    chunks = [
        tokens[start : start + CHUNK_TOKENS]
        for line in path.read_text(encoding="utf-8").splitlines()
        for tokens in [line.split()]
        for start in range(0, len(tokens), CHUNK_TOKENS)
    ]
    random.Random(seed).shuffle(chunks)
    sample: list[list[str]] = []
    total = 0
    for chunk in chunks:
        if total >= token_budget:
            break
        sample.append(chunk)
        total += len(chunk)
    return sample


@dataclass
class PhraseModel:
    """Learned phrases plus the subset that are broken words, written joined."""

    phrases: FrozenPhrases
    joins: dict[str, str]

    def apply(self, line: str) -> str:
        """Merge phrases in one preprocessed document."""
        return " ".join(self.joins.get(token, token) for token in self.phrases[line.split()])

    def save(self, path: Path) -> None:
        self.phrases.save(str(path))
        path.with_suffix(".joins.json").write_text(json.dumps(self.joins, indent=1), "utf-8")

    @classmethod
    def load(cls, path: Path) -> PhraseModel:
        joins = json.loads(path.with_suffix(".joins.json").read_text(encoding="utf-8"))
        return cls(FrozenPhrases.load(str(path)), joins)


def learn_phrases(
    chunks: Iterable[list[str]],
    protected: set[str],
    forced: set[str],
    min_count: int = 20,
    threshold: float = 0.5,
) -> PhraseModel:
    """Learn bigram phrases by normalised PMI, then apply the cleanup and target rules.

    Phrases with a one-letter part (``o_f``) are OCR noise and dropped. A phrase whose halves
    join into a word at least as frequent as the split pair is a broken word and kept as a
    repair: ``lan guage`` is rare next to ``language``, while ``may be`` outnumbers ``maybe``
    and stays two words.
    """
    learned = Phrases(
        chunks,
        min_count=min_count,
        threshold=threshold,
        scoring="npmi",
        connector_words=ENGLISH_CONNECTOR_WORDS,
    )
    frozen = learned.freeze()
    joins: dict[str, str] = {}
    for phrase in list(frozen.phrasegrams):
        if phrase in forced:
            continue
        parts = phrase.split("_")
        joined = "".join(parts)
        if any(len(part) <= 1 for part in parts):
            del frozen.phrasegrams[phrase]
        elif len(parts) == 2 and learned.vocab.get(joined, 0) >= max(
            min_count, learned.vocab.get(phrase, 0)
        ):
            joins[phrase] = joined
        elif protected & set(parts):
            del frozen.phrasegrams[phrase]
    for phrase in forced:
        frozen.phrasegrams[phrase] = math.inf
    return PhraseModel(frozen, joins)
