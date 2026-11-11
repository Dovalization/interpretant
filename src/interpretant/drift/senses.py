"""Senses of a word by field and decade, from contextual embeddings of its occurrences.

The type-level models give a word one vector per cell, which mixes its senses. Here every
occurrence gets its own vector: a contextual encoder reads the sentence around it, and the
vector is the mean of its last layers over the word's subword tokens. All occurrences of a word,
from every field and decade, are clustered together, so a sense is the same set of uses in
every cell, and a cell's sense distribution is the share of its occurrences in each cluster
(Giulianelli et al. 2020). The number of senses is the k in ``SENSE_RANGE`` with the best
silhouette.

Occurrences come from the cased abstracts behind the cleaned cells, with abstracts found in
more than one field's cell removed: up to ``CAP`` abstracts per
cell that contain the word, one occurrence each, drawn per seed so the seed-noise rule of
:func:`interpretant.drift.direction.present` applies.
"""

from __future__ import annotations

import gzip
import json
import math
import random
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from interpretant.corpus.openalex import OpenAlexSource, cell_path, strip_boilerplate

CAP = 500
SENSE_RANGE = range(2, 11)
CONTEXT_WORDS = 100
SILHOUETTE_SAMPLE = 5_000
LAYERS = 4
NARROW_CHARS = 200

Abstracts = Mapping[tuple[str, int], Sequence[tuple[str, str]]]

_LABEL = re.compile(r"^\s*(?:abstract|summary)\b[\s:.\-–—]*", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"“])")
_TOKEN = re.compile(r"[a-z][a-z-]{2,}")
_URL = re.compile(r"https?://\S+|www\.\S+|\bdoi:\s*\S+|\b10\.\d{4,9}/\S+", re.IGNORECASE)
MAX_FOREIGN_SHARE = 0.2


@dataclass(frozen=True)
class Occurrence:
    """One use of a word: where it was found, and the context the encoder reads."""

    field: str
    decade: int
    work_id: str
    context: str
    start: int
    end: int


@dataclass(frozen=True)
class Sampling:
    """How occurrences are drawn: up to ``cap`` per cell, with or without replacement, the
    word alone or with its regular plural, and with or without round C′'s quality rules
    (abstracts mostly outside the Latin script skipped, URLs and DOIs removed)."""

    cap: int = CAP
    replace: bool = False
    plural: bool = False
    quality: bool = False


ROUND_C = Sampling()


def foreign_share(text: str) -> float:
    """The share of a text's letters outside the Latin script."""
    letters = [character for character in text if character.isalpha()]
    return sum(ord(character) > 0x024F for character in letters) / max(len(letters), 1)


def cased_abstracts(raw_dir: Path, field: str, decade: int) -> list[tuple[str, str]]:
    """``(work id, abstract)`` for every document of a cleaned cell, with case and punctuation
    kept, boilerplate stripped and a leading "Abstract" label removed."""
    kept = set(OpenAlexSource(raw_dir, field).cell(decade).ids)
    with gzip.open(cell_path(raw_dir, field, decade), "rt", encoding="utf-8") as cell:
        records = [json.loads(line) for line in cell]
    return [
        (str(record["id"]), _LABEL.sub("", strip_boilerplate(str(record["abstract"]))))
        for record in records
        if record["id"] in kept
    ]


def drop_shared(abstracts: Abstracts) -> dict[tuple[str, int], list[tuple[str, str]]]:
    """Remove every abstract that appears in more than one field's cell of the same decade."""
    cells: Counter[tuple[str, int]] = Counter(
        (work_id, decade)
        for (_, decade), documents in abstracts.items()
        for work_id, _ in documents
    )
    return {
        (field, decade): [
            (work_id, text) for work_id, text in documents if cells[(work_id, decade)] == 1
        ]
        for (field, decade), documents in abstracts.items()
    }


def word_pattern(word: str, plural: bool = False) -> re.Pattern[str]:
    """The word as a whole word, in any case, and with ``plural`` its regular plural too; a
    hyphenated compound (*self-attention*) counts."""
    suffix = "s?" if plural else ""
    return re.compile(rf"\b{re.escape(word)}{suffix}\b", re.IGNORECASE)


def context_around(
    text: str, start: int, end: int, max_words: int = CONTEXT_WORDS
) -> tuple[str, int, int]:
    """The sentence holding ``text[start:end]``, cut to ``max_words`` words around it when
    longer, with the occurrence's offsets in that context."""
    sentence_start = 0
    for boundary in _SENTENCE_END.finditer(text):
        if boundary.end() > start:
            break
        sentence_start = boundary.end()
    following = _SENTENCE_END.search(text, end)
    sentence_end = following.start() if following else len(text)
    words = list(re.finditer(r"\S+", text[sentence_start:sentence_end]))
    if len(words) > max_words:
        target = next(
            index for index, word in enumerate(words) if word.end() + sentence_start > start
        )
        first = min(max(target - max_words // 2, 0), len(words) - max_words)
        last = first + max_words - 1
        sentence_end = sentence_start + words[last].end()
        sentence_start += words[first].start()
    return text[sentence_start:sentence_end], start - sentence_start, end - sentence_start


def sample_occurrences(
    abstracts: Abstracts, word: str, seed: int, sampling: Sampling = ROUND_C
) -> list[Occurrence]:
    """The first occurrence of ``word`` in each of up to ``sampling.cap`` abstracts per cell
    that contain it: the first in a seeded random order, or, with ``sampling.replace``, as many
    as the cell has (up to the cap) drawn with replacement, so seeds differ in small cells."""
    pattern = word_pattern(word, sampling.plural)
    occurrences: list[Occurrence] = []
    for (field, decade), documents in sorted(abstracts.items()):
        order = list(documents)
        rng = random.Random(f"{seed}:{word}:{field}:{decade}")
        rng.shuffle(order)
        found: list[Occurrence] = []
        for work_id, raw in order:
            text = _URL.sub(" ", raw) if sampling.quality else raw
            if sampling.quality and foreign_share(text) > MAX_FOREIGN_SHARE:
                continue
            match = pattern.search(text)
            if match is None:
                continue
            context, start, end = context_around(text, match.start(), match.end())
            found.append(Occurrence(field, decade, work_id, context, start, end))
            if not sampling.replace and len(found) == sampling.cap:
                break
        size = min(len(found), sampling.cap)
        occurrences.extend(rng.choices(found, k=size) if sampling.replace else found)
    return occurrences


_WORD_TOKEN = re.compile(r"\w+")


def sample_many(
    abstracts: Abstracts, plan: Sequence[tuple[str, Sampling]], seed: int
) -> dict[str, list[Occurrence]]:
    """:func:`sample_occurrences` for many words in one pass over each cell, with the same
    result for every word: each abstract is cleaned and tokenised once, and each word's seeded
    order is walked only through the abstracts that contain it. Sampling with replacement is not
    supported, and each word must be a single run of word characters."""
    for word, sampling in plan:
        if sampling.replace or not _WORD_TOKEN.fullmatch(word):
            raise ValueError(f"sample_many cannot draw {word!r} with {sampling}")
    result: dict[str, list[Occurrence]] = {word: [] for word, _ in plan}
    for (field, decade), documents in sorted(abstracts.items()):
        prepared: dict[bool, list[tuple[str, set[str]] | None]] = {}
        for quality in {sampling.quality for _, sampling in plan}:
            rows: list[tuple[str, set[str]] | None] = []
            for _, raw in documents:
                text = _URL.sub(" ", raw) if quality else raw
                if quality and foreign_share(text) > MAX_FOREIGN_SHARE:
                    rows.append(None)
                else:
                    rows.append((text, set(_WORD_TOKEN.findall(text.lower()))))
            prepared[quality] = rows
        for word, sampling in plan:
            rows = prepared[sampling.quality]
            target = word.lower()
            order = list(range(len(documents)))
            random.Random(f"{seed}:{word}:{field}:{decade}").shuffle(order)
            pattern = word_pattern(word, sampling.plural)
            found = result[word]
            taken = 0
            for index in order:
                row = rows[index]
                if row is None:
                    continue
                text, tokens = row
                if target not in tokens and not (sampling.plural and target + "s" in tokens):
                    continue
                match = pattern.search(text)
                if match is None:
                    continue
                context, start, end = context_around(text, match.start(), match.end())
                found.append(Occurrence(field, decade, documents[index][0], context, start, end))
                taken += 1
                if taken == sampling.cap:
                    break
    return result


class Encoder(Protocol):
    """Gives every occurrence a vector."""

    def encode(self, occurrences: Sequence[Occurrence]) -> np.ndarray: ...


def span_mask(offsets: np.ndarray, start: int, end: int) -> np.ndarray:
    """Which tokens, by their ``(start, end)`` character offsets, overlap ``[start, end)``;
    special tokens (empty offsets) never do."""
    return (offsets[:, 0] < end) & (offsets[:, 1] > start) & (offsets[:, 1] > offsets[:, 0])


def narrowed(occurrence: Occurrence, chars: int = NARROW_CHARS) -> Occurrence:
    """The occurrence with its context cut to ``chars`` characters on either side, for a
    context the encoder truncates before reaching it (text without spaces between words)."""
    first = max(occurrence.start - chars, 0)
    context = occurrence.context[first : occurrence.end + chars]
    return Occurrence(
        occurrence.field,
        occurrence.decade,
        occurrence.work_id,
        context,
        occurrence.start - first,
        occurrence.end - first,
    )


class TransformerEncoder:
    """A Hugging Face encoder: an occurrence's vector is the mean of the last ``layers`` hidden
    layers over the subword tokens that overlap it."""

    def __init__(self, model_name: str, layers: int = LAYERS, batch_size: int = 32) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore[no-untyped-call]
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.layers = layers
        self.batch_size = batch_size

    def encode(self, occurrences: Sequence[Occurrence]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for first in range(0, len(occurrences), self.batch_size):
            vectors.extend(self._encode_batch(occurrences[first : first + self.batch_size]))
        return np.vstack(vectors)

    def _encode_batch(self, batch: Sequence[Occurrence], narrow: bool = True) -> list[np.ndarray]:
        """One vector per occurrence; an occurrence past the encoder's 512 tokens is encoded
        again from its :func:`narrowed` context."""
        encoded = self.tokenizer(
            [occurrence.context for occurrence in batch],
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping").numpy()
        with self.torch.no_grad():
            output = self.model(**encoded.to(self.device), output_hidden_states=True)
        hidden = self.torch.stack(output.hidden_states[-self.layers :]).mean(0)
        hidden_rows = hidden.float().cpu().numpy()
        vectors: list[np.ndarray] = []
        for row, occurrence in enumerate(batch):
            mask = span_mask(offsets[row], occurrence.start, occurrence.end)
            if mask.any():
                vectors.append(hidden_rows[row][mask].mean(axis=0))
            elif narrow:
                vectors.extend(self._encode_batch([narrowed(occurrence)], narrow=False))
            else:
                raise ValueError(f"no token covers the occurrence in {occurrence.work_id}")
        return vectors


def induce_senses(
    vectors: np.ndarray, seed: int, sense_range: Iterable[int] = SENSE_RANGE
) -> tuple[np.ndarray, np.ndarray]:
    """K-means on the length-normalised vectors for every k in ``sense_range``; the labels and
    centres of the k with the best silhouette (cosine, on a seeded sample), or of the one k
    when the range holds one."""
    normalised = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    candidates = list(sense_range)
    if len(candidates) == 1:
        model = KMeans(n_clusters=candidates[0], n_init=10, random_state=seed).fit(normalised)
        return model.labels_, model.cluster_centers_
    sample = min(len(normalised), SILHOUETTE_SAMPLE)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for senses in candidates:
        if senses >= len(normalised):
            break
        model = KMeans(n_clusters=senses, n_init=10, random_state=seed).fit(normalised)
        score = float(
            silhouette_score(
                normalised, model.labels_, metric="cosine", sample_size=sample, random_state=seed
            )
        )
        if best is None or score > best[0]:
            best = (score, model.labels_, model.cluster_centers_)
    if best is None:
        raise ValueError("too few occurrences to cluster")
    return best[1], best[2]


def sense_runs(
    abstracts: Abstracts,
    words: Iterable[str],
    seeds: Iterable[int],
    encoder: Encoder,
    sampling: Sampling = ROUND_C,
    sense_range: Iterable[int] = SENSE_RANGE,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Sample, encode and cluster every word's occurrences in every seed.

    An occurrence drawn in several seeds is encoded once. One row per seed, word and
    occurrence, with its sense, the number of senses and its cosine distance to its sense's
    centre (the nearest are a sense's examples).
    """
    rows: list[dict[str, object]] = []
    cache: dict[Occurrence, np.ndarray] = {}
    for word in words:
        for seed in seeds:
            occurrences = sample_occurrences(abstracts, word, seed, sampling)
            new = list(dict.fromkeys(o for o in occurrences if o not in cache))
            if new:
                cache.update(zip(new, encoder.encode(new), strict=True))
            vectors = np.vstack([cache[occurrence] for occurrence in occurrences])
            labels, centres = induce_senses(vectors, seed, sense_range)
            normalised = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            centre_norms = np.linalg.norm(centres, axis=1)
            distances = (
                1 - np.einsum("ij,ij->i", normalised, centres[labels]) / centre_norms[labels]
            )
            rows.extend(
                {
                    "seed": seed,
                    "word": word,
                    "field": occurrence.field,
                    "decade": occurrence.decade,
                    "work_id": occurrence.work_id,
                    "context": occurrence.context,
                    "start": occurrence.start,
                    "end": occurrence.end,
                    "sense": int(label),
                    "senses": len(centres),
                    "distance": float(distance),
                }
                for occurrence, label, distance in zip(occurrences, labels, distances, strict=True)
            )
            log(f"  {word}, seed {seed}: {len(occurrences):,} occurrences, {len(centres)} senses")
    return pd.DataFrame(rows)


def sense_shares(rows: pd.DataFrame) -> pd.DataFrame:
    """Each cell's share of its occurrences in every sense, from one seed and word's rows;
    senses a cell never uses get a zero."""
    counts = rows.groupby(["field", "decade", "sense"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=range(int(rows.senses.iloc[0])), fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0)
    return shares.reset_index().melt(
        id_vars=["field", "decade"], var_name="sense", value_name="share"
    )


def distribution(shares: pd.DataFrame, field: str, decade: int) -> np.ndarray:
    """One cell's sense distribution, ordered by sense."""
    cell = shares[(shares.field == field) & (shares.decade == decade)].sort_values("sense")
    return cell.share.to_numpy()


def jensen_shannon(first: np.ndarray, second: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits, from 0 (the same distribution) to 1 (disjoint)."""
    return float(jensenshannon(first, second, base=2) ** 2)


def context_words(rows: pd.DataFrame, sense: int, top: int = 10, min_count: int = 5) -> list[str]:
    """The words most over-represented in one sense's contexts against the other senses',
    by smoothed log ratio of relative frequencies, among words seen at least ``min_count``
    times in the sense. The word itself is left out."""
    word = str(rows.word.iloc[0]).lower()

    def counts(contexts: Iterable[str]) -> Counter[str]:
        found: Counter[str] = Counter()
        for context in contexts:
            found.update(token for token in _TOKEN.findall(context.lower()) if token != word)
        return found

    inside = counts(rows[rows.sense == sense].context)
    outside = counts(rows[rows.sense != sense].context)
    inside_total, outside_total = sum(inside.values()) + 1, sum(outside.values()) + 1
    scores = {
        token: math.log((count + 1) / inside_total) - math.log((outside[token] + 1) / outside_total)
        for token, count in inside.items()
        if count >= min_count
    }
    return sorted(scores, key=lambda token: (-scores[token], token))[:top]


def examples(rows: pd.DataFrame, sense: int, top: int = 10) -> list[str]:
    """The contexts nearest one sense's centre, from one seed and word's rows."""
    nearest = rows[rows.sense == sense].nsmallest(top, "distance")
    return nearest.context.tolist()
