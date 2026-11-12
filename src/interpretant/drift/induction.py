"""Sense lists induced from the texts (docs/predictions_labels.md, round Q).

Evidence for a word's senses comes from methods that are not generative language models:

- substitutes: a masked language model's fillers for the pattern "target (or even [MASK])"
  (Amrami and Goldberg 2019), grouped into communities of substitutes that appear together
  (Eyal et al. 2022);
- the words found immediately before and after the target;
- the word's entries in Wiktionary.

A language model then consolidates the evidence into a list of senses, citing the evidence each
sense rests on (see ``scripts/score_labels_induced.py``). The substitute communities also give a
labeller with no generative model: a usage takes the sense its community is mapped to.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from interpretant.drift.pairs import Usage, centred

SUBSTITUTE_MODEL = "answerdotai/ModernBERT-large"
SUBSTITUTE_REVISION = "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13"
PATTERN = " (or even {mask})"
TOP_FILLERS = 20
SUFFIXES = ("", "s", "es", "d", "ed", "ing")
WORD = re.compile(r"[A-Za-z]+")


def stop_words() -> frozenset[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return frozenset(ENGLISH_STOP_WORDS)


def patterned(usage: Usage, mask: str, limit: int = 100) -> str:
    """The usage's context cut to about ``limit`` words around the target, with the substitute
    pattern after the target."""
    left = usage.text[: usage.start].split()
    right = usage.text[usage.end :].split()
    target = usage.text[usage.start : usage.end] + PATTERN.format(mask=mask)
    tokens = [*left, target, *right]
    return " ".join(centred(tokens, len(left), len(left) + 1, limit))


def cleaned(fillers: Sequence[str], word: str, surface: str, stops: frozenset[str]) -> list[str]:
    """Fillers lowercased, without non-words, short words, stop words or forms of the target."""
    forms = {word + suffix for suffix in SUFFIXES} | {surface.lower()}
    kept = []
    for filler in fillers:
        token = filler.strip().lower()
        if (
            token.isalpha()
            and len(token) >= 2
            and token not in stops
            and token not in forms
            and token not in kept
        ):
            kept.append(token)
    return kept


class Substituter:
    """ModernBERT-large at a pinned revision: the most likely fillers of the pattern's mask."""

    def __init__(self, batch_size: int = 16) -> None:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            SUBSTITUTE_MODEL, revision=SUBSTITUTE_REVISION
        )
        self.model = (
            AutoModelForMaskedLM.from_pretrained(SUBSTITUTE_MODEL, revision=SUBSTITUTE_REVISION)
            .to(self.device)
            .eval()
        )
        self.batch_size = batch_size

    def fillers(self, usages: Sequence[Usage]) -> list[list[str]]:
        mask = str(self.tokenizer.mask_token)
        result: list[list[str]] = []
        for start in range(0, len(usages), self.batch_size):
            texts = [patterned(u, mask) for u in usages[start : start + self.batch_size]]
            encoded = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
            with self.torch.no_grad():
                logits = self.model(**encoded).logits
            for row in range(len(texts)):
                position = int(
                    (encoded.input_ids[row] == self.tokenizer.mask_token_id).nonzero()[0, 0]
                )
                top = self.torch.topk(logits[row, position], TOP_FILLERS).indices.tolist()
                result.append([self.tokenizer.decode([index]) for index in top])
        return result


@dataclass(frozen=True)
class Community:
    """A group of substitutes that tend to appear together, and its most frequent members."""

    name: str
    members: frozenset[str]
    top: tuple[str, ...]


def communities(
    substitutes: Sequence[Sequence[str]], min_usages: int = 3, seed: int = 2026
) -> list[Community]:
    """Louvain communities of the graph whose nodes are substitutes in at least ``min_usages``
    usages and whose edges count the usages two substitutes share; largest first, named C1, C2…"""
    import networkx as nx

    frequency = Counter(token for found in substitutes for token in set(found))
    nodes = {token for token, count in frequency.items() if count >= min_usages}
    graph = nx.Graph()
    graph.add_nodes_from(sorted(nodes))
    for found in substitutes:
        present = sorted(set(found) & nodes)
        for index, first in enumerate(present):
            for second in present[index + 1 :]:
                weight = graph.get_edge_data(first, second, {"weight": 0})["weight"]
                graph.add_edge(first, second, weight=weight + 1)
    groups = nx.community.louvain_communities(graph, weight="weight", resolution=1, seed=seed)
    ordered = sorted(groups, key=lambda group: (-len(group), sorted(group)))
    return [
        Community(
            f"C{index}",
            frozenset(group),
            tuple(sorted(group, key=lambda token: (-frequency[token], token))[:10]),
        )
        for index, group in enumerate(ordered, start=1)
    ]


def placed(found: Sequence[str], groups: Sequence[Community]) -> str | None:
    """The community holding most of a usage's substitutes (ties to the earlier one), or None."""
    counts = [len(set(found) & group.members) for group in groups]
    if not counts or max(counts) == 0:
        return None
    return groups[counts.index(max(counts))].name


def kept_communities(
    substitutes: Sequence[Sequence[str]], groups: Sequence[Community], min_share: float = 0.05
) -> list[Community]:
    """The communities that hold at least ``min_share`` of the usages."""
    placements = Counter(placed(found, groups) for found in substitutes)
    return [g for g in groups if placements[g.name] >= min_share * len(substitutes)]


def neighbours(
    usages: Sequence[Usage], stops: frozenset[str], top: int = 15
) -> list[tuple[str, int]]:
    """The most frequent words immediately before or after the target, as "before X"/"after X"."""
    counts: Counter[str] = Counter()
    for usage in usages:
        before = WORD.findall(usage.text[: usage.start])
        after = WORD.findall(usage.text[usage.end :])
        if before and before[-1].lower() not in stops:
            counts[f"before {before[-1].lower()}"] += 1
        if after and after[0].lower() not in stops:
            counts[f"after {after[0].lower()}"] += 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top]


def wiktionary_entries(path: Path, pos: str) -> list[str]:
    """Glosses of a kaikki.org per-word extract's senses of that part of speech, with topics."""
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("pos") != pos:
            continue
        for sense in record.get("senses", []):
            glosses = sense.get("glosses") or []
            if not glosses:
                continue
            topics = sense.get("topics") or []
            prefix = f"({', '.join(topics)}) " if topics else ""
            entries.append(prefix + glosses[-1])
    return entries
