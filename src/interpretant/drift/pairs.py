"""Meaning by usage pairs: how far apart a word's usages are, judged pair by pair.

The sense rounds clustered usages into senses and failed calibration; the literature ranks that
configuration, a general encoder and clustering, the weakest (docs/related_work.md). Here a
word's usages are encoded by XL-LEXEME (Cassotti et al. 2023), an encoder trained on
word-in-context judgements, and two sets of usages are compared by the average cosine distance
over all pairs drawn across them (APD; Periti and Tahmasebi 2024), with no clustering step.

XL-LEXEME is XLM-RoBERTa large fine-tuned with the target word wrapped in ``<t>`` and ``</t>``
markers and mean-pooled over the whole sequence. :class:`XLLexeme` reimplements its input
preparation (``tokenize_sentence``, ``center_sentence`` and ``tokenize_batch`` in the authors'
``WordTransformer``) on the published weights, so the authors' package is not needed.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

XL_LEXEME = "pierluigic/xl-lexeme"
XL_LEXEME_REVISION = "c9f85c6f05b85bc7c5014b5c23846f4467476263"
MAX_SEQUENCE = 128


@dataclass(frozen=True)
class Usage:
    """One use of a word: its context and the character span of the word in it."""

    text: str
    start: int
    end: int


def marked_tokens(usage: Usage, tokenize: Callable[[str], list[str]]) -> tuple[list[str], int, int]:
    """The context tokenised as left context, ``<t>``, the word, ``</t>``, right context, and the
    token span from the opening marker to the closing one (the closing one excluded)."""
    left = usage.text[: usage.start]
    target = usage.text[usage.start : usage.end]
    right = usage.text[usage.end :]
    tokens = tokenize(left) if left else []
    first = len(tokens)
    tokens += tokenize("<t>") + tokenize(target) + tokenize("</t>")
    last = len(tokens)
    if right:
        tokens += tokenize(right)
    return tokens, first, last


def centred(tokens: list[str], first: int, last: int, limit: int) -> list[str]:
    """Cut ``tokens`` to about ``limit`` around ``tokens[first:last]``, as the authors' code does:
    equal room on either side, and a short side's room left unused."""
    left, middle, right = tokens[:first], tokens[first:last], tokens[last:]
    room = int((limit - len(middle)) / 2)
    overflow_left, overflow_right = len(left) - room, len(right) - room
    if overflow_left > 0 and overflow_right > 0:
        left, right = left[overflow_left:], right[: len(right) - overflow_right]
    elif overflow_left > 0:
        left = left[overflow_left:]
    else:
        right = right[: len(right) - overflow_right]
    return left + middle + right


class XLLexeme:
    """XL-LEXEME on the published weights: one 1,024-dimensional vector per usage."""

    def __init__(
        self,
        model_name: str = XL_LEXEME,
        revision: str = XL_LEXEME_REVISION,
        batch_size: int = 32,
    ) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)  # type: ignore[no-untyped-call]
        self.model = AutoModel.from_pretrained(model_name, revision=revision).to(self.device).eval()
        self.batch_size = batch_size

    def input_ids(self, usage: Usage) -> list[int]:
        tokens, first, last = marked_tokens(usage, self.tokenizer.tokenize)
        if len(tokens) + 2 > MAX_SEQUENCE:
            tokens = centred(tokens, first, last, MAX_SEQUENCE - 2)
        tokens = [self.tokenizer.cls_token, *tokens, self.tokenizer.sep_token]
        return list(self.tokenizer.convert_tokens_to_ids(tokens))

    def encode(self, usages: Sequence[Usage]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        pad = int(self.tokenizer.pad_token_id)
        for first in range(0, len(usages), self.batch_size):
            batch = [self.input_ids(usage) for usage in usages[first : first + self.batch_size]]
            width = max(len(ids) for ids in batch)
            ids = self.torch.tensor([row + [pad] * (width - len(row)) for row in batch])
            mask = self.torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in batch])
            with self.torch.no_grad():
                hidden = self.model(
                    input_ids=ids.to(self.device), attention_mask=mask.to(self.device)
                ).last_hidden_state
            weights = mask.to(self.device).unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1e-9)
            vectors.append(pooled.float().cpu().numpy())
        return np.vstack(vectors)


def average_pairwise_distance(first: np.ndarray, second: np.ndarray) -> float:
    """The mean cosine distance over every pair of one vector from ``first`` and one from
    ``second``."""
    first = first / np.linalg.norm(first, axis=1, keepdims=True)
    second = second / np.linalg.norm(second, axis=1, keepdims=True)
    return float(1 - (first @ second.T).mean())


def prototype_distance(first: np.ndarray, second: np.ndarray) -> float:
    """The cosine distance between the mean vectors of ``first`` and ``second``."""
    a, b = first.mean(axis=0), second.mean(axis=0)
    return float(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def dwug_usages(root: Path) -> dict[str, dict[int, list[Usage]]]:
    """A DWUG release's usages by word and grouping (period), from ``data/<word>/uses.csv``."""
    usages: dict[str, dict[int, list[Usage]]] = {}
    for path in sorted((root / "data").glob("*/uses.csv")):
        with path.open(encoding="utf-8") as table:
            for row in csv.DictReader(table, delimiter="\t", quoting=csv.QUOTE_NONE):
                start, end = (int(index) for index in row["indexes_target_token"].split(":"))
                usages.setdefault(row["lemma"], {}).setdefault(int(row["grouping"]), []).append(
                    Usage(row["context"], start, end)
                )
    return usages


def dwug_graded_change(root: Path) -> dict[str, float]:
    """A DWUG release's graded change score per word, from ``stats/opt/stats_groupings.csv``."""
    with (root / "stats" / "opt" / "stats_groupings.csv").open(encoding="utf-8") as table:
        return {
            row["lemma"]: float(row["change_graded"])
            for row in csv.DictReader(table, delimiter="\t")
        }
