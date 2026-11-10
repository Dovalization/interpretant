"""Meaning by sense labels: a language model gives each usage one sense from a fixed list.

The usage-pair measures respond to a change of context whether or not the sense changed
(docs/diagnostics.md, check 1b). Here each usage is labelled with one sense from a list written
for its word beforehand, and a set of usages is described by its share of each sense, so a change
of context alone should not move it (docs/plan_sense_labels.md).

Labels come from the Claude Code CLI in headless mode, one call per block of usages of one word.
Blocks mix the word's usages from every cell in a seeded order, so a usage's place in its block
says nothing about its cell, and a share of usages is labelled twice, in another block, to
measure the labels' consistency.
"""

from __future__ import annotations

import json
import random
import subprocess
import tomllib
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic_settings import BaseSettings, SettingsConfigDict
from scipy.spatial.distance import jensenshannon

from interpretant.drift.pairs import Usage, centred

OTHER = "other"
UNCLEAR = "unclear"
FAILED = "failed"
REPEAT_SUFFIX = "#repeat"
SYSTEM_PROMPT = (
    "You label the sense of a word in context. You are given a word, a numbered list of its "
    "senses, and numbered usages in which the word is marked with ⟦ and ⟧. For each usage, give "
    "the one sense from the list that the marked word has in that usage. Use 'other' when the "
    "marked word has a sense that is not on the list, and 'unclear' only when the context does "
    "not allow you to tell. Judge each usage on its own context alone."
)


@dataclass(frozen=True)
class Sense:
    """One entry of a sense list: a short identifier, a definition and an invented example."""

    key: str
    definition: str
    example: str


@dataclass(frozen=True)
class SenseList:
    """The senses a word's usages are labelled with; ``other`` and ``unclear`` are always added."""

    word: str
    senses: tuple[Sense, ...]

    @property
    def labels(self) -> list[str]:
        return [sense.key for sense in self.senses] + [OTHER, UNCLEAR]


def load_sense_lists(path: Path) -> dict[str, SenseList]:
    """Sense lists from a TOML file: one table per entry, with ``word`` and ``[[senses]]``."""
    with path.open("rb") as source:
        table = tomllib.load(source)
    return {
        entry: SenseList(
            body["word"],
            tuple(Sense(s["key"], s["definition"], s["example"]) for s in body["senses"]),
        )
        for entry, body in table.items()
    }


@dataclass(frozen=True)
class Item:
    """A usage to label, under a key that identifies it in its run."""

    key: str
    usage: Usage


def shown(usage: Usage, limit: int = 100) -> str:
    """The usage's context cut to about ``limit`` words around the word, the word marked."""
    left = usage.text[: usage.start].split()
    right = usage.text[usage.end :].split()
    tokens = [*left, f"⟦{usage.text[usage.start : usage.end]}⟧", *right]
    return " ".join(centred(tokens, len(left), len(left) + 1, limit))


def blocks(
    items: Sequence[Item], size: int, repeat_share: float, seed: int | str
) -> list[list[Item]]:
    """``items`` in a seeded order, cut into blocks of ``size``, with a share of them repeated
    (key suffixed ``#repeat``) at a random place in a block other than their own."""
    rng = random.Random(seed)
    order = list(items)
    rng.shuffle(order)
    result = [order[i : i + size] for i in range(0, len(order), size)]
    if len(result) < 2:
        return result
    home = {item.key: index for index, block in enumerate(result) for item in block}
    for item in rng.sample(order, round(repeat_share * len(order))):
        target = rng.choice([i for i in range(len(result)) if i != home[item.key]])
        result[target].insert(
            rng.randrange(len(result[target]) + 1), Item(item.key + REPEAT_SUFFIX, item.usage)
        )
    return result


def prompt(senses: SenseList, block: Sequence[Item]) -> str:
    lines = [f"Word: {senses.word}", "", "Senses:"]
    lines += [f"- {s.key}: {s.definition} (Example: {s.example})" for s in senses.senses]
    lines += [
        f"- {OTHER}: a sense of the word that is not on this list",
        f"- {UNCLEAR}: the context does not allow you to tell",
        "",
        "Usages:",
    ]
    lines += [f"{number}. {shown(item.usage)}" for number, item in enumerate(block, start=1)]
    return "\n".join(lines)


def schema(senses: SenseList) -> dict[str, object]:
    label = {"type": "string", "enum": senses.labels}
    entry = {
        "type": "object",
        "properties": {"item": {"type": "integer"}, "label": label},
        "required": ["item", "label"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"labels": {"type": "array", "items": entry}},
        "required": ["labels"],
        "additionalProperties": False,
    }


def parsed(response: dict[str, object], senses: SenseList, size: int) -> list[str] | None:
    """The block's labels in item order, or None unless every item has exactly one valid label."""
    output = response.get("structured_output")
    if response.get("is_error") or not isinstance(output, dict):
        return None
    entries = output.get("labels")
    if not isinstance(entries, list):
        return None
    found: dict[int, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            return None
        number, label = entry.get("item"), entry.get("label")
        if not isinstance(number, int) or number in found or label not in senses.labels:
            return None
        found[number] = str(label)
    if sorted(found) != list(range(1, size + 1)):
        return None
    return [found[number] for number in range(1, size + 1)]


@dataclass(frozen=True)
class Labeller:
    """Headless Claude Code calls with a fixed system prompt, no tools, settings or MCP servers."""

    model: str
    effort: str
    attempts: int = 3

    def command(self, senses: SenseList, block: Sequence[Item]) -> list[str]:
        return [
            "claude",
            "-p",
            "--model",
            self.model,
            "--effort",
            self.effort,
            "--tools",
            "",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--setting-sources",
            "",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(schema(senses)),
            "--system-prompt",
            SYSTEM_PROMPT,
            prompt(senses, block),
        ]

    def label(self, senses: SenseList, block: Sequence[Item], record: Path) -> dict[str, str]:
        """Labels by item key. A valid stored response is reused; otherwise the block is sent up
        to ``attempts`` times, every response is stored, and items still unlabelled are
        ``failed``."""
        if record.exists():
            stored = json.loads(record.read_text(encoding="utf-8"))
            labels = parsed(stored[-1], senses, len(block)) if stored else None
            if labels is not None:
                return dict(zip((item.key for item in block), labels, strict=True))
        responses: list[dict[str, object]] = []
        labels = None
        for _ in range(self.attempts):
            run = subprocess.run(
                self.command(senses, block), capture_output=True, text=True, check=False
            )
            try:
                response = json.loads(run.stdout)
            except json.JSONDecodeError:
                response = {"is_error": True, "stdout": run.stdout, "stderr": run.stderr}
            responses.append(response)
            labels = parsed(response, senses, len(block))
            if labels is not None:
                break
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(responses), encoding="utf-8")
        if labels is None:
            labels = [FAILED] * len(block)
        return dict(zip((item.key for item in block), labels, strict=True))


def label_blocks(
    labeller: Labeller,
    work: Sequence[tuple[SenseList, list[Item], Path]],
    workers: int,
) -> dict[str, str]:
    """Every block's labels by item key, ``workers`` calls at a time."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        found = pool.map(lambda job: labeller.label(*job), work)
        return {key: label for labels in found for key, label in labels.items()}


def distribution(labels: Sequence[str], senses: SenseList) -> np.ndarray:
    """Share of each listed sense and of ``other``; ``unclear`` and ``failed`` left out."""
    keys = [sense.key for sense in senses.senses] + [OTHER]
    counts = np.array([sum(label == key for label in labels) for key in keys], dtype=float)
    return counts / counts.sum() if counts.sum() else counts


def divergence(first: np.ndarray, second: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits, 0 to 1 (scipy returns its square root)."""
    return float(jensenshannon(first, second, base=2) ** 2)


class TypeSafeSettings(BaseSettings):
    """The TypeSafe API key, from the environment or ``.env``, as the OpenAlex key is read."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    typesafe_api_key: str
