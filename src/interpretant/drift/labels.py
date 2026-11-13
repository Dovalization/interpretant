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


class AnthropicSettings(BaseSettings):
    """The Anthropic API key, from ``.env`` only: exported in the shell, it would also switch
    Claude Code itself from the subscription to the API."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str


REFUSED = "refused"
SINGLE_SYSTEM_PROMPT = (
    "You label the sense of a word in context. You are given a word, a list of its senses, and "
    "one usage in which the word is marked with ⟦ and ⟧. Give the one sense from the list that "
    "the marked word has in that usage. Use 'other' when the marked word has a sense that is not "
    "on the list, and 'unclear' only when the context does not allow you to tell."
)


def single_prompt(senses: SenseList, usage: Usage) -> str:
    lines = [f"Word: {senses.word}", "", "Senses:"]
    lines += [f"- {s.key}: {s.definition} (Example: {s.example})" for s in senses.senses]
    lines += [
        f"- {OTHER}: a sense of the word that is not on this list",
        f"- {UNCLEAR}: the context does not allow you to tell",
        "",
        f"Usage: {shown(usage)}",
    ]
    return "\n".join(lines)


def label_schema(senses: SenseList) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {"label": {"type": "string", "enum": senses.labels}},
        "required": ["label"],
        "additionalProperties": False,
    }


def single_label(result: dict[str, object], senses: SenseList) -> str:
    """The label in one stored batch result: ``refused``, ``failed`` or a label from the list."""
    if result.get("type") != "succeeded":
        return FAILED
    if result.get("stop_reason") == "refusal":
        return REFUSED
    try:
        label = json.loads(str(result.get("text")))["label"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return FAILED
    return str(label) if label in senses.labels else FAILED


@dataclass(frozen=True)
class BatchLabeller:
    """One usage per request through the Message Batches API, at half the standard price.

    A run's batch identifier and its request-to-item map are stored before polling, and every
    result is stored when the batch ends, so an interrupted run resumes and a finished one is
    never sent again. Errored requests are not retried; their items are ``failed``.
    """

    model: str
    effort: str
    max_tokens: int = 2048
    poll_seconds: float = 30.0

    def params(self, senses: SenseList, item: Item) -> dict[str, object]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SINGLE_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": single_prompt(senses, item.usage)}],
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": label_schema(senses)},
            },
        }

    def label(self, work: Sequence[tuple[SenseList, Item]], record: Path) -> dict[str, str]:
        """Labels by item key; ``record`` is a directory holding the run's batch and results."""
        senses_of = {f"r{index}": senses for index, (senses, _) in enumerate(work)}
        key_of = {f"r{index}": item.key for index, (_, item) in enumerate(work)}
        requests: list[dict[str, object]] = [
            {"custom_id": custom_id, "params": self.params(senses, item)}
            for custom_id, (senses, item) in zip(senses_of, work, strict=True)
        ]
        stored = run_batch(
            requests, record, {"model": self.model, "keys": key_of}, self.poll_seconds
        )
        found = {r["custom_id"]: single_label(r, senses_of[str(r["custom_id"])]) for r in stored}
        return {key_of[custom_id]: found.get(custom_id, FAILED) for custom_id in key_of}


def run_batch(
    requests: Sequence[dict[str, object]],
    record: Path,
    meta: dict[str, object],
    poll_seconds: float = 30.0,
) -> list[dict[str, object]]:
    """Send ``requests`` as one Message Batch and return every result as stored.

    The batch's identifier and ``meta`` are stored in ``record`` before polling and every result
    when the batch ends, so an interrupted run resumes and a finished one is never sent again.
    A stored result has the request's ``custom_id``, its ``type`` and, when it succeeded, its
    ``stop_reason``, ``text``, token counts and ``model``.
    """
    import time

    import anthropic

    results_path = record / "results.jsonl"
    if not results_path.exists():
        client = anthropic.Anthropic(api_key=AnthropicSettings().anthropic_api_key)
        batch_path = record / "batch.json"
        if batch_path.exists():
            batch_id = json.loads(batch_path.read_text(encoding="utf-8"))["id"]
        else:
            batch = client.messages.batches.create(requests=requests)  # type: ignore[arg-type]
            batch_id = batch.id
            record.mkdir(parents=True, exist_ok=True)
            batch_path.write_text(json.dumps({"id": batch_id, **meta}), encoding="utf-8")
        while client.messages.batches.retrieve(batch_id).processing_status != "ended":
            time.sleep(poll_seconds)
        lines = []
        for result in client.messages.batches.results(batch_id):
            stored: dict[str, object] = {"custom_id": result.custom_id, "type": result.result.type}
            if result.result.type == "succeeded":
                message = result.result.message
                stored["stop_reason"] = message.stop_reason
                stored["text"] = next((b.text for b in message.content if b.type == "text"), "")
                stored["input_tokens"] = message.usage.input_tokens
                stored["output_tokens"] = message.usage.output_tokens
                stored["model"] = message.model
            lines.append(json.dumps(stored))
        results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]


JEV_YES, JEV_NO = 0.7, 0.3


def jev_label(answer: dict[str, object], senses: SenseList) -> str:
    """A stored JEV answer as a label: the top listed sense when the companion question says a
    listed sense fits (above 0.7), ``other`` when it says none does (below 0.3), else
    ``unclear``; ``failed`` when the request failed."""
    if "error" in answer:
        return FAILED
    listed = float(answer["listed"])  # type: ignore[arg-type]
    probabilities: dict[str, float] = answer["probabilities"]  # type: ignore[assignment]
    if listed < JEV_NO:
        return OTHER
    if listed <= JEV_YES:
        return UNCLEAR
    top = max(probabilities, key=lambda key: probabilities[key])
    return top if top in senses.labels else FAILED


@dataclass(frozen=True)
class JevLabeller:
    """One usage per request to TypeSafe's JEV: a Choice among the listed senses, each described
    by its definition and example, and a companion yes-or-no question on whether any listed
    sense fits, as TypeSafe's guidance recommends over a "none of these" option.

    Each answer is stored as it arrives, so an interrupted run resumes. Transient failures that
    outlast the SDK's retries are stored as errors and read as ``failed``; authentication and
    request errors are raised.
    """

    model: str
    workers: int = 8

    def questions(self, senses: SenseList) -> dict[str, object]:
        from typesafe_sdk import Choice, Noul

        listed = "; ".join(f"{s.key}: {s.definition}" for s in senses.senses)
        return {
            "sense": Choice(
                instructions="Which sense does the word marked with ⟦ and ⟧ in `usage` have?",
                criteria={s.key: f"{s.definition}. Example: {s.example}" for s in senses.senses},
            ),
            "listed": Noul(
                instructions=(
                    "Does the word marked with ⟦ and ⟧ in `usage` have one of these senses: "
                    f"{listed}?"
                ),
                criteria={
                    "true": "The marked word is used in one of the listed senses.",
                    "false": (
                        "The marked word is used in a sense that none of the listed senses "
                        "describes."
                    ),
                },
            ),
        }

    def label(self, work: Sequence[tuple[SenseList, Item]], record: Path) -> dict[str, str]:
        """Labels by item key; ``record`` is a directory with one stored answer per item."""
        import hashlib

        import typesafe_sdk

        transient = (
            typesafe_sdk.TypeSafeAPIConnectionError,
            typesafe_sdk.TypeSafeAPITimeoutError,
            typesafe_sdk.TypeSafeInternalServerError,
            typesafe_sdk.TypeSafeRateLimitError,
        )
        client = typesafe_sdk.TypeSafeClient(
            api_key=TypeSafeSettings().typesafe_api_key,
            model=self.model,
        )
        record.mkdir(parents=True, exist_ok=True)

        def one(job: tuple[SenseList, Item]) -> tuple[str, str]:
            senses, item = job
            path = record / f"{hashlib.sha256(item.key.encode()).hexdigest()[:16]}.json"
            if path.exists():
                return item.key, jev_label(json.loads(path.read_text(encoding="utf-8")), senses)
            try:
                response = client.system_one(
                    {"word": senses.word, "usage": shown(item.usage)},
                    self.questions(senses),  # type: ignore[arg-type]
                    model=self.model,
                )
                answer: dict[str, object] = {
                    "key": item.key,
                    "model": response.model,
                    "listed": response.nouls["listed"].noul,
                    "probabilities": response.choices["sense"].probabilities,
                    "input_tokens": response.usage.input_tokens,
                }
            except transient as error:
                answer = {"key": item.key, "error": repr(error)}
            path.write_text(json.dumps(answer), encoding="utf-8")
            return item.key, jev_label(answer, senses)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return dict(pool.map(one, work))
