"""Review items as the judges see them: the answers each codebook allows, and the items rendered.

Both judges (``pipelines.judge_api``, through the Message Batches API, and
``pipelines.judge_claude_code``, through Claude Code's command line) send items rendered here, so
a codebook reads the same items whichever judges it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

Answers = Literal["usages", "comparison"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]
EFFORTS: tuple[Effort, ...] = ("low", "medium", "high", "xhigh", "max")
PART_SIZE = 50
REASONS = ["none", "other_word", "part_of_speech", "name", "span", "not_english"]
SCHEMAS: dict[str, dict[str, Any]] = {
    "usages": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["right", "wrong"]},
            "reason": {"type": "string", "enum": REASONS},
            "note": {"type": "string"},
        },
        "required": ["verdict", "reason", "note"],
        "additionalProperties": False,
    },
    "comparison": {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": ["A", "B", "both", "neither"]},
            "note": {"type": "string"},
        },
        "required": ["answer", "note"],
        "additionalProperties": False,
    },
}
ADJUDICATION = (
    "Two independent judges answered each item below differently; their answers follow the item, "
    "as judge 1 and judge 2. Settle each item yourself by the codebook: you may agree with either "
    "judge, and you owe neither of them anything. "
)


def render(item: dict[str, Any], answers: Answers) -> str:
    """The item as a judge's message; an adjudication item also shows both judges' answers."""
    lines = [f"Item id: {item['id']}"]
    if answers == "usages":
        lines += [
            f"Kind: {item['kind']}",
            f"Target word: {item['word']}",
            f"Parts of speech it is studied in: {', '.join(item['parts_of_speech'])}",
        ]
        if item.get("reason"):
            lines.append(f"The program left it out for: {item['reason']}")
        lines.append(f"Sentence: {item['sentence']}")
    else:
        lines += [
            f"Target word: {item['word']}",
            f"Parts of speech it is studied in: {', '.join(item['parts_of_speech'])}",
            f"Sentence: {item['sentence']}",
        ]
        for label in ("A", "B"):
            reading = item[label]
            lines.append(
                f"Reading {label}: part of speech {reading['pos']}, lemma {reading['lemma']}, "
                f"name {reading['name'] or 'none'}"
            )
    for judge in ("judge_1", "judge_2"):
        if judge in item:
            lines.append(
                f"{judge.replace('_', ' ').capitalize()} answered: {json.dumps(item[judge])}"
            )
    return "\n".join(lines)


def render_message(items: list[dict[str, Any]], answers: Answers) -> str:
    """Several items as one message, each rendered on its own.

    Items that carry two judges' answers (an adjudication list) get the adjudicator's instruction.
    """
    rendered = "\n\n---\n\n".join(render(item, answers) for item in items)
    settling = ADJUDICATION if any("judge_1" in item for item in items) else ""
    return (
        f"{settling}Judge each of the {len(items)} items below by the codebook, each on its own. "
        f"Give one answer per item, with its item id.\n\n{rendered}"
    )


def build_batch_schema(answers: Answers) -> dict[str, Any]:
    """The codebook's answer schema for a list of items, each answer carrying its item id."""
    answer = SCHEMAS[answers]
    entry = answer | {
        "properties": {"id": {"type": "string"}} | answer["properties"],
        "required": ["id", *answer["required"]],
    }
    return {
        "type": "object",
        "properties": {"answers": {"type": "array", "items": entry}},
        "required": ["answers"],
        "additionalProperties": False,
    }


def load_lines(path: Path) -> list[dict[str, Any]]:
    """The items of one JSON-lines file."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_items(source: Path) -> list[dict[str, Any]]:
    """The items of a batch folder (its ``batch_*.jsonl`` files) or of one JSON-lines file."""
    paths = sorted(source.glob("batch_*.jsonl")) if source.is_dir() else [source]
    return [item for path in paths for item in load_lines(path)]


def split_parts(source: Path) -> dict[str, list[dict[str, Any]]]:
    """The items in parts, one verdict file each: each batch file, or one file in slices of 50."""
    if source.is_dir():
        return {
            path.stem.removeprefix("batch_"): load_lines(path)
            for path in sorted(source.glob("batch_*.jsonl"))
        }
    items = load_lines(source)
    return {
        f"{start // PART_SIZE:02d}": items[start : start + PART_SIZE]
        for start in range(0, len(items), PART_SIZE)
    }
