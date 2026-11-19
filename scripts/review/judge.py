"""Judge review items with Claude through the Message Batches API (`docs/design_compute_hosts.md`).

`python scripts/review/judge.py ITEMS OUT.json --codebook PROMPT.md --answers usages|comparison
--model MODEL --effort LEVEL --max-dollars DOLLARS [--chunk N] [--max-tokens N]`

ITEMS is a folder of `batch_*.jsonl` files or one `.jsonl` file; an adjudication list's items carry
both judges' answers (`judge_1`, `judge_2`), which the request shows. Every item is its own request:
the codebook is the system prompt, byte-identical and cached for an hour; the item is the message;
the answer is constrained to the codebook's schema (structured output). Answers are merged into
OUT.json as each chunk ends, so a stopped run resumes where it left off.

**The cap.** Requests go out in chunks. A chunk is submitted only if the dollars spent so far plus
the chunk's worst case (every input token written to the cache, every request using its full
`--max-tokens`, at batch prices) stay within `--max-dollars`; so the run never spends more than
the cap. **The usage report** (input, cache-write, cache-read and output tokens, and their cost at
batch prices) is printed at the end, written to `OUT.manifest.json` with the batch ids, the model,
the effort and the codebook's sha256, and added to the run record. Raw responses are not kept.
The API key comes from the checkout's `.env` (`ANTHROPIC_API_KEY`), never from the shell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console

from interpretant.catalog.config import read_catalog
from interpretant.cli.progress import LiveProgress
from interpretant.pipelines.progress import SILENT, Progress, StageTimer
from interpretant.pipelines.runs import record_run

# US dollars per million tokens, from Anthropic's model table read 2026-11-03.
PRICES_READ = "2026-11-03"
PRICES = {
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.10},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20},
}
BATCH_DISCOUNT = 0.5
CACHE_WRITE_FACTOR = 2.0  # writing the one-hour cache costs twice the input price
CHARACTERS_PER_TOKEN = 3.0  # a deliberately low figure, so item estimates err high
POLL_SECONDS = 30
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
USAGE_FIELDS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)


class ApiKeySettings(BaseSettings):
    """The Anthropic API key, from ``.env`` only.

    Exported in the shell, it would also switch Claude Code itself from the subscription to the API.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str


@dataclass
class Spend:
    """Tokens used and dollars spent so far."""

    tokens: Counter[str] = field(default_factory=Counter)
    dollars: float = 0.0


def cost(usage: Any, model: str) -> float:  # noqa: ANN401 (an SDK usage record, faked in tests)
    """The dollars one request cost at batch prices."""
    price = PRICES[model]
    total = (
        (getattr(usage, "input_tokens", 0) or 0) * price["input"]
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
        * price["input"]
        * CACHE_WRITE_FACTOR
        + (getattr(usage, "cache_read_input_tokens", 0) or 0) * price["cache_read"]
        + (getattr(usage, "output_tokens", 0) or 0) * price["output"]
    )
    return total / 1e6 * BATCH_DISCOUNT


def worst_case(text: str, system_tokens: int, max_tokens: int, model: str) -> float:
    """The most one request can cost: all input written to the cache, all output tokens used."""
    price = PRICES[model]
    input_tokens = system_tokens + len(text) / CHARACTERS_PER_TOKEN + 50
    total = input_tokens * price["input"] * CACHE_WRITE_FACTOR + max_tokens * price["output"]
    return total / 1e6 * BATCH_DISCOUNT


def render(item: dict[str, Any], answers: str) -> str:
    """The item as the request's message."""
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


def load_items(source: Path) -> list[dict[str, Any]]:
    """The items of a batch folder or of one JSON-lines file."""
    paths = sorted(source.glob("batch_*.jsonl")) if source.is_dir() else [source]
    return [json.loads(line) for path in paths for line in path.read_text().splitlines() if line]


def request(
    item: dict[str, Any], options: argparse.Namespace, system: list[dict[str, Any]]
) -> dict[str, Any]:
    """One batch request for an item."""
    return {
        "custom_id": item["id"],
        "params": {
            "model": options.model,
            "max_tokens": options.max_tokens,
            "system": system,
            "output_config": {
                "effort": options.effort,
                "format": {"type": "json_schema", "schema": SCHEMAS[options.answers]},
            },
            "messages": [{"role": "user", "content": render(item, options.answers)}],
        },
    }


def answer(result: Any, answers: str) -> dict[str, Any] | None:  # noqa: ANN401 (an SDK batch result)
    """The parsed answer of a succeeded result, or None if it was refused or not valid."""
    message = result.result.message
    if message.stop_reason != "end_turn":
        return None
    text = next((block.text for block in message.content if block.type == "text"), "")
    try:
        parsed: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError:
        return None
    if answers == "usages" and parsed.get("reason") == "none":
        parsed["reason"] = None
    return parsed


def judge(
    items: list[dict[str, Any]],
    options: argparse.Namespace,
    client: Any,  # noqa: ANN401 (the SDK client, faked in tests)
    verdicts: dict[str, Any],
    *,
    progress: Progress = SILENT,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Judge every item not yet in ``verdicts`` within the cap; the manifest of the run."""
    codebook = options.codebook.read_text()
    system = [
        {"type": "text", "text": codebook, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    ]
    system_tokens = client.messages.count_tokens(
        model=options.model, system=system, messages=[{"role": "user", "content": "x"}]
    ).input_tokens
    spend, batches, failed = Spend(), [], {}
    pending = [item for item in items if item["id"] not in verdicts]
    retried: set[str] = set()
    progress.stage("judging", total=len(pending))
    stopped_at_cap = False
    while pending:
        chunk, pending = pending[: options.chunk], pending[options.chunk :]
        worst = sum(
            worst_case(render(i, options.answers), system_tokens, options.max_tokens, options.model)
            for i in chunk
        )
        if spend.dollars + worst > options.max_dollars:
            stopped_at_cap = True
            pending = chunk + pending
            break
        batch = client.messages.batches.create(
            requests=[request(i, options, system) for i in chunk]
        )
        batches.append(batch.id)
        while client.messages.batches.retrieve(batch.id).processing_status != "ended":
            sleep(POLL_SECONDS)
        by_id = {item["id"]: item for item in chunk}
        for result in client.messages.batches.results(batch.id):
            item = by_id[result.custom_id]
            if result.result.type != "succeeded":
                if result.custom_id not in retried:
                    retried.add(result.custom_id)
                    pending.append(item)
                else:
                    failed[result.custom_id] = result.result.type
                continue
            usage = result.result.message.usage
            spend.dollars += cost(usage, options.model)
            spend.tokens.update({name: getattr(usage, name, 0) or 0 for name in USAGE_FIELDS})
            parsed = answer(result, options.answers)
            if parsed is None:
                failed[result.custom_id] = f"no valid answer ({result.result.message.stop_reason})"
            else:
                verdicts[result.custom_id] = parsed
            progress.advance()
        options.out.write_text(json.dumps(verdicts, indent=1, ensure_ascii=False) + "\n")
    return {
        "model": options.model,
        "effort": options.effort,
        "codebook_sha256": hashlib.sha256(codebook.encode()).hexdigest(),
        "prices_read": PRICES_READ,
        "max_dollars": options.max_dollars,
        "spent_dollars": round(spend.dollars, 4),
        "tokens": dict(spend.tokens),
        "batches": batches,
        "judged": len(verdicts),
        "failed": failed,
        "left_by_cap": [item["id"] for item in pending] if stopped_at_cap else [],
    }


def report(manifest: dict[str, Any]) -> str:
    """The usage report: tokens by kind and the cost."""
    tokens = manifest["tokens"]
    return (
        f"{manifest['model']} ({manifest['effort']}): {manifest['judged']} judged, "
        f"{len(manifest['failed'])} failed, {len(manifest['left_by_cap'])} left by the cap\n"
        f"tokens: input {tokens.get('input_tokens', 0):,}, cache write "
        f"{tokens.get('cache_creation_input_tokens', 0):,}, cache read "
        f"{tokens.get('cache_read_input_tokens', 0):,}, output {tokens.get('output_tokens', 0):,}\n"
        f"cost at batch prices: ${manifest['spent_dollars']:.2f} "
        f"of a ${manifest['max_dollars']:.2f} cap\n"
    )


def main() -> None:
    """Parse the arguments, judge, report the usage and record the run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("items", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--answers", choices=sorted(SCHEMAS), required=True)
    parser.add_argument("--model", choices=sorted(PRICES), required=True)
    parser.add_argument(
        "--effort", choices=["low", "medium", "high", "xhigh", "max"], required=True
    )
    parser.add_argument("--max-dollars", type=float, required=True)
    parser.add_argument("--chunk", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=4096)
    options = parser.parse_args()
    items = load_items(options.items)
    client = anthropic.Anthropic(api_key=ApiKeySettings().anthropic_api_key)
    verdicts = json.loads(options.out.read_text()) if options.out.exists() else {}
    with LiveProgress(Console()) as live:
        timer = StageTimer(live)
        manifest = judge(items, options, client, verdicts, progress=timer)
    manifest_path = options.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    sys.stdout.write(report(manifest))
    counts = {name: int(manifest["tokens"].get(name, 0)) for name in USAGE_FIELDS}
    counts |= {"judged": manifest["judged"], "cost_micro_usd": int(manifest["spent_dollars"] * 1e6)}
    record_run(
        read_catalog(), command=f"judge {options.model}", stages=timer.finish(), counts=counts
    )


if __name__ == "__main__":
    main()
