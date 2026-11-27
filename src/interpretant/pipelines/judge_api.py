"""Judging review items with Claude through the Message Batches API.

Every item is its own request: the codebook is the system prompt, byte-identical and cached for an
hour; the item is the message (``review.judging.render``); the answer is constrained to the
codebook's schema (structured output). Answers are merged into the verdicts file as each chunk
ends, so a stopped run resumes where it left off.

**The cap.** Requests go out in chunks. A chunk is submitted only if the dollars spent so far plus
the chunk's worst case (every input token written to the cache, every request using its full
``max_tokens``, at batch prices) stay within ``max_dollars``; so a run never spends more than its
cap. The cap counts this run's spending only. **The manifest** holds the usage (input, cache-write,
cache-read and output tokens, and their cost at batch prices), the batch ids, the model, the
effort and the codebook's sha256. Raw responses are not kept. The API key comes from the
checkout's ``.env`` (``ANTHROPIC_API_KEY``), never from the shell: exported there, it would switch
Claude Code itself from the subscription to the API.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from interpretant.pipelines.params import ENV_FILE
from interpretant.pipelines.progress import SILENT, Progress
from interpretant.review.judging import SCHEMAS, Answers, Effort, render

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
USAGE_FIELDS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)
Thinking = Literal["adaptive", "off"]


class ApiKeySettings(BaseSettings):
    """The Anthropic API key, from the checkout's ``.env``."""

    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    anthropic_api_key: str


@dataclass(frozen=True)
class ApiJudging:
    """How a judge judges: its codebook and answers, the model and its effort, the cap."""

    codebook: Path
    answers: Answers
    model: str
    effort: Effort
    max_dollars: float
    chunk: int = 500
    max_tokens: int = 4096
    thinking: Thinking = "adaptive"


@dataclass
class Spend:
    """Tokens used and dollars spent so far."""

    tokens: Counter[str] = field(default_factory=Counter)
    dollars: float = 0.0


def compute_cost(usage: Any, model: str) -> float:  # noqa: ANN401 (an SDK usage record)
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


def compute_worst_case(text: str, system_tokens: int, max_tokens: int, model: str) -> float:
    """The most one request can cost: all input written to the cache, all output tokens used."""
    price = PRICES[model]
    input_tokens = system_tokens + len(text) / CHARACTERS_PER_TOKEN + 50
    total = input_tokens * price["input"] * CACHE_WRITE_FACTOR + max_tokens * price["output"]
    return total / 1e6 * BATCH_DISCOUNT


def build_request(
    item: dict[str, Any], judging: ApiJudging, system: list[dict[str, Any]]
) -> dict[str, Any]:
    """One batch request for an item; with thinking off, the model answers without reasoning."""
    params: dict[str, Any] = {
        "model": judging.model,
        "max_tokens": judging.max_tokens,
        "system": system,
        "output_config": {
            "effort": judging.effort,
            "format": {"type": "json_schema", "schema": SCHEMAS[judging.answers]},
        },
        "messages": [{"role": "user", "content": render(item, judging.answers)}],
    }
    if judging.thinking == "off":
        params["thinking"] = {"type": "between_tools"}
    return {"custom_id": item["id"], "params": params}


def parse_answer(result: Any, answers: Answers) -> dict[str, Any] | None:  # noqa: ANN401 (SDK)
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


def generate_verdicts(
    items: list[dict[str, Any]],
    judging: ApiJudging,
    client: Any,  # noqa: ANN401 (the SDK client, faked in tests)
    verdicts: dict[str, Any],
    out: Path,
    *,
    progress: Progress = SILENT,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Judge every item not yet in ``verdicts`` within the cap; the manifest of the run.

    The verdicts are saved to ``out`` after each chunk. Its folder is made before any batch is
    sent, so a missing folder can't fail the run after the batch has been paid for.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    codebook = judging.codebook.read_text()
    system = [
        {"type": "text", "text": codebook, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    ]
    system_tokens = client.messages.count_tokens(
        model=judging.model, system=system, messages=[{"role": "user", "content": "x"}]
    ).input_tokens
    spend, batches, failed = Spend(), [], {}
    pending = [item for item in items if item["id"] not in verdicts]
    retried: set[str] = set()
    progress.stage("judging", total=len(pending))
    is_stopped_at_cap = False
    while pending:
        chunk, pending = pending[: judging.chunk], pending[judging.chunk :]
        worst = sum(
            compute_worst_case(
                render(item, judging.answers), system_tokens, judging.max_tokens, judging.model
            )
            for item in chunk
        )
        if spend.dollars + worst > judging.max_dollars:
            is_stopped_at_cap = True
            pending = chunk + pending
            break
        batch = client.messages.batches.create(
            requests=[build_request(item, judging, system) for item in chunk]
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
            spend.dollars += compute_cost(usage, judging.model)
            spend.tokens.update({name: getattr(usage, name, 0) or 0 for name in USAGE_FIELDS})
            parsed = parse_answer(result, judging.answers)
            if parsed is None:
                failed[result.custom_id] = f"no valid answer ({result.result.message.stop_reason})"
            else:
                verdicts[result.custom_id] = parsed
            progress.advance()
        out.write_text(json.dumps(verdicts, indent=1, ensure_ascii=False) + "\n")
    return {
        "model": judging.model,
        "effort": judging.effort,
        "thinking": judging.thinking,
        "codebook_sha256": hashlib.sha256(codebook.encode()).hexdigest(),
        "prices_read": PRICES_READ,
        "max_dollars": judging.max_dollars,
        "spent_dollars": round(spend.dollars, 4),
        "tokens": dict(spend.tokens),
        "batches": batches,
        "judged": len(verdicts),
        "failed": failed,
        "left_by_cap": [item["id"] for item in pending] if is_stopped_at_cap else [],
    }


def render_report(manifest: dict[str, Any]) -> str:
    """The usage report: tokens by kind and the cost."""
    tokens = manifest["tokens"]
    return (
        f"{manifest['model']} ({manifest['effort']}, thinking {manifest['thinking']}): "
        f"{manifest['judged']} judged, "
        f"{len(manifest['failed'])} failed, {len(manifest['left_by_cap'])} left by the cap\n"
        f"tokens: input {tokens.get('input_tokens', 0):,}, cache write "
        f"{tokens.get('cache_creation_input_tokens', 0):,}, cache read "
        f"{tokens.get('cache_read_input_tokens', 0):,}, output {tokens.get('output_tokens', 0):,}\n"
        f"cost at batch prices: ${manifest['spent_dollars']:.2f} "
        f"of a ${manifest['max_dollars']:.2f} cap"
    )
