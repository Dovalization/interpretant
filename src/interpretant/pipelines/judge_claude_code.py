"""Judging review items with Claude Code's command line, on the subscription.

The setup is the usages audit's (``docs/usages_audit.md``, the judging setup).

The items come in parts (``review.judging.split_parts``). Each part's items are judged
``items_per_call`` per call of ``claude -p`` (one, by default: an item judged on its own; calls
holding many items gave verdicts their own notes contradicted), and the part's answers go to
``<verdicts>/<name>_part_<NN>.json``, item id to answer, the layout the scorers read, once all its
calls have answered; a part whose file exists is skipped, so a stopped run resumes.

Each call sees only the codebook, as the whole system prompt, and its items: no tools, no
settings, MCP servers or CLAUDE.md (it runs in one empty folder), no saved session; the answer is
constrained to the codebook's schema. A call that fails, or answers another set of ids, is tried
once more, then reported. The judging uses the subscription: the command line never sees an API
key, and its dollar figure is what the same tokens would cost on the API, kept for comparison.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from interpretant.pipelines.progress import SILENT, Progress
from interpretant.review.judging import Answers, Effort, build_batch_schema, render_message

ATTEMPTS = 2
TIMEOUT_SECONDS = 1200
EMPTY_FOLDER = Path(tempfile.gettempdir()) / "interpretant-judge-cli"
# The key that would move the command line from the subscription to the API
API_KEY_VARIABLE = "ANTHROPIC_API_KEY"
USAGE_FIELDS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)
# Runs the command line with a message on its standard input; returns its JSON result.
Call = Callable[[list[str], str], dict[str, Any]]


@dataclass(frozen=True)
class CommandLineJudging:
    """How a judge judges: its name, codebook and answers, the model and its effort, its calls."""

    name: str
    codebook: Path
    answers: Answers
    model: str
    effort: Effort
    items_per_call: int = 1
    parallel: int = 4
    claude: str = "claude"


def build_command(judging: CommandLineJudging) -> list[str]:
    """The isolated print-mode call: the codebook as the whole system prompt, nothing else."""
    return [
        judging.claude,
        "-p",
        "--model",
        judging.model,
        "--effort",
        judging.effort,
        "--system-prompt-file",
        str(judging.codebook.resolve()),
        "--json-schema",
        json.dumps(build_batch_schema(judging.answers)),
        "--tools",
        "",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--output-format",
        "json",
    ]


def generate_with_claude(arguments: list[str], text: str) -> dict[str, Any]:
    """Run the command line from the empty folder, so no project instructions reach it.

    Every call shares the one folder, as Claude Code scopes its prompt cache to a machine and a
    folder. The API key is left out of its environment, whatever this process was given.

    Raises:
        RuntimeError: if the folder is not empty.
    """
    EMPTY_FOLDER.mkdir(exist_ok=True)
    if any(EMPTY_FOLDER.iterdir()):
        raise RuntimeError(f"{EMPTY_FOLDER} must stay empty: files there could reach the judge")
    environment = {name: value for name, value in os.environ.items() if name != API_KEY_VARIABLE}
    done = subprocess.run(
        arguments,
        input=text,
        capture_output=True,
        text=True,
        cwd=EMPTY_FOLDER,
        env=environment,
        timeout=TIMEOUT_SECONDS,
        check=False,
    )
    try:
        result: dict[str, Any] = json.loads(done.stdout)
    except json.JSONDecodeError:
        return {"is_error": True, "result": f"exit {done.returncode}: {done.stderr[-300:]}"}
    return result


def generate_call_answers(
    items: list[dict[str, Any]], arguments: list[str], text: str, call: Call
) -> tuple[dict[str, Any] | None, Counter[str], set[str], str]:
    """One call's answers by item id (None if every attempt failed), its usage, models, problem."""
    expected = [item["id"] for item in items]
    usage: Counter[str] = Counter()
    models: set[str] = set()
    problem = ""
    for _ in range(ATTEMPTS):
        result = call(arguments, text)
        usage.update({name: int(result.get("usage", {}).get(name, 0)) for name in USAGE_FIELDS})
        usage["notional_micro_usd"] += int(float(result.get("total_cost_usd") or 0) * 1e6)
        models |= set(result.get("modelUsage") or {})
        if result.get("is_error"):
            problem = str(result.get("result"))[:200]
            continue
        answers = (result.get("structured_output") or {}).get("answers", [])
        by_id = {answer["id"]: {k: v for k, v in answer.items() if k != "id"} for answer in answers}
        if sorted(by_id) == sorted(expected) and len(answers) == len(expected):
            return by_id, usage, models, ""
        problem = f"answered {len(by_id)} distinct ids for {len(expected)} items"
    return None, usage, models, problem


def generate_verdicts(
    parts: dict[str, list[dict[str, Any]]],
    judging: CommandLineJudging,
    verdicts: Path,
    call: Call,
    *,
    progress: Progress = SILENT,
) -> dict[str, Any]:
    """Judge every part without a verdict file under ``verdicts``, in parallel calls.

    A part's file is written once all of its calls have answered; a part with a failed call is
    reported and left for the next run. Returns this run's usage and failures.
    """
    verdicts.mkdir(parents=True, exist_ok=True)
    pending = {
        part: items
        for part, items in parts.items()
        if not (verdicts / f"{judging.name}_part_{part}.json").exists()
    }
    progress.stage("judging", total=sum(len(items) for items in pending.values()))
    arguments = build_command(judging)
    usage: Counter[str] = Counter()
    models: set[str] = set()
    failed: dict[str, str] = {}
    answered: dict[str, dict[str, Any]] = {part: {} for part in pending}
    calls_left = dict.fromkeys(pending, 0)
    judged = 0
    with ThreadPoolExecutor(max_workers=judging.parallel) as pool:
        futures = {}
        for part, items in pending.items():
            for start in range(0, len(items), judging.items_per_call):
                chunk = items[start : start + judging.items_per_call]
                text = render_message(chunk, judging.answers)
                futures[pool.submit(generate_call_answers, chunk, arguments, text, call)] = part
                calls_left[part] += 1
        for future in as_completed(futures):
            part = futures[future]
            answers, call_usage, call_models, problem = future.result()
            usage.update(call_usage)
            models |= call_models
            calls_left[part] -= 1
            if answers is None:
                failed.setdefault(part, problem)
            else:
                answered[part] |= answers
                progress.advance(len(answers))
            if calls_left[part] == 0 and part not in failed:
                path = verdicts / f"{judging.name}_part_{part}.json"
                path.write_text(json.dumps(answered[part], indent=1, ensure_ascii=False) + "\n")
                judged += len(answered[part])
    return {"judged": judged, "tokens": dict(usage), "models": sorted(models), "failed": failed}


def build_merged_manifest(
    earlier: dict[str, Any], run: dict[str, Any], facts: dict[str, str]
) -> dict[str, Any]:
    """The manifest after this run: totals over every run, this run's failures, the setup."""
    tokens = Counter(earlier.get("tokens", {}))
    tokens.update(run["tokens"])
    return facts | {
        "judged": int(earlier.get("judged", 0)) + run["judged"],
        "tokens": dict(tokens),
        "models": sorted(set(earlier.get("models", [])) | set(run["models"])),
        "failed": run["failed"],
    }
