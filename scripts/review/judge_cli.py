"""Judge review items with Claude Code's command line, on the subscription (`docs/usages_audit.md`).

`python scripts/review/judge_cli.py ITEMS VERDICTS_DIR --name NAME --codebook PROMPT.md
--answers usages|comparison --model MODEL --effort LEVEL [--slice N] [--parallel N] [--claude PATH]`

ITEMS is a folder of `batch_*.jsonl` files, or one `.jsonl` file (an adjudication list) cut into
parts of 50. Each part's items are judged `--slice` per call of `claude -p` (one, by default: an
item judged on its own; calls holding many items gave verdicts their own notes contradicted), and
the part's answers go to `VERDICTS_DIR/NAME_part_NN.json`, item id to answer, the layout the
scorers read, once all its calls have answered; a part whose file exists is skipped, so a stopped
run resumes. Each call sees only the codebook, as the whole
system prompt, and its items: no tools, no settings, MCP servers or CLAUDE.md (it runs in one
empty folder), no saved session; the answer is constrained to the codebook's schema. A call
that fails, or answers another set of ids, is tried once more, then reported. The tokens of every
call, the models the command line reports, its version and the codebook's sha256 accumulate in
`VERDICTS_DIR/NAME.manifest.json`, and each run adds its usage to the run record. The judging
uses the subscription: the command line's dollar figure is what the same tokens would cost on the
API, kept for comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent))
from judge import SCHEMAS, render

from interpretant.catalog.config import read_catalog
from interpretant.cli.progress import LiveProgress
from interpretant.pipelines.progress import SILENT, Progress, StageTimer
from interpretant.pipelines.runs import record_run

PART_SIZE = 50
ATTEMPTS = 2
TIMEOUT_SECONDS = 1200
EMPTY_FOLDER = Path(tempfile.gettempdir()) / "interpretant-judge-cli"
USAGE_FIELDS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)
# Runs the command line with a message on its standard input; returns its JSON result.
Call = Callable[[list[str], str], dict[str, Any]]


def load_lines(path: Path) -> list[dict[str, Any]]:
    """The items of one JSON-lines file."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def parts_of(source: Path) -> dict[str, list[dict[str, Any]]]:
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


def batch_schema(answers: str) -> dict[str, Any]:
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


ADJUDICATION = (
    "Two independent judges answered each item below differently; their answers follow the item, "
    "as judge 1 and judge 2. Settle each item yourself by the codebook: you may agree with either "
    "judge, and you owe neither of them anything. "
)


def message(items: list[dict[str, Any]], answers: str) -> str:
    """The items as one message, each rendered as the API judge renders it.

    Items that carry two judges' answers (an adjudication list) get the adjudicator's instruction.
    """
    rendered = "\n\n---\n\n".join(render(item, answers) for item in items)
    settling = ADJUDICATION if any("judge_1" in item for item in items) else ""
    return (
        f"{settling}Judge each of the {len(items)} items below by the codebook, each on its own. "
        f"Give one answer per item, with its item id.\n\n{rendered}"
    )


def command(options: argparse.Namespace) -> list[str]:
    """The isolated print-mode call: the codebook as the whole system prompt, nothing else."""
    return [
        options.claude,
        "-p",
        "--model",
        options.model,
        "--effort",
        options.effort,
        "--system-prompt-file",
        str(options.codebook.resolve()),
        "--json-schema",
        json.dumps(batch_schema(options.answers)),
        "--tools",
        "",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--output-format",
        "json",
    ]


def run_claude(arguments: list[str], text: str) -> dict[str, Any]:
    """Run the command line from the empty folder, so no project instructions reach it.

    Every call shares the one folder, as Claude Code scopes its prompt cache to a machine and a
    folder.

    Raises:
        RuntimeError: if the folder is not empty.
    """
    EMPTY_FOLDER.mkdir(exist_ok=True)
    if any(EMPTY_FOLDER.iterdir()):
        raise RuntimeError(f"{EMPTY_FOLDER} must stay empty: files there could reach the judge")
    done = subprocess.run(
        arguments,
        input=text,
        capture_output=True,
        text=True,
        cwd=EMPTY_FOLDER,
        timeout=TIMEOUT_SECONDS,
        check=False,
    )
    try:
        result: dict[str, Any] = json.loads(done.stdout)
    except json.JSONDecodeError:
        return {"is_error": True, "result": f"exit {done.returncode}: {done.stderr[-300:]}"}
    return result


def judge_call(
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


def judge_all(
    parts: dict[str, list[dict[str, Any]]],
    options: argparse.Namespace,
    call: Call,
    *,
    progress: Progress = SILENT,
) -> dict[str, Any]:
    """Judge every part without a verdict file, ``options.slice`` items per call, in parallel.

    A part's file is written once all of its calls have answered; a part with a failed call is
    reported and left for the next run. Returns this run's usage and failures.
    """
    out: Path = options.verdicts
    out.mkdir(parents=True, exist_ok=True)
    pending = {
        part: items
        for part, items in parts.items()
        if not (out / f"{options.name}_part_{part}.json").exists()
    }
    progress.stage("judging", total=sum(len(items) for items in pending.values()))
    arguments = command(options)
    usage: Counter[str] = Counter()
    models: set[str] = set()
    failed: dict[str, str] = {}
    answered: dict[str, dict[str, Any]] = {part: {} for part in pending}
    calls_left = dict.fromkeys(pending, 0)
    judged = 0
    with ThreadPoolExecutor(max_workers=options.parallel) as pool:
        futures = {}
        for part, items in pending.items():
            for start in range(0, len(items), options.slice):
                chunk = items[start : start + options.slice]
                text = message(chunk, options.answers)
                futures[pool.submit(judge_call, chunk, arguments, text, call)] = part
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
                path = out / f"{options.name}_part_{part}.json"
                path.write_text(json.dumps(answered[part], indent=1, ensure_ascii=False) + "\n")
                judged += len(answered[part])
    return {"judged": judged, "tokens": dict(usage), "models": sorted(models), "failed": failed}


def merged_manifest(
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


def main() -> None:
    """Parse the arguments, judge, and write the manifest and the run record."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("items", type=Path)
    parser.add_argument("verdicts", type=Path)
    parser.add_argument("--name", required=True, help="the verdict files' prefix, e.g. judge_A")
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--answers", choices=sorted(SCHEMAS), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--effort", choices=["low", "medium", "high", "xhigh", "max"], required=True
    )
    parser.add_argument("--slice", type=int, default=1, help="items in one call")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--claude", default="claude", help="the command line's executable")
    options = parser.parse_args()
    version = subprocess.run(
        [options.claude, "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    with LiveProgress(Console()) as live:
        timer = StageTimer(live)
        run = judge_all(parts_of(options.items), options, run_claude, progress=timer)
    manifest_path = options.verdicts / f"{options.name}.manifest.json"
    earlier = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = merged_manifest(
        earlier,
        run,
        {
            "model": options.model,
            "effort": options.effort,
            "claude_version": version,
            "codebook_sha256": hashlib.sha256(options.codebook.read_bytes()).hexdigest(),
        },
    )
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    sys.stdout.write(
        f"{run['judged']} judged this run, {len(run['failed'])} parts failed; "
        f"models {', '.join(run['models'])}; tokens {run['tokens']}\n"
    )
    counts = {name: int(run["tokens"].get(name, 0)) for name in USAGE_FIELDS}
    counts |= {"judged": run["judged"], "failed_parts": len(run["failed"])}
    counts["notional_micro_usd"] = int(run["tokens"].get("notional_micro_usd", 0))
    record_run(
        read_catalog(), command=f"judge-cli {options.model}", stages=timer.finish(), counts=counts
    )
    if run["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
