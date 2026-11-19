"""The command-line judge (`scripts/review/judge_cli.py`): parts, isolation, retries, resuming."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]


def load_script() -> ModuleType:
    """The command-line judge as a module."""
    path = REPOSITORY / "scripts" / "review" / "judge_cli.py"
    spec = importlib.util.spec_from_file_location("judge_cli_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["judge_cli_script"] = module
    spec.loader.exec_module(module)
    return module


judge_cli = load_script()


def item(item_id: str) -> dict[str, Any]:
    """A usages item as the audit writes it."""
    return {
        "id": item_id,
        "kind": "kept",
        "word": "attention",
        "parts_of_speech": ["NOUN"],
        "sentence": "We pay attention.",
    }


def options(tmp_path: Path) -> argparse.Namespace:
    """The script's options for a judge writing under ``tmp_path``."""
    codebook = tmp_path / "prompt.md"
    codebook.write_text("Judge the usages.")
    return argparse.Namespace(
        verdicts=tmp_path / "verdicts",
        name="judge_A",
        codebook=codebook,
        answers="usages",
        model="sonnet",
        effort="low",
        parallel=2,
        slice=50,
        claude="claude",
    )


def answering(*, wrong_first: int = 0) -> tuple[list[list[str]], Any]:
    """A stand-in command line that judges every item right, after ``wrong_first`` bad answers."""
    calls: list[list[str]] = []

    def call(arguments: list[str], text: str) -> dict[str, Any]:
        calls.append(arguments)
        ids = [
            line.removeprefix("Item id: ")
            for line in text.splitlines()
            if line.startswith("Item id: ")
        ]
        if len(calls) <= wrong_first:
            ids = ids[:-1]
        answers = [
            {"id": item_id, "verdict": "right", "reason": "none", "note": ""} for item_id in ids
        ]
        return {
            "is_error": False,
            "structured_output": {"answers": answers},
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "modelUsage": {"claude-sonnet-5-5": {}},
            "total_cost_usd": 0.001,
        }

    return calls, call


def test_parts_are_batch_files_or_slices_of_one_file(tmp_path: Path) -> None:
    folder = tmp_path / "audit"
    folder.mkdir()
    for number in (0, 1):
        (folder / f"batch_0{number}.jsonl").write_text(json.dumps(item(f"i{number}")) + "\n")
    assert sorted(judge_cli.parts_of(folder)) == ["00", "01"]
    single = tmp_path / "adjudicate.jsonl"
    single.write_text("".join(json.dumps(item(f"i{n}")) + "\n" for n in range(120)))
    slices = judge_cli.parts_of(single)
    assert [len(items) for items in slices.values()] == [50, 50, 20]


def test_every_call_is_isolated_and_constrained_to_the_schema(tmp_path: Path) -> None:
    arguments = judge_cli.command(options(tmp_path))
    flags = dict(itertools.pairwise(arguments))
    assert flags["--tools"] == ""
    assert flags["--setting-sources"] == ""
    assert flags["--system-prompt-file"] == str((tmp_path / "prompt.md").resolve())
    assert {"--strict-mcp-config", "--no-session-persistence", "-p"} <= set(arguments)
    schema = json.loads(flags["--json-schema"])
    assert schema["properties"]["answers"]["items"]["required"] == [
        "id",
        "verdict",
        "reason",
        "note",
    ]


def test_judge_all_writes_each_part_and_skips_those_already_judged(tmp_path: Path) -> None:
    parts = {"00": [item("a"), item("b")], "01": [item("c")]}
    calls, call = answering()
    run = judge_cli.judge_all(parts, options(tmp_path), call)
    written = json.loads((tmp_path / "verdicts" / "judge_A_part_00.json").read_text())
    assert sorted(written) == ["a", "b"]
    assert written["a"] == {"verdict": "right", "reason": "none", "note": ""}
    assert run["judged"] == 3
    assert run["models"] == ["claude-sonnet-5-5"]
    again = judge_cli.judge_all(parts, options(tmp_path), call)
    assert again["judged"] == 0
    assert len(calls) == 2


def test_a_part_answered_with_other_ids_is_retried_once_then_reported(tmp_path: Path) -> None:
    parts = {"00": [item("a"), item("b")]}
    calls, call = answering(wrong_first=1)
    assert judge_cli.judge_all(parts, options(tmp_path), call)["judged"] == 2
    assert len(calls) == 2
    other = tmp_path / "other"
    other.mkdir()
    calls, call = answering(wrong_first=2)
    run = judge_cli.judge_all(parts, options(other), call)
    assert run["failed"] == {"00": "answered 1 distinct ids for 2 items"}
    assert not (other / "verdicts" / "judge_A_part_00.json").exists()
    assert run["tokens"]["input_tokens"] == 20


def test_the_manifest_adds_up_every_run() -> None:
    earlier = {"judged": 50, "tokens": {"input_tokens": 100}, "models": ["claude-sonnet-5-5"]}
    run = {
        "judged": 10,
        "tokens": {"input_tokens": 5},
        "models": ["claude-sonnet-5-5"],
        "failed": {},
    }
    manifest = judge_cli.merged_manifest(earlier, run, {"model": "sonnet"})
    assert manifest["judged"] == 60
    assert manifest["tokens"] == {"input_tokens": 105}
    assert manifest["model"] == "sonnet"


def test_a_part_judged_one_item_per_call_is_written_once_whole(tmp_path: Path) -> None:
    parts = {"00": [item("a"), item("b"), item("c")], "01": [item("d")]}
    one_each = options(tmp_path)
    one_each.slice = 1
    calls, call = answering()
    run = judge_cli.judge_all(parts, one_each, call)
    assert len(calls) == 4
    written = json.loads((tmp_path / "verdicts" / "judge_A_part_00.json").read_text())
    assert sorted(written) == ["a", "b", "c"]
    assert run["judged"] == 4


def test_a_part_with_a_failed_call_is_not_written(tmp_path: Path) -> None:
    one_each = options(tmp_path)
    one_each.slice = 1
    _, call = answering(wrong_first=2)
    run = judge_cli.judge_all({"00": [item("a"), item("b")]}, one_each, call)
    assert "00" in run["failed"]
    assert not (tmp_path / "verdicts" / "judge_A_part_00.json").exists()
