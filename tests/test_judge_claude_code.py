"""The Claude Code judge (`pipelines/judge_claude_code.py`): parts, isolation, retries, manifest."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from interpretant.pipelines import judge_claude_code as judge_cli
from interpretant.pipelines.judge_claude_code import CommandLineJudging
from interpretant.review.judging import ADJUDICATION, render_message, split_parts


def item(item_id: str) -> dict[str, Any]:
    """A usages item as the audit writes it."""
    return {
        "id": item_id,
        "kind": "kept",
        "word": "attention",
        "parts_of_speech": ["NOUN"],
        "sentence": "We pay attention.",
    }


def options(tmp_path: Path, items_per_call: int = 50) -> CommandLineJudging:
    """How a judge writing under ``tmp_path`` judges."""
    codebook = tmp_path / "prompt.md"
    codebook.write_text("Judge the usages.")
    return CommandLineJudging(
        name="judge_A",
        codebook=codebook,
        answers="usages",
        model="sonnet",
        effort="low",
        items_per_call=items_per_call,
        parallel=2,
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
    assert sorted(split_parts(folder)) == ["00", "01"]
    single = tmp_path / "adjudicate.jsonl"
    single.write_text("".join(json.dumps(item(f"i{n}")) + "\n" for n in range(120)))
    slices = split_parts(single)
    assert [len(items) for items in slices.values()] == [50, 50, 20]


def test_every_call_is_isolated_and_constrained_to_the_schema(tmp_path: Path) -> None:
    arguments = judge_cli.build_command(options(tmp_path))
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


def test_judging_writes_each_part_and_skips_those_already_judged(tmp_path: Path) -> None:
    parts = {"00": [item("a"), item("b")], "01": [item("c")]}
    calls, call = answering()
    run = judge_cli.generate_verdicts(parts, options(tmp_path), tmp_path / "verdicts", call)
    written = json.loads((tmp_path / "verdicts" / "judge_A_part_00.json").read_text())
    assert sorted(written) == ["a", "b"]
    assert written["a"] == {"verdict": "right", "reason": "none", "note": ""}
    assert run["judged"] == 3
    assert run["models"] == ["claude-sonnet-5-5"]
    again = judge_cli.generate_verdicts(parts, options(tmp_path), tmp_path / "verdicts", call)
    assert again["judged"] == 0
    assert len(calls) == 2


def test_a_part_answered_with_other_ids_is_retried_once_then_reported(tmp_path: Path) -> None:
    parts = {"00": [item("a"), item("b")]}
    calls, call = answering(wrong_first=1)
    assert (
        judge_cli.generate_verdicts(parts, options(tmp_path), tmp_path / "verdicts", call)["judged"]
        == 2
    )
    assert len(calls) == 2
    other = tmp_path / "other"
    other.mkdir()
    calls, call = answering(wrong_first=2)
    run = judge_cli.generate_verdicts(parts, options(other), other / "verdicts", call)
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
    manifest = judge_cli.build_merged_manifest(earlier, run, {"model": "sonnet"})
    assert manifest["judged"] == 60
    assert manifest["tokens"] == {"input_tokens": 105}
    assert manifest["model"] == "sonnet"


def test_a_part_judged_one_item_per_call_is_written_once_whole(tmp_path: Path) -> None:
    parts = {"00": [item("a"), item("b"), item("c")], "01": [item("d")]}
    calls, call = answering()
    run = judge_cli.generate_verdicts(parts, options(tmp_path, 1), tmp_path / "verdicts", call)
    assert len(calls) == 4
    written = json.loads((tmp_path / "verdicts" / "judge_A_part_00.json").read_text())
    assert sorted(written) == ["a", "b", "c"]
    assert run["judged"] == 4


def test_a_part_with_a_failed_call_is_not_written(tmp_path: Path) -> None:
    _, call = answering(wrong_first=2)
    run = judge_cli.generate_verdicts(
        {"00": [item("a"), item("b")]}, options(tmp_path, 1), tmp_path / "verdicts", call
    )
    assert "00" in run["failed"]
    assert not (tmp_path / "verdicts" / "judge_A_part_00.json").exists()


def test_an_adjudication_item_is_sent_with_both_answers_and_the_instruction() -> None:
    disputed = item("a") | {
        "judge_1": {"verdict": "right", "reason": "none", "note": "a use"},
        "judge_2": {"verdict": "wrong", "reason": "name", "note": "a name"},
    }
    text = render_message([disputed], "usages")
    assert text.startswith(ADJUDICATION)
    assert "Judge 1 answered:" in text
    assert "Judge 2 answered:" in text
    assert not render_message([item("b")], "usages").startswith(ADJUDICATION)


def test_the_command_line_never_sees_the_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the key, Claude Code would bill the API instead of the subscription."""
    monkeypatch.setenv(judge_cli.API_KEY_VARIABLE, "sk-test")
    monkeypatch.setattr(judge_cli, "EMPTY_FOLDER", tmp_path / "empty")
    probe = "import json, os; print(json.dumps({'key': os.environ.get('ANTHROPIC_API_KEY')}))"

    result = judge_cli.generate_with_claude([sys.executable, "-c", probe], "")

    assert result == {"key": None}
