"""The matching audit's scorer (`scripts/review/score_usages.py`): reading the judges' verdicts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]


def load_scorer() -> ModuleType:
    """The scoring script as a module."""
    path = REPOSITORY / "scripts" / "review" / "score_usages.py"
    spec = importlib.util.spec_from_file_location("score_usages_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["score_usages_script"] = module
    spec.loader.exec_module(module)
    return module


scorer = load_scorer()
RIGHT = {"verdict": "right", "reason": "none", "note": ""}


def test_a_judge_is_read_from_its_part_files_without_the_manifests(tmp_path: Path) -> None:
    (tmp_path / "judge_A_part_0.json").write_text(json.dumps({"item-1": RIGHT}))
    (tmp_path / "judge_A_part_1.json").write_text(json.dumps({"item-2": RIGHT}))
    manifest = {"model": "claude-sonnet-5-5", "judged": 2}
    (tmp_path / "judge_A_part_0.manifest.json").write_text(json.dumps(manifest))
    assert scorer.load_judge(tmp_path, "A") == {"item-1": RIGHT, "item-2": RIGHT}


def test_an_item_judged_twice_by_one_judge_is_refused(tmp_path: Path) -> None:
    (tmp_path / "judge_B_part_0.json").write_text(json.dumps({"item-1": RIGHT}))
    (tmp_path / "judge_B_part_1.json").write_text(json.dumps({"item-1": RIGHT}))
    with pytest.raises(SystemExit, match="twice"):
        scorer.load_judge(tmp_path, "B")


def test_the_adjudication_is_read_from_one_file_or_its_parts(tmp_path: Path) -> None:
    (tmp_path / "opus_adjudication.json").write_text(json.dumps({"item-1": RIGHT}))
    (tmp_path / "opus_adjudication_part_00.json").write_text(json.dumps({"item-2": RIGHT}))
    (tmp_path / "opus_adjudication.manifest.json").write_text(json.dumps({"judged": 1}))
    assert scorer.load_parts(tmp_path, "opus_adjudication") == {"item-1": RIGHT, "item-2": RIGHT}
