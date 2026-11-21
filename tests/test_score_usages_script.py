"""The matching audit's scorer (`scripts/review/score_usages.py`): reading the judges' verdicts."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
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


def test_a_words_loss_is_its_pool_times_the_share_judged_real() -> None:
    tallies = {"agent": Counter({"wrong": 3, "right": 7})}
    in_scope = {"agent": 9700, "user": 50}
    losses = scorer.estimated_losses(tallies, {"agent": 1000}, in_scope, "wrong")
    assert losses["agent"]["estimated"] == 300.0
    assert losses["agent"]["share"] == 0.03
    assert losses["user"] == {"pool": 0, "judged": 0, "real": 0, "estimated": 0.0, "share": 0.0}
    assert scorer.over_two_percent(losses) == ["agent"]
    assert scorer.pooled_share(losses, in_scope) == 300 / (9750 + 300)


def write_calibration(folder: Path, pairs: list[tuple[str, str]]) -> None:
    """An audit folder whose kept items have these (author, judges) verdicts, all calibrated."""
    ids = [f"U{index:04d}" for index in range(len(pairs))]
    key = {item_id: {"kind": "kept", "word": "agent"} for item_id in ids}
    judged = {
        item_id: {**RIGHT, "verdict": judges}
        for item_id, (_, judges) in zip(ids, pairs, strict=True)
    }
    (folder / "verdicts").mkdir()
    for judge in ("A", "B"):
        (folder / "verdicts" / f"judge_{judge}_part_0.json").write_text(json.dumps(judged))
    (folder / "key.json").write_text(json.dumps(key))
    (folder / "calibration.json").write_text(json.dumps(ids))
    author = {item_id: verdict for item_id, (verdict, _) in zip(ids, pairs, strict=True)}
    (folder / "author.json").write_text(json.dumps(author))


def test_the_judge_error_check_keeps_a_rate_the_author_confirms(tmp_path: Path) -> None:
    write_calibration(tmp_path, [("wrong", "wrong")] * 10 + [("right", "right")] * 30)
    [rate] = scorer.corrected_rates(tmp_path, {"kept": (250, 1000)}).values()
    assert rate["observed"] == 0.25
    assert rate["calibration"] == {"both": 10, "author_only": 0, "judges_only": 0, "neither": 30}
    low, high = rate["interval"]
    assert low < 0.25 < high


def test_the_judge_error_check_lowers_a_rate_the_judges_overstate(tmp_path: Path) -> None:
    pairs = [("wrong", "wrong")] * 10 + [("right", "wrong")] * 10 + [("right", "right")] * 20
    write_calibration(tmp_path, pairs)
    [rate] = scorer.corrected_rates(tmp_path, {"kept": (500, 1000)}).values()
    assert rate["corrected"] < 0.4
