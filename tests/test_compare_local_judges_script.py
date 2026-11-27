"""The local trial's comparison script: its setups, combinations and stage-2 choice."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from interpretant.catalog.junk_gate import Run3Verdict

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "review" / "compare_local_judges.py"


@pytest.fixture(scope="module")
def compare() -> ModuleType:
    spec = importlib.util.spec_from_file_location("compare_local_judges", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["compare_local_judges"] = module
    spec.loader.exec_module(module)
    return module


def answer(verdict: str, own_prose: float | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"verdict": verdict, "kind": "none", "sentence": "none"}
    if own_prose is not None:
        entry["probabilities"] = {"own_prose": own_prose}
    return entry


def write_setup(
    folder: Path,
    name: str,
    answers: dict[str, dict[str, Any]],
    *,
    kind: str = "chat",
    machine: str = "runner",
    seconds_per_item: float = 1.0,
) -> None:
    local = folder / "verdicts" / "local"
    local.mkdir(parents=True, exist_ok=True)
    (local / f"{name}_part_00.json").write_text(json.dumps(answers))
    run = {
        "host_label": machine,
        "items_judged": len(answers),
        "seconds_total": seconds_per_item * len(answers),
        "size": 100,
        "size_vram": 80,
    }
    (local / f"{name}.manifest.json").write_text(json.dumps({"kind": kind, "runs": [run]}))


def test_a_setup_is_scored_against_the_author_with_r018_as_junk(compare: ModuleType) -> None:
    author = [
        Run3Verdict(id="r001", work_id="W1", decade=1950, field="ai", verdict="not_abstract"),
        Run3Verdict(id="r002", work_id="W2", decade=1950, field="ai", verdict="abstract"),
        Run3Verdict(id="r003", work_id="W3", decade=1950, field="ai", verdict="unsure"),
        Run3Verdict(id="r018", work_id="W4", decade=1950, field="ai", verdict="abstract"),
    ]
    reference = compare.build_reference(author)
    assert reference == {"r001": True, "r002": False, "r018": True}
    scores = compare.score_calls({"r001": True, "r002": True, "r018": False}, reference)
    assert (scores["agreement"], scores["junk_found"], scores["abstracts_called_junk"]) == (1, 1, 1)
    assert scores["meets_bars"] is False


def test_a_panels_third_setup_settles_only_the_first_twos_differences(
    compare: ModuleType, tmp_path: Path
) -> None:
    write_setup(tmp_path, "a", {"r1": answer("junk"), "r2": answer("abstract")})
    write_setup(tmp_path, "b", {"r1": answer("junk"), "r2": answer("junk")})
    write_setup(tmp_path, "c", {"r1": answer("abstract"), "r2": answer("abstract")})
    candidates = compare.build_candidates(compare.load_setups(tmp_path), ["r1", "r2"])
    panel = next(candidate for candidate in candidates if candidate.name == "panel a + b / c")
    assert panel.calls == {"r1": True, "r2": False}
    jury = next(candidate for candidate in candidates if candidate.shape == "jury")
    assert jury.calls == {"r1": True, "r2": False}


def test_a_cascade_sends_only_its_band_to_the_chat_model(
    compare: ModuleType, tmp_path: Path
) -> None:
    write_setup(
        tmp_path,
        "decider",
        {"r1": answer("junk", 0.05), "r2": answer("abstract", 0.6), "r3": answer("abstract", 0.95)},
        kind="decision",
    )
    write_setup(tmp_path, "chat", {item: answer("junk") for item in ("r1", "r2", "r3")})
    candidates = compare.build_candidates(compare.load_setups(tmp_path), ["r1", "r2", "r3"])
    narrow = next(candidate for candidate in candidates if "[0.2, 0.8]" in candidate.name)
    assert narrow.calls == {"r1": True, "r2": True, "r3": False}


def test_setups_on_two_machines_run_together_and_on_one_in_turn(
    compare: ModuleType, tmp_path: Path
) -> None:
    items = {"r1": answer("junk")}
    write_setup(tmp_path, "a", items, machine="runner", seconds_per_item=2.0)
    write_setup(tmp_path, "b", items, machine="second", seconds_per_item=3.0)
    write_setup(tmp_path, "c", items, machine="runner", seconds_per_item=1.0)
    candidates = {
        candidate.name: candidate
        for candidate in compare.build_candidates(compare.load_setups(tmp_path), ["r1"])
    }
    per_item = compare.STAGE2_ITEMS / 3600
    assert candidates["jury a + b + c"].hours == pytest.approx(3.0 * per_item)
    assert candidates["panel a + c / b"].hours == pytest.approx(3.0 * per_item)


def test_stage2_takes_the_candidates_nothing_beats_on_both_accuracy_and_time(
    compare: ModuleType,
) -> None:
    def entry(name: str, agreement: int, found: int, hours: float) -> dict[str, Any]:
        return {
            "name": name,
            "hours_for_l_and_e": hours,
            "run3": {"agreement": agreement, "junk_found": found},
        }

    scored = [
        entry("fast", 180, 27, 1.0),
        entry("accurate", 190, 29, 5.0),
        entry("beaten", 179, 27, 2.0),
        entry("too slow", 195, 30, 13.0),
    ]
    assert compare.select_for_stage2(scored) == ["fast", "accurate"]
