"""Run 3's records for the junk gate (`scripts/review/junk_gate_run3.py`), from a past table."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import polars as pl
import pytest

REPOSITORY = Path(__file__).resolve().parents[1]


def load_script() -> ModuleType:
    """The script as a module."""
    path = REPOSITORY / "scripts" / "review" / "junk_gate_run3.py"
    spec = importlib.util.spec_from_file_location("junk_gate_run3_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["junk_gate_run3_script"] = module
    spec.loader.exec_module(module)
    return module


script = load_script()


def write_part(folder: Path, name: str, works: dict[str, str]) -> None:
    removal = {"start": 0, "end": 3, "reason": "abstract_label"}
    pl.DataFrame(
        {
            "work_id": list(works),
            "text": list(works.values()),
            "removals": [[removal] for _ in works],
            "flags": [[] for _ in works],
        }
    ).write_parquet(folder / f"{name}.parquet")


def test_the_judged_records_are_taken_from_every_part(tmp_path: Path) -> None:
    write_part(tmp_path, "ai_1950", {"W1": "Ab. One.", "W2": "Ab. Two."})
    write_part(tmp_path, "philosophy_1960", {"W3": "Ab. Three."})

    records = script.build_run3_records(tmp_path, {"W1", "W3"})

    assert records.columns == ["work_id", "text", "removals"]
    assert records["work_id"].to_list() == ["W1", "W3"]
    assert records["removals"].to_list()[0] == [{"start": 0, "end": 3, "reason": "abstract_label"}]


def test_a_judged_record_missing_from_the_table_is_refused(tmp_path: Path) -> None:
    write_part(tmp_path, "ai_1950", {"W1": "Ab. One."})

    with pytest.raises(SystemExit, match="1 judged records are missing"):
        script.build_run3_records(tmp_path, {"W1", "W2"})
