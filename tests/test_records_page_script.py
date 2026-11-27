"""The records page (`scripts/review/records_page.py`): the chosen items, no verdict, not in git."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]


def load_script() -> ModuleType:
    """The page script as a module."""
    path = REPOSITORY / "scripts" / "review" / "records_page.py"
    spec = importlib.util.spec_from_file_location("records_page_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["records_page_script"] = module
    spec.loader.exec_module(module)
    return module


script = load_script()
TEXTS = {
    "S0001": "First record.\nIts second line, ⟦marked⟧ [… cut …] end.",
    "S0002": "Second record </script> with a closing tag.",
    "S0003": "Third record, not chosen.",
}


def write_draw(folder: Path) -> None:
    """A draw folder with three items and a judge's answers beside them."""
    (folder / "verdicts").mkdir(parents=True)
    items = [{"id": item_id, "text": text} for item_id, text in TEXTS.items()]
    (folder / "batch_00.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8"
    )
    judged = {item_id: {"verdict": "junk", "note": "JUDGE NOTE"} for item_id in TEXTS}
    (folder / "verdicts" / "judge_A_part_00.json").write_text(json.dumps(judged))


def run_page(monkeypatch: pytest.MonkeyPatch, *arguments: str) -> None:
    monkeypatch.setattr(sys, "argv", ["records_page.py", *arguments])
    script.main()


def test_the_page_holds_the_chosen_records_in_order_and_no_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_draw(tmp_path / "draw")
    ids = tmp_path / "fresh.json"
    ids.write_text(json.dumps({"seed": 1, "items": [{"id": "S0002"}, {"id": "S0001"}]}))
    page = tmp_path / "page.html"

    run_page(
        monkeypatch,
        *[str(tmp_path / "draw"), str(ids), str(page)],
        *["--title", "Fresh <30>", "--collection", "junk_gate_fresh", "--key", "items"],
    )

    html = page.read_text(encoding="utf-8")
    items = json.loads(html.split("const ITEMS = ", 1)[1].split(";\n", 1)[0].replace("<\\/", "</"))
    assert items == [
        {"id": "S0002", "text": TEXTS["S0002"]},
        {"id": "S0001", "text": TEXTS["S0001"]},
    ]
    assert "</script> with" not in html
    assert TEXTS["S0003"] not in html
    assert "JUDGE NOTE" not in html
    assert '"junk_gate_fresh"' in html
    assert "<title>Fresh &lt;30&gt;</title>" in html
    assert all(f"__{name}__" not in html for name in ("ITEMS", "TITLE", "COLLECTION"))


def test_the_page_is_never_written_inside_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_draw(tmp_path / "draw")
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps(["S0001"]))
    page = REPOSITORY / "records_page_test.html"

    with pytest.raises(SystemExit, match="outside the repository"):
        run_page(
            monkeypatch,
            *[str(tmp_path / "draw"), str(ids), str(page)],
            *["--title", "Second look", "--collection", "junk_gate_second_look"],
        )
    assert not page.exists()
