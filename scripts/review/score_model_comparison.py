"""Decide the model comparison (`docs/usages_audit.md`): the flips and the judged disagreements.

Reads the comparison folder's `summary.json` and `key.json` and the judges' answers in `verdicts/`:
`judge_A.json` and `judge_B.json`, each mapping an item id to `{"answer": "A" | "B" | "both" |
"neither", "note": ...}`. Where the judges differ, the Opus adjudicator's answer stands
(`verdicts/opus_adjudication.json`). With `pending`, the script writes the items the adjudicator
must see to `adjudicate.jsonl` (they quote corpus sentences and stay out of git). Otherwise it
applies the pre-registered rule: the transformer annotates the table if more than 2% of the sampled
usages flip and, among the disagreements where exactly one reading is right, the transformer's is
right on more than half. Writes `decision.json`. Run from the repository root:
`python scripts/review/score_model_comparison.py COMPARISON_DIR pending|score`.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

FLIP_BAR = 0.02


def answers(folder: Path) -> tuple[dict[str, str], int, list[str]]:
    """Each item's final answer, how many the judges agreed on, and the items still unsettled."""
    first = json.loads((folder / "verdicts" / "judge_A.json").read_text())
    second = json.loads((folder / "verdicts" / "judge_B.json").read_text())
    path = folder / "verdicts" / "opus_adjudication.json"
    settled = json.loads(path.read_text()) if path.exists() else {}
    key = json.loads((folder / "key.json").read_text())
    final, agreed, unsettled = {}, 0, []
    for item_id in key:
        one, two = first[item_id]["answer"], second[item_id]["answer"]
        if one == two:
            final[item_id], agreed = one, agreed + 1
        elif item_id in settled:
            final[item_id] = settled[item_id]["answer"]
        else:
            unsettled.append(item_id)
    return final, agreed, unsettled


def decide(folder: Path) -> dict[str, Any]:
    """The rule's two conditions and the decision.

    Raises:
        SystemExit: if an item still needs the adjudicator.
    """
    final, agreed, unsettled = answers(folder)
    if unsettled:
        raise SystemExit(f"{len(unsettled)} items still need the adjudicator: run pending first")
    key = json.loads((folder / "key.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    tally: Counter[str] = Counter()
    for item_id, answer in final.items():
        if answer in ("A", "B"):
            tally[key[item_id]["A"] if answer == "A" else _other(key[item_id]["A"])] += 1
        else:
            tally[answer] += 1
    decided = tally["transformer"] + tally["small"]
    transformer_share = tally["transformer"] / decided if decided else 0.0
    flips = summary["flip_share"] > FLIP_BAR
    better = transformer_share > 0.5
    return {
        "flip_share": summary["flip_share"],
        "flips_over_bar": flips,
        "judged": len(final),
        "judges_agreed": agreed,
        "right": dict(tally),
        "transformer_share_where_one_is_right": round(transformer_share, 3),
        "transformer_right_more_often": better,
        "model": "en_core_web_trf" if flips and better else "en_core_web_sm",
    }


def _other(model: str) -> str:
    return "small" if model == "transformer" else "transformer"


def main() -> None:
    """Write the adjudicator's items, or apply the rule and write the decision."""
    folder, mode = Path(sys.argv[1]), sys.argv[2]
    if mode == "pending":
        _, agreed, unsettled = answers(folder)
        first = json.loads((folder / "verdicts" / "judge_A.json").read_text())
        second = json.loads((folder / "verdicts" / "judge_B.json").read_text())
        items = {}
        for line in (folder / "items.jsonl").read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            items[item["id"]] = item
        with (folder / "adjudicate.jsonl").open("w", encoding="utf-8") as handle:
            for item_id in unsettled:
                row = items[item_id] | {"judge_1": first[item_id], "judge_2": second[item_id]}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        sys.stdout.write(f"judges agree on {agreed}; {len(unsettled)} items for the adjudicator\n")
        return
    decision = decide(folder)
    (folder / "decision.json").write_text(json.dumps(decision, indent=1) + "\n")
    sys.stdout.write(json.dumps(decision) + "\n")


if __name__ == "__main__":
    main()
