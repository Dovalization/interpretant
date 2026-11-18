"""Score the sentence view audit by `docs/sentence_view.md`'s rules.

Reads the key and the judges' verdicts in `data/review/sentence_view/`. A whole abstract's marks are
its wrong cuts and the sentences holding a missed break; a target sentence's verdict is right or
wrong. Where the two judges differ, the Opus adjudicator's verdict stands
(`verdicts/opus_adjudication.json`). With `pending`, the script writes the items the adjudicator
must see (`adjudicate.jsonl`, out of git) and stops; otherwise it scores:

- **calibration:** the agents' marks equal the author's from run 3's review on at least 45 of 50;
- **cut precision:** wrong cuts are at most 2% of the sample's cuts;
- **boundaries:** at most 5 of the 150 target sentences are wrong.

Usage: `python scripts/review/score_sentences.py [pending]`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REVIEW = Path("data/review/sentence_view")
VERDICTS = REVIEW / "verdicts"
RUN3 = Path("data/review/run3/verdicts.jsonl")
CALIBRATION_AGREEMENT = 45
PRECISION_MIN = 0.98
BOUNDARIES_MAX_WRONG = 5
Marks = tuple[frozenset[int], frozenset[int]]


def load(name: str) -> dict[str, Any]:
    """A verdict file, or nothing when it does not exist yet."""
    path = VERDICTS / name
    return json.loads(path.read_text()) if path.exists() else {}


def marks(verdict: dict[str, Any]) -> Marks:
    """A whole abstract's wrong cuts and the sentences holding a missed break."""
    wrong = frozenset(int(cut) for cut in verdict.get("wrong_cuts", []))
    missed = frozenset(int(miss["sentence"]) for miss in verdict.get("missed_breaks", []))
    return wrong, missed


def author_marks(review: dict[str, Any]) -> Marks:
    """The author's marks in run 3's review: wrong breaks, and missed breaks by sentence."""
    missed = frozenset(int(miss.split(":")[0]) for miss in review["missed_breaks"])
    return frozenset(review["wrong_breaks"]), missed


def final_verdicts(
    first: dict[str, Any], second: dict[str, Any], settled: dict[str, Any], *, whole: bool
) -> tuple[dict[str, Any], list[str], int]:
    """Each id's final verdict, the ids still pending, and how many the judges agree on.

    The final verdict is the judges' when they agree, otherwise the adjudicator's.
    """
    final: dict[str, Any] = {}
    pending = []
    agreed = 0
    for item_id in first:
        one, two = (
            (marks(first[item_id]), marks(second[item_id]))
            if whole
            else (first[item_id]["verdict"], second[item_id]["verdict"])
        )
        if one == two:
            final[item_id] = one
            agreed += 1
        elif item_id in settled:
            final[item_id] = marks(settled[item_id]) if whole else settled[item_id]["verdict"]
        else:
            pending.append(item_id)
    return final, pending, agreed


def write_pending(pending: list[str], first: dict[str, Any], second: dict[str, Any]) -> None:
    """The items the adjudicator must see, each with both judges' verdicts."""
    items = {}
    for name in ("abstracts.jsonl", "sentences.jsonl"):
        for line in (REVIEW / name).read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            items[item["id"]] = item
    with (REVIEW / "adjudicate.jsonl").open("w", encoding="utf-8") as handle:
        for item_id in pending:
            row = items[item_id] | {"judge_1": first[item_id], "judge_2": second[item_id]}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    sys.stdout.write(f"{len(pending)} items to adjudicate in {REVIEW / 'adjudicate.jsonl'}\n")


def main() -> None:
    """Apply the rules and report, or write the items to adjudicate."""
    key = json.loads((REVIEW / "key.json").read_text())
    settled = load("opus_adjudication.json")
    first = load("judge_A_abstracts.json") | load("judge_A_sentences.json")
    second = load("judge_B_abstracts.json") | load("judge_B_sentences.json")
    whole_ids = [item_id for item_id in key if item_id.startswith("A")]
    whole, pending_whole, whole_agreed = final_verdicts(
        {i: first[i] for i in whole_ids}, {i: second[i] for i in whole_ids}, settled, whole=True
    )
    target_ids = [item_id for item_id in key if item_id.startswith("T")]
    targets, pending_targets, targets_agreed = final_verdicts(
        {i: first[i] for i in target_ids}, {i: second[i] for i in target_ids}, settled, whole=False
    )
    write = sys.stdout.write
    write(
        f"judges agree on {whole_agreed} of {len(whole_ids)} abstracts and {targets_agreed} of "
        f"{len(target_ids)} sentences\n"
    )
    if len(sys.argv) > 1 and sys.argv[1] == "pending":
        write_pending(pending_whole + pending_targets, first, second)
        return
    if pending_whole or pending_targets:
        raise SystemExit("items still need the adjudicator: run with pending first")
    reviews = {
        row["id"]: row
        for row in map(json.loads, RUN3.read_text(encoding="utf-8").splitlines())
        if row["part"] == "sentence_breaks"
    }
    calibration = [i for i in whole_ids if key[i]["set"] == "calibration"]
    disagreements = [
        i for i in calibration if whole[i] != author_marks(reviews[key[i]["review_id"]])
    ]
    agreed = len(calibration) - len(disagreements)
    write(
        f"calibration: the agents agree with the author on {agreed} of {len(calibration)} "
        f"(rule: at least {CALIBRATION_AGREEMENT})\n"
    )
    for item_id in disagreements:
        review = reviews[key[item_id]["review_id"]]
        agent_wrong, agent_missed = whole[item_id]
        author_wrong, author_missed = author_marks(review)
        write(
            f"  {key[item_id]['review_id']}: author wrong {sorted(author_wrong)} missed "
            f"{sorted(author_missed)}; agents wrong {sorted(agent_wrong)} missed "
            f"{sorted(agent_missed)}\n"
        )
    if agreed < CALIBRATION_AGREEMENT:
        raise SystemExit("the calibration does not pass: the author judges both samples")
    precision_ids = [i for i in whole_ids if key[i]["set"] == "precision"]
    cuts = sum(key[i]["cuts"] for i in precision_ids)
    wrong_cuts = sum(len(whole[i][0]) for i in precision_ids)
    missed = sum(len(whole[i][1]) for i in precision_ids)
    precision = 1 - wrong_cuts / cuts
    write(
        f"cut precision: {wrong_cuts} wrong of {cuts} cuts, precision {precision:.1%} "
        f"(rule: at least {PRECISION_MIN:.0%}); sentences with a missed break: {missed}\n"
    )
    wrong_targets = sorted(i for i in target_ids if targets[i] == "wrong")
    write(
        f"boundaries: {len(wrong_targets)} of {len(target_ids)} target sentences wrong "
        f"(rule: at most {BOUNDARIES_MAX_WRONG}): {wrong_targets}\n"
    )
    passed = precision >= PRECISION_MIN and len(wrong_targets) <= BOUNDARIES_MAX_WRONG
    write(f"the audit {'passes' if passed else 'fails'}\n")


if __name__ == "__main__":
    main()
