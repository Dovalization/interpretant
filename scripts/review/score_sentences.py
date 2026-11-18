"""Score the fresh sentence view audit by `docs/sentence_view.md`'s rules.

Reads a review folder's key, the judges' verdicts in its `verdicts/` and, to score, the author's
recalibration answers (the page's store: wrong cuts, missed breaks, not sure). A whole abstract's
marks are its wrong cuts and the sentences holding a missed break; a join or a target sentence is
right or wrong. Where the two judges differ, the Opus adjudicator's verdict stands
(`verdicts/opus_adjudication.json`). With `pending`, the script writes the items the adjudicator
must see (`adjudicate.jsonl`, out of git) and stops. Otherwise it scores:

- **recalibration:** the agents' marks equal the author's on at least 27 of the 30;
- **cut precision:** wrong cuts are at most 2% of the sample's cuts;
- **joined cuts:** at most 2 of the 80 joins are wrong;
- **split cuts** (round 3): at most 2 of the 80 splits are wrong;
- **boundaries:** at most 5 of the 150 target sentences are wrong.

The first round's version is `854691ea`. Usage:
`python scripts/review/score_sentences.py REVIEW_DIR pending`, `... REVIEW_DIR AUTHOR.json`, or
`... REVIEW_DIR carried` for a round whose recalibration carries over.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

RECALIBRATION_AGREEMENT = 27
PRECISION_MIN = 0.98
JOINS_MAX_WRONG = 2
SPLITS_MAX_WRONG = 2
BOUNDARIES_MAX_WRONG = 5
KINDS = {"B": "abstracts", "J": "joins", "S": "splits", "T": "sentences"}
Marks = tuple[frozenset[int], frozenset[int]]


def marks(verdict: dict[str, Any]) -> Marks:
    """A whole abstract's wrong cuts and the sentences holding a missed break."""
    wrong = frozenset(int(cut) for cut in verdict.get("wrong_cuts", []))
    missed = frozenset(int(miss["sentence"]) for miss in verdict.get("missed_breaks", []))
    return wrong, missed


def judged(item_id: str, verdict: dict[str, Any]) -> Marks | str:
    """What is compared for an item: an abstract's marks, else the verdict."""
    return marks(verdict) if item_id.startswith("B") else verdict["verdict"]


def load_verdicts(folder: Path, judge: str) -> dict[str, Any]:
    """One judge's verdicts on every kind of item, merged."""
    found: dict[str, Any] = {}
    for kind in KINDS.values():
        path = folder / f"judge_{judge}_{kind}.json"
        if path.exists():
            found |= json.loads(path.read_text())
    return found


def write_pending(
    review: Path, pending: list[str], first: dict[str, Any], second: dict[str, Any]
) -> None:
    """The items the adjudicator must see, each with both judges' verdicts."""
    items = {}
    for kind in KINDS.values():
        path = review / f"{kind}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            items[item["id"]] = item
    with (review / "adjudicate.jsonl").open("w", encoding="utf-8") as handle:
        for item_id in pending:
            row = items[item_id] | {"judge_1": first[item_id], "judge_2": second[item_id]}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    sys.stdout.write(f"{len(pending)} items to adjudicate in {review / 'adjudicate.jsonl'}\n")


def main() -> None:
    """Apply the rules and report, or write the items to adjudicate."""
    review, mode = Path(sys.argv[1]), sys.argv[2]
    key = json.loads((review / "key.json").read_text())
    folder = review / "verdicts"
    first, second = load_verdicts(folder, "A"), load_verdicts(folder, "B")
    settled_path = folder / "opus_adjudication.json"
    settled = json.loads(settled_path.read_text()) if settled_path.exists() else {}
    final: dict[str, Any] = {}
    pending: list[str] = []
    agreed: Counter[str] = Counter()
    for item_id in key:
        one, two = judged(item_id, first[item_id]), judged(item_id, second[item_id])
        if one == two:
            final[item_id] = one
            agreed[KINDS[item_id[0]]] += 1
        elif item_id in settled:
            final[item_id] = judged(item_id, settled[item_id])
        else:
            pending.append(item_id)
    write = sys.stdout.write
    totals = Counter(KINDS[item_id[0]] for item_id in key)
    agreement = ", ".join(f"{agreed[kind]} of {totals[kind]} {kind}" for kind in totals)
    write(f"judges agree on {agreement}\n")
    if mode == "pending":
        write_pending(review, pending, first, second)
        return
    if pending:
        raise SystemExit("items still need the adjudicator: run with pending first")
    recalibration = sorted(i for i, entry in key.items() if entry["set"] == "recalibration")
    if mode == "carried":
        write("recalibration: carried over from the fresh audit (round 2)\n")
        score_samples(key, final)
        return
    author = {i: marks(a) for i, a in json.loads(Path(mode).read_text()).items()}
    disagreements = [i for i in recalibration if final[i] != author.get(i)]
    agreed_with_author = len(recalibration) - len(disagreements)
    write(
        f"recalibration: the agents agree with the author on {agreed_with_author} of "
        f"{len(recalibration)} (rule: at least {RECALIBRATION_AGREEMENT})\n"
    )
    for item_id in disagreements:
        agents, mine = final[item_id], author.get(item_id, (frozenset(), frozenset()))
        write(
            f"  {item_id}: author wrong {sorted(mine[0])} missed {sorted(mine[1])}; "
            f"agents wrong {sorted(agents[0])} missed {sorted(agents[1])}\n"
        )
    if agreed_with_author < RECALIBRATION_AGREEMENT:
        raise SystemExit("the recalibration does not pass: the author judges the samples")
    score_samples(key, final)


def score_samples(key: dict[str, Any], final: dict[str, Any]) -> None:
    """Report each sample the round drew against its bar, and whether the round passes."""
    write = sys.stdout.write
    passed = True
    precision_ids = [i for i, entry in key.items() if entry["set"] == "precision"]
    if precision_ids:
        cuts = sum(key[i]["cuts"] for i in precision_ids)
        wrong_cuts = sum(len(final[i][0]) for i in precision_ids)
        missed = sum(len(final[i][1]) for i in precision_ids)
        precision = 1 - wrong_cuts / cuts
        passed &= precision >= PRECISION_MIN
        write(
            f"cut precision: {wrong_cuts} wrong of {cuts} cuts, precision {precision:.1%} "
            f"(rule: at least {PRECISION_MIN:.0%}); sentences with a missed break: {missed}\n"
        )
    for name, label, bar in (
        ("joins", "joined cuts", JOINS_MAX_WRONG),
        ("splits", "split cuts", SPLITS_MAX_WRONG),
        ("boundaries", "boundaries", BOUNDARIES_MAX_WRONG),
    ):
        ids = [i for i, entry in key.items() if entry["set"] == name]
        if not ids:
            continue
        wrong = sorted(i for i in ids if final[i] == "wrong")
        passed &= len(wrong) <= bar
        rules = dict(Counter(key[i]["rule"] for i in wrong)) if name == "joins" else ""
        write(f"{label}: {len(wrong)} of {len(ids)} wrong (rule: at most {bar}): {wrong} {rules}\n")
    write(f"the audit {'passes' if passed else 'fails'}\n")


if __name__ == "__main__":
    main()
