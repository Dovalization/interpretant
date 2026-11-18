"""The second recalibration's re-judging set (`docs/sentence_view.md`, 2026-11-18).

Under the settled prompt (bullets and headings between sentences start new units), the agents
re-judge round 3's 50 precision abstracts and 80 joins and round 5's 80 splits and 150 target
sentences, with the 30 fresh recalibration abstracts of round 6 shuffled in among the abstracts
(seed 20261246) so they cannot tell them apart. This script assembles that set in the round-6
folder from the earlier rounds' items and keys, renumbering the abstracts. The items quote corpus
text and stay out of git; the merged key does not. Usage:
`python scripts/review/sentence_rejudge.py REVIEW_DIR` (the folder holding round3 to round6).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

SHUFFLE_SEED = 20261246


def items(path: Path) -> dict[str, dict[str, Any]]:
    """A jsonl file's items by id."""
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return {row["id"]: row for row in rows}


def main() -> None:
    """Write the round-6 abstracts, joins, splits and sentences, and the merged key."""
    review = Path(sys.argv[1])
    round3, round5, round6 = review / "round3", review / "round5", review / "round6"
    keys = {
        name: json.loads((review / name / "key.json").read_text())
        for name in ("round3", "round5", "round6")
    }
    fresh = items(round6 / "abstracts.jsonl")
    precision = {
        i: row
        for i, row in items(round3 / "abstracts.jsonl").items()
        if keys["round3"][i]["set"] == "precision"
    }
    whole = [(row, keys["round6"][i], "round6") for i, row in fresh.items()]
    whole += [(row, keys["round3"][i], "round3") for i, row in precision.items()]
    random.Random(SHUFFLE_SEED).shuffle(whole)
    key: dict[str, Any] = {}
    with (round6 / "abstracts.jsonl").open("w", encoding="utf-8") as handle:
        for number, (row, entry, source) in enumerate(whole):
            item_id = f"B{number:03d}"
            key[item_id] = entry | {"from": f"{source}:{row['id']}"}
            handle.write(
                json.dumps({"id": item_id, "sentences": row["sentences"]}, ensure_ascii=False)
                + "\n"
            )
    for name, folder, source in (
        ("joins", round3, "round3"),
        ("splits", round5, "round5"),
        ("sentences", round5, "round5"),
    ):
        kept = items(folder / f"{name}.jsonl")
        with (round6 / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for item_id, row in kept.items():
                key[item_id] = keys[source][item_id] | {"from": f"{source}:{item_id}"}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (round6 / "key.json").write_text(json.dumps(key, indent=1) + "\n")
    counts: dict[str, int] = {}
    for entry in key.values():
        counts[entry["set"]] = counts.get(entry["set"], 0) + 1
    sys.stdout.write(f"round 6 assembled: {counts}\n")


if __name__ == "__main__":
    main()
