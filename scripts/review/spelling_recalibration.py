"""The spelling audit's second calibration set (`docs/spellings.md`, 2026-11-17).

Draws 30 sample pairs the first calibration did not use (seed 20261122) from the audit's items
and writes them as `calibration_2.json` next to them, for the author to judge blind on the
clarified page. Usage: `python scripts/review/spelling_recalibration.py REVIEW_DIR`, REVIEW_DIR
holding `key.json`, the batches and `verdicts/author_calibration.json`.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

SEED = 20261122
CALIBRATION = 30


def main(review: Path) -> None:
    """Draw the set and write it."""
    key = json.loads((review / "key.json").read_text(encoding="utf-8"))
    first = json.loads((review / "verdicts" / "author_calibration.json").read_text())
    items = {
        item["id"]: item
        for batch in sorted(review.glob("batch_*.jsonl"))
        for item in map(json.loads, batch.read_text(encoding="utf-8").splitlines())
    }
    unused = sorted(
        item_id
        for item_id, entry in key.items()
        if "sample" in entry["sets"] and item_id not in first
    )
    drawn = sorted(random.Random(SEED).sample(unused, CALIBRATION))
    (review / "calibration_2.json").write_text(
        json.dumps([items[item_id] for item_id in drawn], indent=1, ensure_ascii=False)
    )
    sys.stdout.write(
        f"{CALIBRATION} pairs drawn from the {len(unused)} sample pairs the first calibration "
        "did not use\n"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
