"""The spelling audit's calibration page: the pairs the author judges blind.

`docs/spellings.md` fixes the audit. Fills `spelling_template.html` with the items of a
calibration file (each pair with its corpus sentences, no agent verdict) and the name of the
store collection its answers go to, one per calibration. Usage:
`python scripts/review/spelling_page.py CALIBRATION.json OUT.html COLLECTION`.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TEMPLATE = Path(__file__).with_name("spelling_template.html")


def main(calibration_path: Path, out_path: Path, collection: str) -> None:
    """Write the page with the calibration items and their collection."""
    if not re.fullmatch(r"[a-z0-9_]+", collection):
        raise SystemExit(f"collection names use lowercase letters, digits and _: {collection!r}")
    items = json.loads(calibration_path.read_text(encoding="utf-8"))
    page = (
        TEMPLATE.read_text(encoding="utf-8")
        .replace("__ITEMS__", json.dumps(items, ensure_ascii=False))
        .replace("__COLLECTION__", collection)
    )
    out_path.write_text(page, encoding="utf-8")
    sys.stdout.write(f"{len(items)} pairs written to {out_path} (collection {collection})\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
