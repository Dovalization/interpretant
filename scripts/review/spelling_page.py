"""The spelling audit's calibration page: the 30 pairs the author judges blind.

`docs/spellings.md` fixes the audit. Fills `spelling_template.html` with the items of
`calibration.json`, as `spelling_audit.py` wrote them (each pair with its corpus sentences, no
agent verdict). Usage:
`python scripts/review/spelling_page.py CALIBRATION.json OUT.html`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TEMPLATE = Path(__file__).with_name("spelling_template.html")


def main(calibration_path: Path, out_path: Path) -> None:
    """Write the page with the calibration items."""
    items = json.loads(calibration_path.read_text(encoding="utf-8"))
    page = TEMPLATE.read_text(encoding="utf-8").replace(
        "__ITEMS__", json.dumps(items, ensure_ascii=False)
    )
    out_path.write_text(page, encoding="utf-8")
    sys.stdout.write(f"{len(items)} pairs written to {out_path}\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
