"""Write the matching audit's calibration page: the 60 calibration items, for the author to judge.

Reads the audit folder's `calibration.json` and its batch files, and fills `usages_template.html`
with those items in the order drawn. The page quotes corpus sentences, so it is written outside the
repository and published as a private artifact. Run from the repository root:
`python scripts/review/usages_page.py AUDIT_DIR OUT.html`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TEMPLATE = Path(__file__).with_name("usages_template.html")
SHOWN = ("id", "kind", "word", "parts_of_speech", "sentence", "reason")


def main() -> None:
    """Fill the template with the calibration items and write the page."""
    audit, out = Path(sys.argv[1]), Path(sys.argv[2])
    if Path.cwd().resolve() in out.resolve().parents:
        raise SystemExit("write the page outside the repository: it quotes corpus sentences")
    ids = json.loads((audit / "calibration.json").read_text())
    items = {}
    for path in sorted(audit.glob("batch_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            items[item["id"]] = {key: item[key] for key in SHOWN if key in item}
    chosen = [items[item_id] for item_id in ids]
    page = TEMPLATE.read_text(encoding="utf-8").replace(
        "__ITEMS__", json.dumps(chosen, ensure_ascii=False)
    )
    out.write_text(page, encoding="utf-8")
    sys.stdout.write(f"{len(chosen)} items written to {out}\n")


if __name__ == "__main__":
    main()
