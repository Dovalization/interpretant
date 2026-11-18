"""The sentence view audit's page for the author (`docs/sentence_view.md`).

Fills `sentence_template.html` with the named items of a review folder's `abstracts.jsonl` and
`sentences.jsonl` (whole abstracts first, then target sentences; no agent verdict) and the name of
the store collection its answers go to. Usage:
`python scripts/review/sentence_page.py REVIEW_DIR OUT.html COLLECTION ID [ID ...]`.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TEMPLATE = Path(__file__).with_name("sentence_template.html")


def main(review: Path, out_path: Path, collection: str, ids: list[str]) -> None:
    """Write the page with the named items and their collection."""
    if not re.fullmatch(r"[a-z0-9_]+", collection):
        raise SystemExit(f"collection names use lowercase letters, digits and _: {collection!r}")
    found = {}
    for name, kind in (("abstracts.jsonl", "abstract"), ("sentences.jsonl", "sentence")):
        for line in (review / name).read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            found[item["id"]] = item | {"kind": kind}
    missing = [item_id for item_id in ids if item_id not in found]
    if missing:
        raise SystemExit(f"no such items: {missing}")
    items = sorted((found[item_id] for item_id in ids), key=lambda item: item["id"])
    page = (
        TEMPLATE.read_text(encoding="utf-8")
        .replace("__ITEMS__", json.dumps(items, ensure_ascii=False))
        .replace("__COLLECTION__", collection)
    )
    out_path.write_text(page, encoding="utf-8")
    sys.stdout.write(f"{len(items)} items written to {out_path} (collection {collection})\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4:])
