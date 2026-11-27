"""Write a page for the author to judge records one at a time: the second look or the fresh 30.

Reads the chosen item ids (a JSON list of ids, or of objects with an `id`; `--key` names the list
inside a JSON object, such as `disputed` in `calibration.json` or `items` in `fresh.json`) and the
items of a draw folder's batch files, and fills `records_template.html` with those items' texts,
in the order given, the page's title and the store collection its answers go to. No judge's answer
is shown. The page quotes the corpus, so it is written outside the repository and published as a
private artifact. Run from the repository root:
`python scripts/review/records_page.py ITEMS_DIR IDS.json OUT.html --title TITLE --collection NAME`.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).with_name("records_template.html")


def load_ids(path: Path, key: str | None) -> list[str]:
    """The chosen item ids, in their order."""
    chosen: Any = json.loads(path.read_text(encoding="utf-8"))
    if key is not None:
        chosen = chosen[key]
    return [entry["id"] if isinstance(entry, dict) else entry for entry in chosen]


def load_texts(items: Path) -> dict[str, str]:
    """Every item's text by its id, from the folder's batch files."""
    texts = {}
    for path in sorted(items.glob("batch_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            texts[item["id"]] = item["text"]
    return texts


def render_page(chosen: list[dict[str, str]], title: str, collection: str) -> str:
    """The template filled with the chosen items, the title and the collection.

    The items' JSON never closes the page's script: a slash after ``<`` is escaped.
    """
    items = json.dumps(chosen, ensure_ascii=False).replace("</", "<\\/")
    return (
        TEMPLATE.read_text(encoding="utf-8")
        .replace("__TITLE__", html.escape(title))
        .replace("__COLLECTION__", collection)
        .replace("__ITEMS__", items)
    )


def main() -> None:
    """Fill the template with the chosen items and write the page outside the repository."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("items", type=Path, help="the draw folder holding the batch files")
    parser.add_argument("ids", type=Path, help="the JSON file of the chosen item ids")
    parser.add_argument("out", type=Path, help="the page, written outside the repository")
    parser.add_argument("--title", required=True, help="the page's title")
    parser.add_argument("--collection", required=True, help="the store collection of the answers")
    parser.add_argument("--key", help="the list of ids inside the JSON object, if it is one")
    options = parser.parse_args()
    if REPOSITORY in options.out.resolve().parents:
        raise SystemExit("write the page outside the repository: it quotes the corpus")
    if not re.fullmatch(r"[a-z0-9_]+", options.collection):
        raise SystemExit("a collection name uses lowercase letters, digits and _")
    texts = load_texts(options.items)
    ids = load_ids(options.ids, options.key)
    missing = [item_id for item_id in ids if item_id not in texts]
    if missing:
        raise SystemExit(f"{len(missing)} items are not in {options.items}: {missing[:5]}")
    chosen = [{"id": item_id, "text": texts[item_id]} for item_id in ids]
    options.out.write_text(render_page(chosen, options.title, options.collection), encoding="utf-8")
    sys.stdout.write(f"{len(chosen)} records written to {options.out}\n")


if __name__ == "__main__":
    main()
