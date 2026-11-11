"""Render a review page from build.py's output.

    uv run python scripts/review/render.py PREVIEWS.json OUT.html "Page title" [INTRO]

The page is the template with the evidence embedded; decisions are kept in the published
artifact's own store (the `db` capability), not in the file. INTRO, if given, replaces the first
sentence of the page's introduction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TEMPLATE = Path(__file__).with_name("template.html")
DEFAULT_INTRO = "Each word below has a draft list of meanings for the interpretant corpus."


def main() -> None:
    previews, out, title = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    data = json.loads(previews.read_text(encoding="utf-8"))
    embedded = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__TITLE__", title)
    html = html.replace("__DATA__", embedded)
    if len(sys.argv) > 4:
        html = html.replace(DEFAULT_INTRO, sys.argv[4])
    out.write_text(html, encoding="utf-8")
    print(f"{sum(len(e['senses']) for e in data.values())} meanings → {out}")


if __name__ == "__main__":
    main()
