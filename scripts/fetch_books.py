#!/usr/bin/env python3
"""Download freely available book texts into data/external/books/.

Run from the project root:
    uv run python scripts/fetch_books.py

Each book is fetched, cleaned, and written to the correct decade folder.
Already-present files are skipped unless --force is passed.
"""

from __future__ import annotations

import re
import sys
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

BOOKS_DIR = Path("data/external/books")
RATE_LIMIT_SECONDS = 1.5  # polite delay between requests


# ---------------------------------------------------------------------------
# Fetch + clean helpers
# ---------------------------------------------------------------------------


def _fetch(url: str) -> str:
    req = urllib.request.Request(
        url, headers={"User-Agent": "interpretant-corpus-fetcher/1.0 (research)"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


def _strip_gutenberg(text: str) -> str:
    """Remove Project Gutenberg header and footer boilerplate."""
    upper = text.upper()
    start_idx = 0
    for marker in ("*** START OF THE PROJECT GUTENBERG", "*** START OF THIS PROJECT GUTENBERG"):
        idx = upper.find(marker)
        if idx != -1:
            start_idx = text.find("\n", idx) + 1
            break
    end_idx = len(text)
    for marker in ("*** END OF THE PROJECT GUTENBERG", "*** END OF THIS PROJECT GUTENBERG"):
        idx = upper.find(marker)
        if idx != -1:
            end_idx = idx
            break
    return text[start_idx:end_idx].strip()


class _HtmlToText(HTMLParser):
    """Minimal HTML → plain text converter."""

    _BLOCK_TAGS = {"p", "div", "br", "h1", "h2", "h3", "h4", "h5", "li", "blockquote", "tr"}
    _SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._depth += 1
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS:
            self._depth = max(0, self._depth - 1)

    def handle_data(self, data: str) -> None:
        if self._depth == 0:
            self._parts.append(data)

    def result(self) -> str:
        raw = "".join(self._parts)
        return re.sub(r"\n{3,}", "\n\n", raw).strip()


def _html_to_text(html: str) -> str:
    parser = _HtmlToText()
    parser.feed(html)
    return parser.result()


def _strip_ia_watermarks(text: str) -> str:
    """Remove Internet Archive download watermark lines."""
    lines = text.splitlines()
    cleaned = [
        line for line in lines
        if not re.match(r"^\s*Downloaded from\s+https?://", line, re.IGNORECASE)
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


def fetch_gutenberg(ebook_id: str) -> str:
    url = f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt"
    return _strip_gutenberg(_fetch(url))


def fetch_html(url: str) -> str:
    return _html_to_text(_fetch(url))


def fetch_ia_text(url: str) -> str:
    return _strip_ia_watermarks(_fetch(url))


# ---------------------------------------------------------------------------
# Book definitions
# ---------------------------------------------------------------------------
# Each entry: (decade_label, filename, list_of_(label, url_or_callable))
# Multiple items in the last list are concatenated with a separator.

_SEP = "\n\n" + ("—" * 60) + "\n\n"

BOOKS: list[tuple[str, str, list[tuple[str, str]]]] = [
    (
        "1890s",
        "james_principles_psychology.txt",
        [
            ("James PoP Vol 1", "gutenberg:57628"),
            ("James PoP Vol 2", "gutenberg:57634"),
        ],
    ),
    (
        "1900s",
        "bergson_creative_evolution.txt",
        [("Bergson Creative Evolution", "gutenberg:26163")],
    ),
    (
        "1920s",
        "russell_analysis_mind.txt",
        [("Russell Analysis of Mind", "gutenberg:2529")],
    ),
    (
        "1920s",
        "wittgenstein_tractatus.txt",
        [
            (
                "Wittgenstein Tractatus",
                "html:https://standardebooks.org/ebooks/ludwig-wittgenstein/"
                "tractatus-logico-philosophicus/c-k-ogden/text/single-page",
            )
        ],
    ),
    (
        "1950s",
        "benjamin_illuminations.txt",
        [
            (
                "Benjamin Work of Art",
                "html:https://www.marxists.org/reference/subject/philosophy/works/ge/benjamin.htm",
            ),
            (
                "Benjamin Theses on History",
                "html:https://www.marxists.org/reference/archive/benjamin/1940/history.htm",
            ),
            (
                "Benjamin Author as Producer",
                "html:https://www.marxists.org/reference/archive/benjamin/1970/author-producer.htm",
            ),
        ],
    ),
    (
        "1980s",
        "baudrillard_simulacra_simulation.txt",
        [
            (
                "Baudrillard Simulacra",
                "ia:https://archive.org/download/"
                "simulacra-and-simulation-jean-baudrillard-translated-by-sheila-faria-glaser/"
                "Simulacra%20and%20Simulation%20-%20by%20Jean%20Baudrillard%3B"
                "%20translated%20by%20Sheila%20Faria%20Glaser_djvu.txt",
            )
        ],
    ),
    (
        "1990s",
        "mitchell_city_bits.txt",
        [
            (
                "Mitchell City of Bits",
                "ia:https://archive.org/download/"
                "mit_press_book_9780262279956/"
                "mit_press_book_9780262279956_djvu.txt",
            )
        ],
    ),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _fetch_part(scheme_url: str) -> str:
    if scheme_url.startswith("gutenberg:"):
        ebook_id = scheme_url.split(":", 1)[1]
        return fetch_gutenberg(ebook_id)
    elif scheme_url.startswith("html:"):
        url = scheme_url.split(":", 1)[1]
        return fetch_html(url)
    elif scheme_url.startswith("ia:"):
        url = scheme_url.split(":", 1)[1]
        return fetch_ia_text(url)
    else:
        raise ValueError(f"Unknown scheme in: {scheme_url!r}")


def main(force: bool = False) -> None:
    if not BOOKS_DIR.exists():
        print(f"ERROR: {BOOKS_DIR} does not exist. Run from project root.", file=sys.stderr)
        sys.exit(1)

    total = len(BOOKS)
    for i, (decade, filename, parts) in enumerate(BOOKS, 1):
        dest = BOOKS_DIR / decade / filename
        print(f"[{i}/{total}] {filename}", end="  ")

        if dest.exists() and not force:
            words = len(dest.read_text(encoding="utf-8").split())
            print(f"skip (already present, {words:,} words)")
            continue

        segments: list[str] = []
        for label, scheme_url in parts:
            print(f"\n         fetching {label} ...", end="", flush=True)
            try:
                text = _fetch_part(scheme_url)
                segments.append(text)
                words = len(text.split())
                print(f" {words:,} words", end="")
            except Exception as exc:  # noqa: BLE001
                print(f" ERROR: {exc}", file=sys.stderr)
                segments.append(f"[FETCH ERROR: {label}: {exc}]")
            time.sleep(RATE_LIMIT_SECONDS)

        combined = _SEP.join(segments)
        dest.write_text(combined, encoding="utf-8")
        total_words = len(combined.split())
        print(f"\n         → saved {total_words:,} words to {dest}")

    print("\nDone. Run `uv run interpretant corpus books validate` to check status.")


if __name__ == "__main__":
    force = "--force" in sys.argv
    main(force=force)
