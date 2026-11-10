"""Find the sense rounds' data gaps and download word-targeted cells where the grid is thin.

The plan is every word, field and decade the redesigned calibrations (*hallucination* and
*prompt*, the held-out known answers) and round D's open questions need. *attention*, the redesign's
development data, is not in it: it is read from round C's grid design as encoded. A word whose
planned cells all have at least ``MIN_USES`` abstracts containing it in the equal-size grid is
read from the grid. Any other word has all its planned cells downloaded by word, a seeded
sample of up to ``TARGET`` works whose abstract contains it, into
``data/raw/openalex_words/<word>/``, so each word's cells come from one source. Without
``--download`` the script only prints the gaps.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from interpretant.corpus.openalex import (
    BudgetExhaustedError,
    OpenAlexClient,
    OpenAlexSettings,
    download_cell,
)

TEXTS = Path("data/processed/openalex/texts")
WORDS_DIR = Path("data/raw/openalex_words")
MIN_USES = 500
TARGET = 1_000
DECADES = (1990, 2000, 2010, 2020)
PLAN: dict[str, tuple[tuple[str, ...], tuple[int, ...]]] = {
    "hallucination": (("ai", "compsci", "psychology", "neuroscience"), (2000, 2010, 2020)),
    "prompt": (("ai", "psychology"), (2000, 2020)),
    "information": (("psychology", "cogpsych", "compsci", "ai", "hci"), DECADES),
    "consciousness": (("philosophy", "psychology", "neuroscience", "ai"), DECADES),
    "embodiment": (("hci", "philosophy", "cogpsych"), DECADES),
    "intelligence": (
        ("ai", "cogpsych", "psychology", "communication", "histphilsci", "sociology", "philosophy"),
        DECADES,
    ),
}


def grid_count(word: str, field: str, decade: int) -> int:
    pattern = re.compile(rf"\b{re.escape(word)}\b")
    with (TEXTS / f"{field}_{decade}.txt").open(encoding="utf-8") as cell:
        return sum(1 for line in cell if pattern.search(line))


def main() -> None:
    download = "--download" in sys.argv
    client = OpenAlexClient(OpenAlexSettings().openalex_api_key)  # type: ignore[call-arg]
    for word, (fields, decades) in PLAN.items():
        counts = {(f, d): grid_count(word, f, d) for f in fields for d in decades}
        thin = sorted(cell for cell, count in counts.items() if count < MIN_USES)
        source = "grid" if not thin else f"download ({len(thin)} of {len(counts)} cells thin)"
        print(f"\n{word}: {source}")
        for field in fields:
            print(f"  {field:14}" + "".join(f"{counts[(field, d)]:>7}" for d in decades))
        if not thin or not download:
            continue
        for field in fields:
            for decade in decades:
                try:
                    works = download_cell(
                        client, WORDS_DIR / word, field, decade, TARGET, word=word
                    )
                except BudgetExhaustedError as error:
                    print(f"Daily budget spent ({error}); rerun to resume.")
                    return
                print(f"    downloaded {field} {decade}s: {works:,} works")


if __name__ == "__main__":
    main()
