"""Choose the control words fixed by round C (docs/predictions_senses.md).

A control is a word whose sense change in AI and psychology shows what a typical word's looks
like, so the controls are drawn at random, not chosen: 30 words, with a fixed seed, from every
word that appears in at least 100 abstracts in each of AI's and psychology's 1990s and 2020s
cells. Stopwords, words under four letters, words with digits and the round's target words are
left out. A hyphenated token counts for each of its parts, as the sense pipeline matches a word
inside a compound. The choice uses counts only and writes data/control_words_senses.txt.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TEXTS = Path("data/processed/openalex/texts")
OUTPUT = Path("data/control_words_senses.txt")
CELLS = (("ai", 1990), ("ai", 2020), ("psychology", 1990), ("psychology", 2020))
MIN_ABSTRACTS = 100
KEEP = 30
SEED = 2026
TARGETS = {"attention", "intelligence"}
WORD = re.compile(r"[a-z]{4,}")


def abstract_counts(path: Path) -> Counter[str]:
    """How many abstracts of a cell contain each word."""
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as text:
        for line in text:
            counts.update({part for token in line.split() for part in token.split("-")})
    return counts


def main() -> None:
    counts = [abstract_counts(TEXTS / f"{field}_{decade}.txt") for field, decade in CELLS]
    eligible = sorted(
        word
        for word in counts[0]
        if WORD.fullmatch(word)
        and word not in ENGLISH_STOP_WORDS
        and word not in TARGETS
        and all(cell[word] >= MIN_ABSTRACTS for cell in counts)
    )
    chosen = sorted(random.Random(SEED).sample(eligible, KEEP))
    header = [
        "# Round C control words: 30 drawn at random (seed 2026) from the words in at least",
        "# 100 abstracts in each of AI's and psychology's 1990s and 2020s cells",
        f"# ({len(eligible)} words; scripts/select_sense_controls.py).",
        "# word  abstracts in ai_1990 ai_2020 psychology_1990 psychology_2020",
    ]
    rows = [f"{word}  # {' '.join(str(cell[word]) for cell in counts)}" for word in chosen]
    OUTPUT.write_text("\n".join(header + rows) + "\n", encoding="utf-8")
    print(f"{len(eligible)} eligible; wrote {KEEP} to {OUTPUT}")


if __name__ == "__main__":
    main()
