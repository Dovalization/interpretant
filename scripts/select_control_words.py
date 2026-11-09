"""Choose control words spanning the targets' frequency range (November 2026).

A control word is an everyday word whose meaning and use should not shift in academic prose
across the period. The noise floor itself comes from the whole vocabulary: for each pair of
cells, drift as a function of frequency over every word above the threshold in both. The
controls are a sanity check on it and should sit at or below that curve. They cannot carry
the floor themselves: everyday words are rare in abstracts, and most controls reach the
threshold in only a minority of cells.

The candidates are hand-picked everyday words; their counts come from the phrased main
corpus (books and the OpenAlex grid). Each target is paired with the unused candidate closest
to it in log frequency, so the controls span the targets' frequency range. Writes
data/control_words.txt with each control's count and the target it was paired with.

Candidates exclude everyday words with academic senses: table (data table), window (time
window), paper, field, hand (on the other hand), ground (ground truth), floor (floor
effect), body (embodiment), state, power, light, head, letter, book, cloud (cloud
computing), age (the information age), school (school of thought), family (a family of
models), man (generic "man" for humanity fell out of academic use). No everyday word is as
frequent as system, model or information.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from interpretant.corpus.audit import load_target_words

PHRASED_DIR = Path("data/phrased")
OUTPUT = Path("data/control_words.txt")
CANDIDATES = (  # noqa: SIM905
    "people woman day year month week hour minute death country home war hospital money "
    "son daughter boy girl car town water tree house river stone bread chair wall morning "
    "night city child mother father door road street horse dog food fire rain sea garden "
    "kitchen milk iron salt village mountain bed coat wood sun moon winter summer island "
    "bird fish egg apple sugar wine meat shoe boat bridge church farm forest lake hill grass "
    "flower snow wind soldier king doctor teacher student husband wife brother sister friend"
).split()


def corpus_counts(words: set[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in sorted(PHRASED_DIR.glob("*/texts/*.txt")):
        with path.open(encoding="utf-8") as text:
            for line in text:
                counts.update(token for token in line.split() if token in words)
    return counts


def main() -> None:
    targets = load_target_words(Path("data/words_of_interest.txt"))
    counts = corpus_counts(set(targets) | set(CANDIDATES))
    available = {word for word in CANDIDATES if counts[word] > 0}
    matches: list[tuple[str, str]] = []
    for target in sorted(targets, key=lambda word: -counts[word]):
        if not available or counts[target] == 0:
            continue
        control = min(
            available, key=lambda word: abs(math.log(counts[word]) - math.log(counts[target]))
        )
        available.discard(control)
        matches.append((control, target))
    lines = [
        "# Control words: everyday words expected to keep their meaning, chosen to span the",
        "# targets' frequency range on the phrased main corpus (scripts/select_control_words.py).",
        "# They check the whole-vocabulary noise floor: they should not drift above it.",
        "# word  count  matched_target  target_count",
    ]
    lines += [
        f"{control}  # {counts[control]:,}  {target}  {counts[target]:,}"
        for control, target in matches
    ]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
