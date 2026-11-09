"""Choose the cross-field control words fixed by prediction R8 (docs/predictions.md).

A cross-field control must be stable across fields as well as across time, so from
data/control_words.txt this keeps the 15 words whose per-million rate varies least across the
11 OpenAlex fields (lowest coefficient of variation, decades pooled). The choice uses counts
only, never drift, and writes data/control_words_cross_field.txt.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from interpretant.corpus.audit import load_target_words

TEXTS = Path("data/phrased/openalex/texts")
OUTPUT = Path("data/control_words_cross_field.txt")
KEEP = 15


def main() -> None:
    controls = set(load_target_words(Path("data/control_words.txt")))
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    tokens: Counter[str] = Counter()
    for path in sorted(TEXTS.glob("*.txt")):
        field = path.stem.rsplit("_", 1)[0]
        with path.open(encoding="utf-8") as text:
            for line in text:
                words = line.split()
                tokens[field] += len(words)
                counts[field].update(word for word in words if word in controls)
    fields = sorted(tokens)
    variation = {}
    for word in controls:
        rates = np.array([counts[field][word] / tokens[field] * 1e6 for field in fields])
        variation[word] = rates.std() / rates.mean() if rates.mean() else float("inf")
    chosen = sorted(variation, key=variation.__getitem__)[:KEEP]
    lines = [
        "# Cross-field control words (prediction R8): the 15 control words whose per-million",
        "# rate varies least across the 11 OpenAlex fields",
        "# (scripts/select_cross_field_controls.py).",
        "# word  coefficient_of_variation",
    ]
    lines += [f"{word}  # {variation[word]:.2f}" for word in chosen]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nmost variable (left out):", sorted(variation, key=variation.__getitem__)[-8:])


if __name__ == "__main__":
    main()
