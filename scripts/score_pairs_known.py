"""Score round X (X1-X5): does the usage-pair pipeline find this project's known changes?

Committed before it was run. For every transition, samples up to 150 usages per cell of the
known word and of round C's control words (interpretant.drift.senses, round C′'s quality rules),
encodes them once with XL-LEXEME (cached under data/drift/), computes APD and J, and judges each
known and quiet case against 200 shuffles of its own usages and against the transition's control
words, as docs/predictions_pairs.md defines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from interpretant.drift.pairs import (
    Usage,
    XLLexeme,
    across_minus_within,
    average_pairwise_distance,
)
from interpretant.drift.senses import Sampling, cased_abstracts, drop_shared, sample_occurrences
from interpretant.predictions import report

GRID = Path("data/raw/openalex")
WORDS = Path("data/raw/openalex_words")
CACHE = Path("data/drift/pairs_known.npz")
SAMPLING = Sampling(cap=150, plural=True, quality=True)
MIN_USAGES = 100
SHUFFLES = 200
MEASURES = {"APD": average_pairwise_distance, "J": across_minus_within}
CONTROLS = [
    line.split()[0]
    for line in Path("data/control_words_senses.txt").read_text(encoding="utf-8").splitlines()
    if line and not line.startswith("#")
]


@dataclass(frozen=True)
class Case:
    name: str
    word: str
    root: Path
    field: str
    early: int
    late: int
    kind: str  # "documented", "weak" or "quiet"


CASES = (
    Case("attention, AI 2010s→2020s", "attention", GRID, "ai", 2010, 2020, "documented"),
    Case("prompt, AI 2000s→2020s", "prompt", WORDS / "prompt", "ai", 2000, 2020, "documented"),
    Case(
        "platform, communication 2000s→2010s",
        "platform",
        GRID,
        "communication",
        2000,
        2010,
        "documented",
    ),
    Case("network, AI 1980s→1990s", "network", GRID, "ai", 1980, 1990, "documented"),
    Case("alignment, AI 2010s→2020s", "alignment", GRID, "ai", 2010, 2020, "weak"),
    Case("attention, AI 1990s→2000s", "attention", GRID, "ai", 1990, 2000, "quiet"),
    Case("attention, psychology 2010s→2020s", "attention", GRID, "psychology", 2010, 2020, "quiet"),
)


class Vectors:
    """Usage vectors by key, encoded once and cached across runs."""

    def __init__(self) -> None:
        self.stored: dict[str, np.ndarray] = dict(np.load(CACHE)) if CACHE.exists() else {}
        self.encoder: XLLexeme | None = None

    def get(self, key: str, usages: list[Usage]) -> np.ndarray:
        if key not in self.stored:
            if self.encoder is None:
                self.encoder = XLLexeme()
            self.stored[key] = self.encoder.encode(usages)
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            np.savez(CACHE, **self.stored)  # type: ignore[arg-type]
        return self.stored[key]


def pair(
    vectors: Vectors, root: Path, field: str, early: int, late: int, word: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """The word's usage vectors in the two cells, or None if either has under MIN_USAGES."""
    cells = drop_shared(
        {(field, decade): cased_abstracts(root, field, decade) for decade in (early, late)}
    )
    found = sample_occurrences(cells, word, seed=1, sampling=SAMPLING)
    by_decade = {
        decade: [Usage(o.context, o.start, o.end) for o in found if o.decade == decade]
        for decade in (early, late)
    }
    if min(len(usages) for usages in by_decade.values()) < MIN_USAGES:
        return None
    key = f"{root.name}|{field}|{early}|{late}|{word}"
    return (
        vectors.get(f"{key}|{early}", by_decade[early]),
        vectors.get(f"{key}|{late}", by_decade[late]),
    )


def shuffled(first: np.ndarray, second: np.ndarray, seed: int) -> dict[str, list[float]]:
    """Each measure over SHUFFLES random reassignments of the pooled usages to the two cells."""
    rng = np.random.default_rng(seed)
    pooled = np.vstack([first, second])
    values: dict[str, list[float]] = {name: [] for name in MEASURES}
    for _ in range(SHUFFLES):
        order = rng.permutation(len(pooled))
        a, b = pooled[order[: len(first)]], pooled[order[len(first) :]]
        for name, measure in MEASURES.items():
            values[name].append(measure(a, b))
    return values


def main() -> None:
    vectors = Vectors()
    transitions = sorted({(case.field, case.early, case.late) for case in CASES})
    controls: dict[tuple[str, int, int], dict[str, list[float]]] = {}
    passes_a: list[bool] = []
    for index, (field, early, late) in enumerate(transitions):
        values: dict[str, list[float]] = {name: [] for name in MEASURES}
        for offset, word in enumerate(CONTROLS):
            measured = pair(vectors, GRID, field, early, late, word)
            if measured is None:
                continue
            for name, measure in MEASURES.items():
                values[name].append(measure(*measured))
            null = shuffled(*measured, seed=2026 + 1000 * index + offset)
            passes_a.append(bool(values["J"][-1] > np.percentile(null["J"], 95)))
        controls[(field, early, late)] = values
        print(f"{field} {early}s→{late}s: {len(values['J'])} controls measured", flush=True)

    found: dict[str, dict[str, bool]] = {}
    for index, case in enumerate(CASES):
        measured = pair(vectors, case.root, case.field, case.early, case.late, case.word)
        if measured is None:
            print(f"{case.name}: not measured (under {MIN_USAGES} usages)")
            continue
        null = shuffled(*measured, seed=9000 + index)
        found[case.name] = {}
        parts = []
        for name, measure in MEASURES.items():
            value = measure(*measured)
            shuffle_bar = float(np.percentile(null[name], 95))
            control_bar = float(
                np.percentile(controls[(case.field, case.early, case.late)][name], 90)
            )
            found[case.name][name] = value > shuffle_bar and value > control_bar
            parts.append(
                f"{name} {value:.4f} (shuffles' 95th {shuffle_bar:.4f}, controls' 90th "
                f"{control_bar:.4f}){' found' if found[case.name][name] else ''}"
            )
        print(f"{case.name} [{case.kind}]: " + "; ".join(parts))

    documented = [case.name for case in CASES if case.kind == "documented"]
    quiet = [case.name for case in CASES if case.kind == "quiet"]
    for number, name in ((1, "APD"), (3, "J")):
        hits = [case for case in documented if found.get(case, {}).get(name)]
        report(
            f"X{number} {name} finds the documented changes",
            len(hits) >= 3,
            f"{len(hits)} of {len(documented)}: {', '.join(hits) or 'none'}",
        )
        flagged = [case for case in quiet if found.get(case, {}).get(name)]
        report(
            f"X{number + 1} {name} stays quiet where nothing documented changed",
            not flagged,
            f"flagged: {', '.join(flagged) or 'none'}",
        )
    share = sum(passes_a) / len(passes_a)
    report(
        "X5 every word drifts between decades",
        share > 0.30,
        f"J above its shuffles' 95th percentile in {sum(passes_a)} of {len(passes_a)} control "
        f"transitions ({share:.0%})",
    )
    weak = found.get("alignment, AI 2010s→2020s", {})
    print(
        f"\nalignment (weak case, counts towards nothing): APD {weak.get('APD')}, J {weak.get('J')}"
    )


if __name__ == "__main__":
    main()
