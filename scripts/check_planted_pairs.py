"""Check 1b: how large a change can the usage-pair instrument see in this corpus?

Committed before it was run (docs/diagnostics.md). Plants AI 2020s usages of *attention* that
carry check 1a's attention-mechanism markers into psychology's 2020s cell (round X's quiet
transition) at five shares, five draws each, plus a placebo of unmarked AI usages, and judges
each planted pair with round X's two bars: its own shuffles and psychology's control words.
Base and control vectors come from round X's cache; planted usages are encoded once and cached.
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import score_pairs_known as round_x

from interpretant.drift.pairs import Usage, XLLexeme, across_minus_within
from interpretant.drift.senses import Sampling, cased_abstracts, sample_occurrences
from interpretant.predictions import report

CACHE = Path("data/drift/planted_attention.npz")
SHARES = (0.1, 0.2, 0.3, 0.4, 0.6)
PLACEBO_SHARE = 0.4
DRAWS = 5
SEED = 2026
# Check 1a's attention-mechanism markers (moved here when check 1a's script was deleted).
MECHANISM_MARKERS = (
    r"self-attention|self attention|cross-attention|transformers?|multi-head|multihead"
    r"|attention mechanisms?|attention layers?|attention weights?|attention heads?"
    r"|attention modules?"
)


def has_mechanism_marker(context: str) -> bool:
    """Whether a context carries one of check 1a's attention-mechanism markers."""
    return re.search(rf"\b(?:{MECHANISM_MARKERS})\b", context, re.IGNORECASE) is not None


def planting_pool() -> dict[str, np.ndarray]:
    """Vectors of every AI 2020s usage of *attention*, split by check 1a's markers."""
    if CACHE.exists():
        return dict(np.load(CACHE))
    cell = {("ai", 2020): cased_abstracts(round_x.GRID, "ai", 2020)}
    found = sample_occurrences(
        cell, "attention", seed=1, sampling=Sampling(cap=100_000, plural=True, quality=True)
    )
    usages = [Usage(o.context, o.start, o.end) for o in found]
    marked = [u for u in usages if has_mechanism_marker(u.text)]
    unmarked = [u for u in usages if not has_mechanism_marker(u.text)]
    encoder = XLLexeme()
    pool = {"marked": encoder.encode(marked), "unmarked": encoder.encode(unmarked)}
    np.savez(CACHE, **pool)  # type: ignore[arg-type]
    return pool


def judged(early: np.ndarray, late: np.ndarray, control_bar: float, seed: int) -> bool:
    """Round X's rule for J: above its shuffles' 95th percentile and the controls' 90th."""
    value = across_minus_within(early, late)
    shuffle_bar = float(np.percentile(round_x.shuffled(early, late, seed)["J"], 95))
    return value > shuffle_bar and value > control_bar


def planted(late: np.ndarray, source: np.ndarray, share: float, rng: random.Random) -> np.ndarray:
    count = round(share * len(late))
    result = late.copy()
    positions = rng.sample(range(len(late)), count)
    donors = rng.sample(range(len(source)), count)
    result[positions] = source[donors]
    return result


def main() -> None:
    vectors = round_x.Vectors()
    base = round_x.pair(vectors, round_x.GRID, "psychology", 2010, 2020, "attention")
    assert base is not None
    early, late = base
    controls = [
        across_minus_within(*measured)
        for word in round_x.CONTROLS
        for measured in [round_x.pair(vectors, round_x.GRID, "psychology", 2010, 2020, word)]
        if measured is not None
    ]
    control_bar = float(np.percentile(controls, 90))
    pool = planting_pool()
    print(
        f"base: {len(early)} and {len(late)} usages; {len(controls)} controls, 90th percentile "
        f"{control_bar:.4f}; pool {len(pool['marked'])} marked, {len(pool['unmarked'])} unmarked"
    )

    found: dict[float, int] = {}
    for share in SHARES:
        hits = 0
        for draw in range(DRAWS):
            rng = random.Random(f"{SEED}:{share}:{draw}")
            hits += judged(
                early, planted(late, pool["marked"], share, rng), control_bar, rng.randrange(10**6)
            )
        found[share] = hits
        print(f"  mechanism planted at {share:.0%}: found in {hits} of {DRAWS} draws")
    placebo = 0
    for draw in range(DRAWS):
        rng = random.Random(f"{SEED}:placebo:{draw}")
        placebo += judged(
            early,
            planted(late, pool["unmarked"], PLACEBO_SHARE, rng),
            control_bar,
            rng.randrange(10**6),
        )
    print(f"  placebo (unmarked) at {PLACEBO_SHARE:.0%}: found in {placebo} of {DRAWS} draws")

    detected = [share for share in SHARES if found[share] >= 4]
    smallest = min(detected) if detected else None
    report(
        "P1 the instrument sees a planted change of a third or less",
        smallest is not None
        and smallest <= 0.3
        and all(s in detected for s in SHARES if s >= smallest),
        f"smallest detected share {smallest if smallest is not None else 'none'}; "
        + ", ".join(f"{s:.0%}: {found[s]}/{DRAWS}" for s in SHARES),
    )
    report(
        "P2 it does not see a change of source without a change of sense",
        placebo <= 1,
        f"placebo found in {placebo} of {DRAWS} draws",
    )


if __name__ == "__main__":
    main()
