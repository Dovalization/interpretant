"""Check 1a: are round X's known changes present in the cells it reads? (docs/diagnostics.md)

Committed before it was run. Two steps, so the reading stays blind:

    uv run python scripts/check_known_presence.py sample   # writes the blind sample and its key
    uv run python scripts/check_known_presence.py score    # after labelling: shares and verdicts

`sample` writes data/diagnostics/check_1a_blind.tsv (item, word, the sense to judge, the context
with the occurrence between ⟦ and ⟧) and, separately, check_1a_key.tsv (item, case, decade);
the labeller fills data/diagnostics/check_1a_labels.tsv (item, label: yes, no or unclear)
without opening the key. `score` joins them and counts marker matches over every usage.
"""

from __future__ import annotations

import csv
import random
import re
import sys
from pathlib import Path

from interpretant.drift.senses import Sampling, cased_abstracts, drop_shared, sample_occurrences

GRID = Path("data/raw/openalex")
WORDS = Path("data/raw/openalex_words")
OUT = Path("data/diagnostics")
SAMPLING = Sampling(cap=150, plural=True, quality=True)
PER_CELL = 20
SEED = 2026

SENSES = {
    "attention": "the attention mechanism of neural networks",
    "prompt": "an instruction or input given to a language model",
    "platform": "a digital or social-media platform",
    "network": "an artificial neural network",
    "alignment": "aligning an AI system with human intentions, preferences or values",
}
MARKERS = {
    "attention": r"self-attention|self attention|cross-attention|transformers?|multi-head|multihead"
    r"|attention mechanisms?|attention layers?|attention weights?|attention heads?"
    r"|attention modules?",
    "prompt": r"language models?|large language|llms?|chatgpt|gpt|generative ai|chatbots?"
    r"|few-shot|zero-shot|in-context|prompt engineering|prompt tuning|prompt learning",
    "platform": r"social media|social networks?|social networking|facebook|twitter|youtube"
    r"|instagram|tiktok|wechat|weibo|reddit|digital platforms?|online platforms?"
    r"|platform economy|platform companies|platformisation|platformization",
    "network": r"neural networks?|neural nets?|connectionist|connectionism|backpropagation"
    r"|back-propagation|perceptrons?|hidden units?|hidden layers?|hopfield|boltzmann",
    "alignment": r"language models?|llms?|rlhf|human feedback|human preferences?|human values"
    r"|value alignment|ai alignment|ai safety",
}
CASES = {
    "attention, AI": ("attention", GRID, "ai", 2010, 2020, "documented"),
    "prompt, AI": ("prompt", WORDS / "prompt", "ai", 2000, 2020, "documented"),
    "platform, communication": ("platform", GRID, "communication", 2000, 2010, "documented"),
    "network, AI": ("network", GRID, "ai", 1980, 1990, "documented"),
    "alignment, AI": ("alignment", GRID, "ai", 2010, 2020, "documented"),
    "attention, AI (quiet)": ("attention", GRID, "ai", 1990, 2000, "quiet"),
    "attention, psychology (quiet)": ("attention", GRID, "psychology", 2010, 2020, "quiet"),
}


def usages(case: str) -> dict[int, list[tuple[str, int, int]]]:
    """The case's usages by decade: (context, start, end), as round X's scorer samples them."""
    word, root, field, early, late, _ = CASES[case]
    cells = drop_shared({(field, d): cased_abstracts(root, field, d) for d in (early, late)})
    found = sample_occurrences(cells, word, seed=1, sampling=SAMPLING)
    return {
        decade: [(o.context, o.start, o.end) for o in found if o.decade == decade]
        for decade in (early, late)
    }


def marked(word: str, context: str) -> bool:
    return re.search(rf"\b(?:{MARKERS[word]})\b", context, re.IGNORECASE) is not None


def sample() -> None:
    rng = random.Random(SEED)
    items = []
    for case, (word, *_rest) in CASES.items():
        for decade, found in usages(case).items():
            for context, start, end in rng.sample(found, min(PER_CELL, len(found))):
                shown = f"{context[:start]}⟦{context[start:end]}⟧{context[end:]}"
                items.append((case, decade, word, shown))
    rng.shuffle(items)
    OUT.mkdir(parents=True, exist_ok=True)
    with (
        (OUT / "check_1a_blind.tsv").open("w", encoding="utf-8", newline="") as blind,
        (OUT / "check_1a_key.tsv").open("w", encoding="utf-8", newline="") as key,
    ):
        blind_rows, key_rows = csv.writer(blind, delimiter="\t"), csv.writer(key, delimiter="\t")
        blind_rows.writerow(["item", "word", "sense", "context"])
        key_rows.writerow(["item", "case", "decade"])
        for number, (case, decade, word, shown) in enumerate(items, start=1):
            blind_rows.writerow([number, word, SENSES[word], shown.replace("\t", " ")])
            key_rows.writerow([number, case, decade])
    print(f"{len(items)} items → {OUT / 'check_1a_blind.tsv'} (key kept separately)")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as table:
        return list(csv.DictReader(table, delimiter="\t"))


def score() -> None:
    key = {row["item"]: (row["case"], int(row["decade"])) for row in read(OUT / "check_1a_key.tsv")}
    labels = {
        row["item"]: row["label"].strip().lower() for row in read(OUT / "check_1a_labels.tsv")
    }
    missing = set(key) - set(labels)
    if missing:
        raise SystemExit(f"{len(missing)} items unlabelled")
    read_share: dict[tuple[str, int], float] = {}
    counts: dict[tuple[str, int], tuple[int, int, int]] = {}
    for cell in sorted(set(key.values())):
        cell_labels = [labels[item] for item, value in key.items() if value == cell]
        yes, no = cell_labels.count("yes"), cell_labels.count("no")
        counts[cell] = (yes, no, cell_labels.count("unclear"))
        read_share[cell] = yes / (yes + no) if yes + no else float("nan")
    for case, (word, _root, _field, early, late, kind) in CASES.items():
        found = usages(case)
        mark = {d: sum(marked(word, c) for c, _, _ in found[d]) / len(found[d]) for d in found}
        mark_rise = mark[late] - mark[early]
        read_rise = read_share[(case, late)] - read_share[(case, early)]
        if kind == "documented":
            verdict = (
                "PRESENT"
                if mark_rise >= 0.10 and read_rise >= 0.10
                else ("ABSENT" if mark_rise < 0.10 and read_rise < 0.10 else "SHARES DISAGREE")
            )
        else:
            verdict = "QUIET" if abs(mark_rise) < 0.05 and abs(read_rise) < 0.10 else "NOT QUIET"
        print(f"\n{case} [{kind}]: {verdict}")
        for decade in (early, late):
            yes, no, unclear = counts[(case, decade)]
            print(
                f"  {decade}s: marked {mark[decade]:.2f} of {len(found[decade])}; read "
                f"{read_share[(case, decade)]:.2f} ({yes} yes, {no} no, {unclear} unclear)"
            )
        print(f"  rise: marked {mark_rise:+.2f}, read {read_rise:+.2f}")


if __name__ == "__main__":
    {"sample": sample, "score": score}[sys.argv[1]]()
