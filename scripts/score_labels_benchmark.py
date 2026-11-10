"""Score round Y (Y1-Y5): sense labels on DWUG EN 2.0.1, for each labeller separately.

Committed before it was run (docs/predictions_labels.md). Three steps:

    uv run --extra labels python scripts/score_labels_benchmark.py sonnet   # Batch API
    uv run --extra labels python scripts/score_labels_benchmark.py jev      # TypeSafe API
    uv run python scripts/score_labels_benchmark.py score

Labels are written to data/labels/round_y/ as DWUG identifiers and labels only; raw responses
stay under data/labels/round_y/raw/, which is not committed (DWUG's licence forbids
redistributing its text).
"""

from __future__ import annotations

import csv
import random
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from interpretant.drift.labels import (
    FAILED,
    OTHER,
    REFUSED,
    REPEAT_SUFFIX,
    UNCLEAR,
    BatchLabeller,
    Item,
    JevLabeller,
    SenseList,
    distribution,
    divergence,
    load_sense_lists,
)
from interpretant.drift.pairs import Usage, dwug_graded_change
from interpretant.predictions import report

ROOT = Path("data/external/dwug_en_2.0.1/dwug_en")
LISTS = Path("data/sense_lists/dwug_en.toml")
OUT = Path("data/labels/round_y")
LABELLERS: dict[str, BatchLabeller | JevLabeller] = {
    "sonnet": BatchLabeller(model="claude-sonnet-5-5", effort="low"),
    "jev": JevLabeller(model="jev-1.14.0"),
}
REPEATS = 0.1
SEED = 2026
SHUFFLES = 20


def clusters() -> dict[str, int]:
    """DWUG's cluster of every usage, by ``lemma|identifier``."""
    found = {}
    for path in sorted((ROOT / "clusters" / "opt").glob("*.csv")):
        with path.open(encoding="utf-8") as table:
            for row in csv.DictReader(table, delimiter="\t"):
                found[f"{path.stem}|{row['identifier']}"] = int(row["cluster"])
    return found


def usages() -> dict[str, tuple[str, int, Usage]]:
    """Every usage DWUG kept (cluster other than -1), by ``lemma|identifier``: lemma, period,
    usage."""
    kept = {key for key, cluster in clusters().items() if cluster != -1}
    found = {}
    for path in sorted((ROOT / "data").glob("*/uses.csv")):
        with path.open(encoding="utf-8") as table:
            for row in csv.DictReader(table, delimiter="\t", quoting=csv.QUOTE_NONE):
                key = f"{row['lemma']}|{row['identifier']}"
                if key in kept:
                    start, end = (int(i) for i in row["indexes_target_token"].split(":"))
                    found[key] = (
                        row["lemma"],
                        int(row["grouping"]),
                        Usage(row["context"], start, end),
                    )
    return found


def work() -> list[tuple[SenseList, Item]]:
    lists = load_sense_lists(LISTS)
    found = sorted(usages().items())
    items = [(lists[lemma], Item(key, usage)) for key, (lemma, _, usage) in found]
    rng = random.Random(SEED)
    repeats = [
        (senses, Item(item.key + REPEAT_SUFFIX, item.usage))
        for senses, item in rng.sample(items, round(REPEATS * len(items)))
    ]
    return items + repeats


def label(name: str) -> None:
    labels = LABELLERS[name].label(work(), OUT / "raw" / name)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{name}_labels.tsv").open("w", encoding="utf-8", newline="") as table:
        writer = csv.writer(table, delimiter="\t")
        writer.writerow(["usage", "label"])
        writer.writerows(sorted(labels.items()))
    counts: dict[str, int] = {}
    for value in labels.values():
        counts[value] = counts.get(value, 0) + 1
    print(
        f"{name}: {len(labels)} labels; failed {counts.get(FAILED, 0)}, "
        f"refused {counts.get(REFUSED, 0)}"
    )


def read(name: str) -> dict[str, str]:
    with (OUT / f"{name}_labels.tsv").open(encoding="utf-8") as table:
        return {row["usage"]: row["label"] for row in csv.DictReader(table, delimiter="\t")}


def divergences(
    labels: dict[str, str], found: dict[str, tuple[str, int, Usage]], lists: dict[str, SenseList]
) -> dict[str, float]:
    by_word: dict[str, dict[int, list[str]]] = {}
    for key, (lemma, period, _) in found.items():
        by_word.setdefault(lemma, {}).setdefault(period, []).append(labels[key])
    return {
        lemma: divergence(
            distribution(periods[1], lists[lemma]), distribution(periods[2], lists[lemma])
        )
        for lemma, periods in by_word.items()
    }


def score_one(name: str) -> dict[str, str]:
    lists = load_sense_lists(LISTS)
    found = usages()
    labels = read(name)
    originals = {key: labels[key] for key in found}
    graded = dwug_graded_change(ROOT)
    words = sorted(graded)

    measured = divergences(originals, found, lists)
    rho = float(spearmanr([measured[w] for w in words], [graded[w] for w in words]).statistic)

    cluster_of = clusters()
    rand = []
    for word in words:
        keys = [key for key in found if found[key][0] == word]
        truth = [cluster_of[key] for key in keys]
        if len(set(truth)) >= 2:
            rand.append(adjusted_rand_score(truth, [originals[key] for key in keys]))

    shuffled = []
    for seed in range(SHUFFLES):
        rng = random.Random(seed)
        moved = {}
        for word in words:
            keys = [key for key in found if found[key][0] == word]
            periods = [found[key][1] for key in keys]
            rng.shuffle(periods)
            moved.update(
                {
                    key: (word, period, found[key][2])
                    for key, period in zip(keys, periods, strict=True)
                }
            )
        values = divergences(originals, moved, lists)
        shuffled.append(
            float(spearmanr([values[w] for w in words], [graded[w] for w in words]).statistic)
        )

    repeats = [key for key in labels if key.endswith(REPEAT_SUFFIX)]
    consistency = statistics.mean(
        labels[k] == labels[k.removesuffix(REPEAT_SUFFIX)] for k in repeats
    )
    shares = {
        value: statistics.mean(label == value for label in originals.values())
        for value in (OTHER, UNCLEAR, FAILED, REFUSED)
    }
    print(
        f"{name}: {len(originals)} usages; other {shares[OTHER]:.1%}, "
        f"unclear {shares[UNCLEAR]:.1%}, failed {shares[FAILED]:.1%}, refused {shares[REFUSED]:.1%}"
    )
    measurable = shares[FAILED] + shares[REFUSED] <= 0.05
    if not measurable:
        print(f"{name}: more than 5% failed or refused; not measured on this round")
    report(
        f"Y1 ({name}) the labels rank the benchmark's changes",
        measurable and rho >= 0.70,
        f"Spearman {rho:.3f} over {len(words)} words (needed 0.70; BERT 0.563, XL-LEXEME 0.864)",
    )
    report(
        f"Y2 ({name}) the labels group usages as the annotators did",
        measurable and float(np.mean(rand)) >= 0.273,
        f"mean ARI {np.mean(rand):.3f} over {len(rand)} words with two or more clusters "
        "(needed 0.273; GPT-4 0.340, human 0.633)",
    )
    report(
        f"Y3 ({name}) the labels measure change, not variety",
        measurable and statistics.mean(shuffled) <= 0.20,
        f"mean Spearman with periods shuffled {statistics.mean(shuffled):.3f} over {SHUFFLES} "
        f"shuffles (at most 0.20; range {min(shuffled):.3f} to {max(shuffled):.3f})",
    )
    report(
        f"Y4 ({name}) the labels are consistent",
        measurable and consistency >= 0.85,
        f"{consistency:.1%} of {len(repeats)} repeated usages labelled the same (needed 85%)",
    )
    return originals


def score() -> None:
    found = {name: score_one(name) for name in LABELLERS if (OUT / f"{name}_labels.tsv").exists()}
    if len(found) == 2:
        sonnet, jev = found["sonnet"], found["jev"]
        agreement = statistics.mean(sonnet[key] == jev[key] for key in sonnet)
        report(
            "Y5 the two labellers agree",
            agreement >= 0.80,
            f"same label on {agreement:.1%} of {len(sonnet)} usages (at least 80%; descriptive)",
        )


if __name__ == "__main__":
    if sys.argv[1] == "score":
        score()
    else:
        label(sys.argv[1])
