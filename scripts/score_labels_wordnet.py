"""Score round Z (Z1-Z4): round Y with WordNet sense lists, for each labeller separately.

Committed before it was run (docs/predictions_labels.md). Reuses round Y's usages, repeats and
measures from scripts/score_labels_benchmark.py, with the lists from
scripts/wordnet_sense_lists.py:

    uv run --extra labels python scripts/score_labels_wordnet.py sonnet
    uv run --extra labels python scripts/score_labels_wordnet.py jev
    uv run python scripts/score_labels_wordnet.py score   # needs round Y's labels too

Labels go to data/labels/round_z/ as DWUG identifiers and labels only.
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

sys.path.insert(0, str(Path(__file__).parent))
import score_labels_benchmark as round_y

from interpretant.drift.labels import (
    FAILED,
    REFUSED,
    REPEAT_SUFFIX,
    Item,
    SenseList,
    load_sense_lists,
)
from interpretant.drift.pairs import dwug_graded_change
from interpretant.predictions import report

LISTS = Path("data/sense_lists/dwug_en_wordnet.toml")
OUT = Path("data/labels/round_z")


def work() -> list[tuple[SenseList, Item]]:
    """Round Y's requests, the same usages and repeats, with the WordNet lists."""
    lists = load_sense_lists(LISTS)
    found = sorted(round_y.usages().items())
    items = [(lists[lemma], Item(key, usage)) for key, (lemma, _, usage) in found]
    rng = random.Random(round_y.SEED)
    repeats = [
        (senses, Item(item.key + REPEAT_SUFFIX, item.usage))
        for senses, item in rng.sample(items, round(round_y.REPEATS * len(items)))
    ]
    return items + repeats


def label(name: str) -> None:
    labels = round_y.LABELLERS[name].label(work(), OUT / "raw" / name)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{name}_labels.tsv").open("w", encoding="utf-8", newline="") as table:
        writer = csv.writer(table, delimiter="\t")
        writer.writerow(["usage", "label"])
        writer.writerows(sorted(labels.items()))
    failed = sum(value in (FAILED, REFUSED) for value in labels.values())
    print(f"{name}: {len(labels)} labels; failed or refused {failed}")


def read(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as table:
        return {row["usage"]: row["label"] for row in csv.DictReader(table, delimiter="\t")}


def spearman_and_rand(labels: dict[str, str], lists: dict[str, SenseList]) -> tuple[float, float]:
    found = round_y.usages()
    graded = dwug_graded_change(round_y.ROOT)
    words = sorted(graded)
    originals = {key: labels[key] for key in found}
    measured = round_y.divergences(originals, found, lists)
    rho = float(spearmanr([measured[w] for w in words], [graded[w] for w in words]).statistic)
    cluster_of = round_y.clusters()
    rand = []
    for word in words:
        keys = [key for key in found if found[key][0] == word]
        truth = [cluster_of[key] for key in keys]
        if len(set(truth)) >= 2:
            rand.append(adjusted_rand_score(truth, [originals[key] for key in keys]))
    return rho, float(np.mean(rand))


def score() -> None:
    wordnet_lists = load_sense_lists(LISTS)
    written_lists = load_sense_lists(round_y.LISTS)
    for name in round_y.LABELLERS:
        path = OUT / f"{name}_labels.tsv"
        if not path.exists():
            continue
        labels = read(path)
        rho, rand = spearman_and_rand(labels, wordnet_lists)
        rho_y, rand_y = spearman_and_rand(read(round_y.OUT / f"{name}_labels.tsv"), written_lists)
        repeats = [k for k in labels if k.endswith(REPEAT_SUFFIX)]
        consistency = statistics.mean(
            labels[k] == labels[k.removesuffix(REPEAT_SUFFIX)] for k in repeats
        )
        originals = [labels[k] for k in labels if not k.endswith(REPEAT_SUFFIX)]
        broken = statistics.mean(value in (FAILED, REFUSED) for value in originals)
        print(f"{name}: failed or refused {broken:.1%}")
        report(
            f"Z1 ({name}) the WordNet labels rank the benchmark's changes",
            rho >= 0.70,
            f"Spearman {rho:.3f} over 46 words (needed 0.70)",
        )
        report(
            f"Z2 ({name}) the lists do not drive round Y's result",
            abs(rho - rho_y) <= 0.10,
            f"round Z {rho:.3f}, round Y {rho_y:.3f}, difference {rho - rho_y:+.3f} "
            "(at most 0.10 either way)",
        )
        report(
            f"Z3 ({name}) the labels are consistent",
            consistency >= 0.85,
            f"{consistency:.1%} of {len(repeats)} repeated usages labelled the same (needed 85%)",
        )
        report(
            f"Z4 ({name}) grouping, compared",
            "descriptive",
            f"mean ARI {rand:.3f} with WordNet lists, {rand_y:.3f} with round Y's",
        )


if __name__ == "__main__":
    if sys.argv[1] == "score":
        score()
    else:
        label(sys.argv[1])
