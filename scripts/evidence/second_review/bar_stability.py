"""Round X's stored vectors (150 usages per cell, 30 control words per transition): how stable is
"the controls' 95th percentile" as a bar, and does a control's J depend on its frequency once the
number of usages is fixed?"""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from interpretant.drift.pairs import across_minus_within

stored = np.load("data/drift/pairs_known.npz")
controls = {
    line.split("#")[0].strip()
    for line in Path("data/control_words_senses.txt").read_text(encoding="utf-8").splitlines()
    if line.split("#")[0].strip()
}
TARGETS = {"attention", "network", "platform", "alignment"}
controls -= TARGETS
TEXTS = Path("data/lemmatised/openalex/texts")

by_transition: dict[tuple[str, str, str], dict[str, dict[str, np.ndarray]]] = defaultdict(dict)
for key in stored.files:
    source, field, early, late, word, decade = key.split("|")
    if source != "openalex":
        continue
    by_transition[(field, early, late)].setdefault(word, {})[decade] = stored[key]


def share_of_abstracts(field: str, decade: str) -> tuple[Counter[str], int]:
    counts: Counter[str] = Counter()
    abstracts = 0
    with (TEXTS / f"{field}_{decade}.txt").open(encoding="utf-8") as lines:
        for line in lines:
            abstracts += 1
            counts.update(set(line.split()))
    return counts, abstracts


print(f"{'transition':<28}{'controls':>9}{'95th pct':>10}{'if one control is dropped':>28}"
      f"{'top two controls':>30}")
points = []
for (field, early, late), words in sorted(by_transition.items()):
    scores = {}
    for word, cells in words.items():
        if word in controls and early in cells and late in cells:
            n = min(len(cells[early]), len(cells[late]))
            if n < 100:
                continue
            scores[word] = across_minus_within(cells[early][:n], cells[late][:n])
    if len(scores) < 20:
        continue
    values = np.array(list(scores.values()))
    bar = np.percentile(values, 95)
    dropped = [np.percentile(np.delete(values, index), 95) for index in range(len(values))]
    top = sorted(scores.items(), key=lambda item: -item[1])[:2]
    print(f"{field + ' ' + early + '->' + late:<28}{len(values):>9}{bar:>10.4f}"
          f"{f'{min(dropped):.4f} to {max(dropped):.4f}':>28}"
          f"{', '.join(f'{word} {value:.4f}' for word, value in top):>30}")
    ranked = np.sort(values)
    for word, cells in words.items():
        if word in TARGETS and early in cells and late in cells:
            n = min(len(cells[early]), len(cells[late]))
            value = across_minus_within(cells[early][:n], cells[late][:n])
            beaten = int((ranked < value).sum())
            print(f"    target {word}: J {value:.4f}, above {beaten} of {len(values)} controls, "
                  f"rank p-value {(len(values) - beaten + 1) / (len(values) + 1):.3f}")
    (first, size_first), (second, size_second) = share_of_abstracts(field, early), share_of_abstracts(field, late)
    for word, value in scores.items():
        if first[word] and second[word]:
            early_share, late_share = first[word] / size_first, second[word] / size_second
            points.append((value, np.log(np.sqrt(early_share * late_share)), abs(np.log(late_share / early_share))))

table = np.array(points)
print(f"\ncontrols pooled over transitions: {len(table)} (word, transition) pairs, 150 usages per cell")
print(f"  Spearman of J with log frequency (mean of the two periods): "
      f"{stats.spearmanr(table[:, 0], table[:, 1]).statistic:+.2f} "
      f"(p = {stats.spearmanr(table[:, 0], table[:, 1]).pvalue:.3f})")
print(f"  Spearman of J with |log frequency change| between periods:  "
      f"{stats.spearmanr(table[:, 0], table[:, 2]).statistic:+.2f} "
      f"(p = {stats.spearmanr(table[:, 0], table[:, 2]).pvalue:.3f})")
