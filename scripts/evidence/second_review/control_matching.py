"""A12's matching rule: how wide must the frequency tolerance be to give 30 (and 60) same-POS
candidates matched in BOTH periods, for the planned comparisons? Counts from the existing
lemmatised cells (about 10,000 abstracts each); frequency = share of abstracts containing the word,
since one usage is drawn per abstract. Part of speech approximated by each word's majority tag in
the lemma votes table (nouns only here: every target below is a noun)."""

from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

TEXTS = Path("data/lemmatised/openalex/texts")
COMPARISONS = [
    ("attention", "ai", 2010, 2020),
    ("alignment", "ai", 2010, 2020),
    ("prompt", "ai", 2010, 2020),
    ("network", "ai", 1980, 1990),
    ("platform", "communication", 2000, 2010),
    ("attention", "psychology", 2010, 2020),
    ("hallucination", "psychology", 2010, 2020),
    ("component", "ai", 2010, 2020),
]
FLOOR = 100  # abstracts containing the candidate in each period (A10: never below the floor)

votes = pl.read_parquet("data/processed/lemma_votes.parquet")
by_lemma = (
    votes.with_columns(noun=pl.col("tag").is_in(["NN", "NNS"]))
    .group_by("lemma")
    .agg(total=pl.col("count").sum(), nouns=(pl.col("count") * pl.col("noun")).sum())
    .filter(pl.col("nouns") / pl.col("total") >= 0.5)
)
nouns = set(by_lemma["lemma"].to_list())


def document_frequency(field: str, decade: int) -> tuple[Counter[str], int]:
    counts: Counter[str] = Counter()
    abstracts = 0
    with (TEXTS / f"{field}_{decade}.txt").open(encoding="utf-8") as lines:
        for line in lines:
            abstracts += 1
            counts.update(set(line.split()))
    return counts, abstracts


cache: dict[tuple[str, int], tuple[Counter[str], int]] = {}
print(f"{'comparison':<38}{'abstracts with target':>22}{'tolerance for 30':>18}{'for 60':>9}"
      f"{'within ±15%':>13}")
for word, field, early, late in COMPARISONS:
    for decade in (early, late):
        if (field, decade) not in cache:
            cache[(field, decade)] = document_frequency(field, decade)
    (first, size_first), (second, size_second) = cache[(field, early)], cache[(field, late)]
    target = np.array([first[word] / size_first, second[word] / size_second])
    if target.min() == 0:
        print(f"{word + ' ' + field + f' {early}->{late}':<38}{f'{first[word]} / {second[word]}':>22}"
              "   target absent in a period")
        continue
    gaps = []
    for candidate in nouns:
        if candidate == word or first[candidate] < FLOOR or second[candidate] < FLOOR:
            continue
        shares = np.array([first[candidate] / size_first, second[candidate] / size_second])
        gaps.append(float(np.max(np.abs(shares - target) / target)))
    gaps.sort()
    within = sum(gap <= 0.15 for gap in gaps)
    need = lambda count: f"±{gaps[count - 1]:.0%}" if len(gaps) >= count else f"only {len(gaps)}"  # noqa: E731
    print(f"{word + ' ' + field + f' {early}->{late}':<38}{f'{first[word]} / {second[word]}':>22}"
          f"{need(30):>18}{need(60):>9}{within:>13}")
