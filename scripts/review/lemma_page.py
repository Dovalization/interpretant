"""The lemma review's judging page, from the list `scripts/evidence/lemma_review_list.py` writes.

The author's groups of `docs/lemma_review.md`, in this order and by frequency within each: the
100 most frequent disagreements, the 30 of the tail check and the kept-plural candidates. Each
form's candidate lemmas (spaCy's and the voted map's, one when they agree) are shown in an order
drawn at random (seed 20261116), without saying which lemmatiser gave which; the page records the
source of the chosen lemma beside it. Usage:
`python scripts/review/lemma_page.py REVIEW_LIST.parquet OUT.html`.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import polars as pl

SEED = 20261116
TEMPLATE = Path(__file__).with_name("lemma_template.html")
GROUPS = ("author", "tail_sample", "policy_keep")
HINTS = {
    "policy_keep": "The project keeps plurals that are words of their own when the singular is "
    "mostly an adjective (fundamentals, morals). Keep this one as it is, or merge it?"
}


def candidates(form: str, spacy_lemma: str, voted_lemma: str) -> list[tuple[str, str]]:
    """Each distinct candidate lemma other than the form, with its source."""
    if spacy_lemma == voted_lemma:
        return [("both", spacy_lemma)]
    pairs = [("spacy", spacy_lemma), ("voted", voted_lemma)]
    return [(source, lemma) for source, lemma in pairs if lemma != form]


def items(review_list: pl.DataFrame) -> list[dict[str, object]]:
    """The page's items, group by group in the order of GROUPS, by frequency within each."""
    rng = random.Random(SEED)
    order = {group: index for index, group in enumerate(GROUPS)}
    chosen = review_list.filter(pl.col("group").is_in(GROUPS)).sort(
        pl.col("group").replace_strict(order), pl.col("count"), descending=[False, True]
    )
    result: list[dict[str, object]] = []
    for row in chosen.iter_rows(named=True):
        pairs = candidates(row["form"], row["spacy_lemma"], row["voted_lemma"])
        rng.shuffle(pairs)
        result.append(
            {
                "id": row["form"],
                "form": row["form"],
                "count": row["count"],
                "group": row["group"],
                "hint": HINTS.get(row["group"], ""),
                "contexts": row["contexts"],
                "options": [lemma for _, lemma in pairs],
                "sources": [source for source, _ in pairs],
            }
        )
    return result


def main(review_list_path: Path, out_path: Path) -> None:
    """Fill the template with the items and write the page."""
    page_items = items(pl.read_parquet(review_list_path))
    page = TEMPLATE.read_text(encoding="utf-8").replace(
        "__ITEMS__", json.dumps(page_items, ensure_ascii=False)
    )
    out_path.write_text(page, encoding="utf-8")
    sys.stdout.write(f"{len(page_items)} forms written to {out_path}\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
