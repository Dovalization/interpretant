"""The lemma review's judging page, from the review list and the agent judges' verdicts.

The list is the one `scripts/evidence/lemma_review_list.py` writes. Three parts, in this order
(`docs/lemma_review.md`):

1. Blind: the author's groups, the 100 most frequent disagreements, the 30 of the tail check and
   the kept-plural candidates, by frequency within each. Each form's candidate lemmas (spaCy's and
   the voted map's, one when they agree) are shown in an order drawn at random (seed 20261116),
   without saying which lemmatiser gave which; the page records the source of the chosen lemma.
2. Second look: once every blind form is answered, the forms where the author and the agents
   differ, with both answers. The blind answers stay as they are; the second look's is final.
3. The agents' calls: the most frequent suspect merges and homographs, with the agents' lemma
   suggested; then, since the calibration failed (2026-11-17), the follow-up review: every other
   suspect or homograph with at least 50 uses where the agents changed spaCy's lemma or merged a
   plural into a singular that is mostly an adjective, most frequent first.

Usage: `python scripts/review/lemma_page.py REVIEW_LIST.parquet AGENTS.json OUT.html`, where
AGENTS.json maps each judged form to `{"lemma": ..., "reason": ...}`, the agents' final lemma and,
where the Opus judge settled the form, its reason.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import polars as pl

SEED = 20261116
TEMPLATE = Path(__file__).with_name("lemma_template.html")
BLIND_GROUPS = ("author", "tail_sample", "policy_keep")
CALL_GROUPS = ("suspect_merge", "homograph")
CALLS = 50
FOLLOW_UP_MIN_USES = 50
ADJECTIVE_SHARE = 0.5
DOCUMENT_ID = re.compile(r"[A-Za-z0-9_\-.~:@+]{1,200}")
HINTS = {
    "policy_keep": "The project keeps plurals that are words of their own when the singular is "
    "mostly an adjective (fundamentals, morals). Keep this one as it is, or merge it?"
}

Verdicts = dict[str, dict[str, str]]


def candidates(form: str, spacy_lemma: str, voted_lemma: str) -> list[tuple[str, str]]:
    """Each distinct candidate lemma other than the form, with its source."""
    if spacy_lemma == voted_lemma:
        return [("both", spacy_lemma)]
    pairs = [("spacy", spacy_lemma), ("voted", voted_lemma)]
    return [(source, lemma) for source, lemma in pairs if lemma != form]


def blind_items(review_list: pl.DataFrame, agents: Verdicts) -> list[dict[str, object]]:
    """The blind part, group by group in the order of BLIND_GROUPS, by frequency within each."""
    rng = random.Random(SEED)
    order = {group: index for index, group in enumerate(BLIND_GROUPS)}
    chosen = review_list.filter(pl.col("group").is_in(BLIND_GROUPS)).sort(
        pl.col("group").replace_strict(order), pl.col("count"), descending=[False, True]
    )
    result: list[dict[str, object]] = []
    for row in chosen.iter_rows(named=True):
        pairs = candidates(row["form"], row["spacy_lemma"], row["voted_lemma"])
        rng.shuffle(pairs)
        verdict = agents[row["form"]]
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
                "agents": verdict["lemma"],
                "reason": verdict.get("reason", ""),
            }
        )
    return result


def call_rows(review_list: pl.DataFrame, agents: Verdicts) -> pl.DataFrame:
    """The CALLS most frequent suspect merges and homographs, then the follow-up review's forms."""
    pool = review_list.filter(pl.col("group").is_in(CALL_GROUPS)).sort("count", descending=True)
    lemmas = {form: verdict["lemma"] for form, verdict in agents.items()}
    rest = pool.slice(CALLS).with_columns(agents=pl.col("form").replace_strict(lemmas))
    changed = pl.col("agents") != pl.col("spacy_lemma")
    adjective_merge = (
        (pl.col("agents") == pl.col("spacy_lemma"))
        & (pl.col("tag") == "NNS")
        & (pl.col("adjective_share") >= ADJECTIVE_SHARE)
    )
    follow_up = rest.filter(
        (pl.col("count") >= FOLLOW_UP_MIN_USES) & (changed | adjective_merge)
    ).drop("agents")
    return pl.concat([pool.head(CALLS), follow_up])


def call_items(chosen: pl.DataFrame, agents: Verdicts) -> list[dict[str, object]]:
    """The agents' calls part, with spaCy's lemma and the agents' beside each form."""
    return [
        {
            "id": row["form"],
            "form": row["form"],
            "count": row["count"],
            "group": row["group"],
            "contexts": row["contexts"],
            "spacy": row["spacy_lemma"],
            "agents": agents[row["form"]]["lemma"],
            "reason": agents[row["form"]].get("reason", ""),
        }
        for row in chosen.iter_rows(named=True)
    ]


def main(review_list_path: Path, agents_path: Path, out_path: Path) -> None:
    """Fill the template with the three parts' items and write the page."""
    review_list = pl.read_parquet(review_list_path)
    agents: Verdicts = json.loads(agents_path.read_text(encoding="utf-8"))
    chosen = call_rows(review_list, agents)
    blind, calls = blind_items(review_list, agents), call_items(chosen, agents)
    unusable = [item["id"] for item in blind + calls if not DOCUMENT_ID.fullmatch(str(item["id"]))]
    if unusable:
        raise ValueError(f"forms the page can't store as document ids: {unusable}")
    page = (
        TEMPLATE.read_text(encoding="utf-8")
        .replace("__ITEMS__", json.dumps(blind, ensure_ascii=False))
        .replace("__CALLS__", json.dumps(calls, ensure_ascii=False))
    )
    out_path.write_text(page, encoding="utf-8")
    pool = int(review_list.filter(pl.col("group").is_in(CALL_GROUPS))["count"].sum())
    covered = int(chosen["count"].sum())
    sys.stdout.write(
        f"{len(blind)} blind forms and {len(calls)} calls written to {out_path}; the calls hold "
        f"{covered:,} of the {pool:,} suspect and homograph tokens ({covered / pool:.0%})\n"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
