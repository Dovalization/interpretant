"""Build `curated/lemma_exceptions.tsv`, the static view's lemma exceptions (agenda 3.2, 3.3).

From the lemma review (`docs/lemma_review.md`), each form takes, in this order of precedence:

1. the override file's decision (`data/lemma_overrides.tsv`; a bare form keeps itself), except
   *goes*, which merges as an ordinary verb (3.2);
2. the author's final answer, for the author's 142 forms;
3. the author's answer on the calls, for the 50 suspects and homographs the author saw;
4. the agents' verdict, for the other suspects and homographs (the calibration counts as passed)
   and for the untested `disagree` tail;
5. the rejoined last segment, for every hyphenated form (the hyphenated rule passed).

A form is written only when its lemma differs from spaCy's, except the override file's forms,
which are all written as the project's policy list. The lemma is spelled out even when it is the
form itself. The build fails if a target word would be mapped to another word.

Usage, from the repository root: `python scripts/evidence/lemma_exceptions.py
REVIEW_LIST_TAIL.parquet` (the list `lemma_review_list.py --tail` writes). Reads
`curated/lemma_review_judgements.tsv`, writes `curated/lemma_exceptions.tsv`.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import polars as pl

from interpretant.corpus.audit import load_target_words
from interpretant.corpus.lemmas import read_overrides

OVERRIDES = Path("data/lemma_overrides.tsv")
JUDGEMENTS = Path("curated/lemma_review_judgements.tsv")
TARGETS = Path("data/words_of_interest.txt")
OUT = Path("curated/lemma_exceptions.tsv")
NOT_CARRIED = frozenset({"goes"})
AUTHOR_GROUPS = frozenset({"author", "tail_sample", "policy_keep"})
AGENT_GROUPS = frozenset({"suspect_merge", "homograph", "tail"})
HEADER = """\
# The static view's lemma exceptions (docs/decision_agenda.md 3.2, 3.3; docs/lemma_review.md).
# form<TAB>lemma<TAB>source; the lemma is spelled out even when it is the form itself.
# Sources: override (data/lemma_overrides.tsv), author (the author's final answer), author_call
# (the author's answer on a call), agents (the agent judges), hyphen_rule (the rejoined last
# segment). Built by scripts/evidence/lemma_exceptions.py; do not edit by hand.
"""


def decided(judgement: dict[str, str]) -> tuple[str, str] | None:
    """A judged form's lemma and its source, or None when it is decided by the hyphenated rule."""
    group = judgement["group"]
    if group in AUTHOR_GROUPS:
        return judgement["author_final"], "author"
    if judgement["author_call"]:
        return judgement["author_call"], "author_call"
    if group in AGENT_GROUPS:
        return judgement["agents"], "agents"
    return None


def exceptions(review_list: pl.DataFrame) -> dict[str, tuple[str, str]]:
    """Each form's exception lemma and source, by the precedence in the module docstring."""
    result: dict[str, tuple[str, str]] = {
        form: (lemma or form, "override")
        for form, lemma in read_overrides(OVERRIDES).items()
        if form not in NOT_CARRIED
    }
    spacy_lemma = dict(zip(review_list["form"], review_list["spacy_lemma"], strict=True))
    with JUDGEMENTS.open(encoding="utf-8") as handle:
        for judgement in csv.DictReader(handle, delimiter="\t"):
            choice = decided(judgement)
            form = judgement["form"]
            if choice is None or form in result:
                continue
            if not choice[0]:
                raise ValueError(f"no lemma for {form} ({judgement['group']})")
            if choice[0] != spacy_lemma[form]:
                result[form] = choice
    hyphenated = review_list.filter(pl.col("kind") == "hyphenated")
    for form, rejoined in hyphenated.select("form", "voted_lemma").iter_rows():
        if form not in result:
            result[form] = (rejoined, "hyphen_rule")
    return result


def main() -> None:
    """Build the exceptions, check the targets and write the file."""
    review_list = pl.read_parquet(sys.argv[1])
    result = exceptions(review_list)
    targets = {word for word in load_target_words(TARGETS) if "_" not in word}
    mapped_away = sorted(
        f"{form} -> {result[form][0]}"
        for form in targets & result.keys()
        if result[form][0] != form
    )
    if mapped_away:
        raise SystemExit(f"target words mapped to another word: {mapped_away}")
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        handle.write(HEADER)
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["form", "lemma", "source"])
        for form in sorted(result):
            writer.writerow([form, *result[form]])
    sources = Counter(source for _, source in result.values())
    kept = sum(lemma == form for form, (lemma, _) in result.items())
    touched = sorted(targets & result.keys())
    merged_in = sorted(
        f"{form} -> {lemma}"
        for form, (lemma, _) in result.items()
        if lemma in targets and lemma != form
    )
    sys.stdout.write(
        f"{len(result)} exceptions written to {OUT}: {dict(sources.most_common())}; {kept} keep "
        f"their form; target words in the file (all kept as themselves): {touched}; forms merged "
        f"into a target word: {merged_in}\n"
    )


if __name__ == "__main__":
    main()
