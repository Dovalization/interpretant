"""Build `curated/lemma_exceptions.tsv`, the static view's lemma exceptions (agenda 3.2, 3.3).

From the lemma review (`docs/lemma_review.md`), each form takes, in this order of precedence:

1. the override file's decision (`data/lemma_overrides.tsv`; a bare form keeps itself), except
   *goes*, which merges as an ordinary verb (3.2);
2. the adjective rule: a plural noun whose singular is mostly an adjective keeps its form (at
   least 20 uses of the singular, at least half of them tagged as adjectives, and an adjective
   reading in spaCy's WordNet lemma index; hyphenated forms aside);
3. the author's final answer, for the author's 142 forms;
4. the author's answer on the calls, for the 50 suspects and homographs the author saw;
5. the agents' verdict, for the other suspects and homographs (the calibration counts as passed)
   and for the untested `disagree` tail;
6. the rejoined last segment, for every hyphenated form (the hyphenated rule passed), except
   where, under the spelling map (`curated/spellings.tsv`, 3.4), it only respells the form
   (*edge-colourings* as *edge-colorings*, undoing spaCy's singular) or differs from spaCy's lemma
   only in spelling (*anti-sceptics*): spelling is the map's job (2026-11-17).

A form is written only when its lemma differs from spaCy's, except the override file's forms,
which are all written as the project's policy list. The lemma is spelled out even when it is the
form itself. Each entry names the reading it applies to (`NNS` or `VBZ`): the override file's
forms and the rule's as plural nouns (every override form is read mostly as one; *goes*, the only
verb, is not carried), and each judged form as the judges saw it. The build fails if a word of the
word table (`curated/words.tsv`) would be mapped to another word, and it reports every earlier
answer the rule changes.

Usage, from the repository root: `python scripts/evidence/lemma_exceptions.py REVIEW_LIST.parquet
PLURAL_MERGES.parquet`: the judged review list with the tail (`lemma_review_list.py --tail`) and
the plural merges of a deterministic run of the same script. Reads
`curated/lemma_review_judgements.tsv`, writes `curated/lemma_exceptions.tsv`.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import polars as pl

from interpretant.catalog.config import read_catalog
from interpretant.catalog.spellings import load_spellings
from interpretant.catalog.words import load_words
from interpretant.corpus.lemmas import american, read_overrides
from interpretant.corpus.spelling import spell

OVERRIDES = Path("data/lemma_overrides.tsv")
JUDGEMENTS = Path("curated/lemma_review_judgements.tsv")
OUT = Path("curated/lemma_exceptions.tsv")
NOT_CARRIED = frozenset({"goes"})
AUTHOR_GROUPS = frozenset({"author", "tail_sample", "policy_keep"})
AGENT_GROUPS = frozenset({"suspect_merge", "homograph", "tail"})
MIN_SINGULAR_USES = 20
ADJECTIVE_SHARE = 0.5
HEADER = """\
# The static view's lemma exceptions (docs/decision_agenda.md 3.2, 3.3; docs/lemma_review.md).
# form<TAB>lemma<TAB>tag<TAB>source; the lemma is spelled out even when it is the form itself;
# the tag is the reading the entry applies to (NNS or VBZ).
# Sources: override (data/lemma_overrides.tsv), adjective_rule (a plural whose singular is mostly
# an adjective), author (the author's final answer), author_call (the author's answer on a call),
# agents (the agent judges), hyphen_rule (the rejoined last segment).
# Built by scripts/evidence/lemma_exceptions.py; do not edit by hand.
"""


class Entry(NamedTuple):
    """One form's exception: its lemma, the reading it applies to and where it comes from."""

    lemma: str
    tag: str
    source: str


def adjective_words() -> frozenset[str]:
    """The words with an adjective reading in spaCy's WordNet lemma index."""
    import spacy

    pipeline = spacy.load("en_core_web_sm", exclude=["tok2vec", "tagger", "parser", "ner"])
    lemmatizer = pipeline.get_pipe("lemmatizer")
    return frozenset(lemmatizer.lookups.get_table("lemma_index")["adj"])  # type: ignore[attr-defined]


def rule_plurals(merges: pl.DataFrame) -> dict[str, str]:
    """The plurals the adjective rule keeps, each with spaCy's lemma for it."""
    chosen = merges.filter(
        (pl.col("singular_uses") >= MIN_SINGULAR_USES)
        & (pl.col("adjective_share") >= ADJECTIVE_SHARE)
        & ~pl.col("form").str.contains("-", literal=True)
        & pl.col("spacy_lemma").is_in(sorted(adjective_words()))
    )
    return dict(zip(chosen["form"], chosen["spacy_lemma"], strict=True))


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


def read_judgements() -> dict[str, dict[str, str]]:
    """The review's judgements, by form."""
    with JUDGEMENTS.open(encoding="utf-8") as handle:
        return {row["form"]: row for row in csv.DictReader(handle, delimiter="\t")}


def respelled(word: str, spellings: dict[str, str]) -> str:
    """The word with each hyphen segment respelled by the project's rules, then by the map."""
    return "-".join(spell(american(segment) or segment, spellings) for segment in word.split("-"))


def respells_only(row: dict[str, str], spellings: dict[str, str]) -> bool:
    """Whether a hyphenated form's rejoined lemma only respells the form, or spaCy's lemma."""
    if row["voted_lemma"] == row["spacy_lemma"]:
        return False
    lemma = respelled(row["voted_lemma"], spellings)
    return lemma in {respelled(row["form"], spellings), respelled(row["spacy_lemma"], spellings)}


def exceptions(
    review_list: pl.DataFrame,
    rule: dict[str, str],
    judgements: dict[str, dict[str, str]],
    spellings: dict[str, str],
) -> dict[str, Entry]:
    """Each form's exception, by the precedence in the module docstring."""
    result = {
        form: Entry(lemma or form, "NNS", "override")
        for form, lemma in read_overrides(OVERRIDES).items()
        if form not in NOT_CARRIED
    }
    for form in rule:
        result.setdefault(form, Entry(form, "NNS", "adjective_rule"))
    listed = {row["form"]: row for row in review_list.iter_rows(named=True)}
    for form, judgement in judgements.items():
        choice = decided(judgement)
        if choice is None or form in result:
            continue
        if not choice[0]:
            raise ValueError(f"no lemma for {form} ({judgement['group']})")
        if choice[0] != listed[form]["spacy_lemma"]:
            result[form] = Entry(choice[0], listed[form]["tag"], choice[1])
    for row in listed.values():
        if row["kind"] == "hyphenated" and not respells_only(row, spellings):
            result.setdefault(row["form"], Entry(row["voted_lemma"], row["tag"], "hyphen_rule"))
    return result


def rule_changes(
    rule: dict[str, str], merges: pl.DataFrame, judgements: dict[str, dict[str, str]]
) -> list[tuple[str, int, str, str]]:
    """The earlier answers the rule overturns: form, uses, the earlier lemma and its source."""
    counts = dict(zip(merges["form"], merges["count"], strict=True))
    overrides = read_overrides(OVERRIDES)
    changes = []
    for form, spacy_lemma in rule.items():
        if form in overrides and form not in NOT_CARRIED:
            continue
        choice = decided(judgements[form]) if form in judgements else None
        lemma, source = choice if choice else (spacy_lemma, "spacy")
        if lemma != form:
            changes.append((form, counts[form], lemma, source))
    return sorted(changes, key=lambda change: -change[1])


def report(rule: dict[str, str], changes: list[tuple[str, int, str, str]]) -> str:
    """The rule's reach and the answers it overturns, for the review's results."""
    by_source: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    for _, uses, _, source in changes:
        by_source[source] += 1
        tokens[source] += uses
    lines = [
        f"the rule keeps {len(rule)} plurals; it overturns {len(changes)} earlier answers:",
        *(
            f"  {source}: {by_source[source]} forms, {tokens[source]:,} tokens"
            for source in by_source
        ),
        "  by the author: "
        + ", ".join(
            f"{f} ({n}, was {lemma})" for f, n, lemma, s in changes if s.startswith("author")
        ),
        "  largest of the rest: "
        + ", ".join([f"{f} ({n})" for f, n, _, s in changes if not s.startswith("author")][:25]),
    ]
    overrides = read_overrides(OVERRIDES)
    disagree = sorted(f for f in rule if overrides.get(f, f) not in ("", f))
    lines.append(f"  override entries the rule disagrees with (the override file wins): {disagree}")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Build the exceptions, check the targets, report the rule and write the file."""
    review_list = pl.read_parquet(sys.argv[1])
    merges = pl.read_parquet(sys.argv[2])
    judgements = read_judgements()
    rule = rule_plurals(merges)
    catalog = read_catalog()
    spellings = {spelling.form: spelling.standard for spelling in load_spellings(catalog)}
    result = exceptions(review_list, rule, judgements, spellings)
    left_to_the_map = sorted(
        row["form"]
        for row in review_list.iter_rows(named=True)
        if row["kind"] == "hyphenated"
        and row["form"] not in result
        and respells_only(row, spellings)
    )
    sys.stdout.write(f"hyphenated forms left to the spelling map: {left_to_the_map}\n")
    targets = {word.lemma for word in load_words(catalog) if not word.is_phrase}
    mapped_away = sorted(
        f"{form} -> {result[form].lemma}"
        for form in targets & result.keys()
        if result[form].lemma != form
    )
    if mapped_away:
        raise SystemExit(f"target words mapped to another word: {mapped_away}")
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        handle.write(HEADER)
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["form", "lemma", "tag", "source"])
        for form in sorted(result):
            writer.writerow([form, result[form].lemma, result[form].tag, result[form].source])
    sources = Counter(entry.source for entry in result.values())
    kept = sum(entry.lemma == form for form, entry in result.items())
    merged_in = sorted(
        f"{form} -> {entry.lemma}"
        for form, entry in result.items()
        if entry.lemma in targets and entry.lemma != form
    )
    sys.stdout.write(
        f"{len(result)} exceptions written to {OUT}: {dict(sources.most_common())}; {kept} keep "
        f"their form; target words mapped away: none; merged into a target word: {merged_in}\n"
    )
    sys.stdout.write(report(rule, rule_changes(rule, merges, judgements)))


if __name__ == "__main__":
    main()
