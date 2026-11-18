"""Score the spelling audit by `docs/spellings.md`'s rules and write `curated/spellings.tsv`.

Reads the key and the second round's agent verdicts in `data/review/spellings/verdicts_v2/` (the
two judges' agreed verdict, else the adjudicator's) and the author's answers to the second
calibration (a JSON object of blind id to verdict). The calibration passes if the agents agree with
the author on at least 27 of the 30 ("not sure" counts as a disagreement); then the sample passes
if at most 3 of its 150 pairs are wrong, the author's own calls counting as wrong. The map is the
proposal minus every pair judged wrong in the sample or the census, minus the author's calls,
and minus the singular or plural of each such pair.
Usage: `python scripts/review/score_spellings.py PROPOSAL.parquet AUTHOR.json`.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import polars as pl

from interpretant.catalog.config import read_catalog
from interpretant.catalog.words import load_words
from interpretant.corpus.spelling import mapped_away

REVIEW = Path("data/review/spellings")
VERDICTS = REVIEW / "verdicts_v2"
# The author's census call (2026-11-17): programme marks a sense program does not.
AUTHOR_CALLS = frozenset({"programme", "programmes"})
OUT = Path("curated/spellings.tsv")
CALIBRATION_AGREEMENT = 27
SAMPLE_MAX_WRONG = 3
HEADER = """\
# The spelling map (decision agenda 3.4; docs/spellings.md): form<TAB>standard<TAB>source.
# Each form is written as its standard in the static view and in target matching, per hyphen
# segment; the standard is the form the corpus uses more often, a word-table lemma keeping its own.
# Built by scripts/review/score_spellings.py from the audited proposal; do not edit by hand.
"""


def agent_verdicts() -> dict[str, str]:
    """Each blind id's final agent verdict: the judges' agreed one, else the adjudicator's."""

    def load(judge: str) -> dict[str, str]:
        found: dict[str, str] = {}
        for path in sorted((VERDICTS).glob(f"judge_{judge}_batch_*.json")):
            found |= {i: v["verdict"] for i, v in json.loads(path.read_text()).items()}
        return found

    first, second = load("A"), load("B")
    settled = {
        i: v["verdict"]
        for i, v in json.loads((VERDICTS / "opus_adjudication.json").read_text()).items()
    }
    return {i: first[i] if first[i] == second[i] else settled[i] for i in first}


def other_number(pair: frozenset[str]) -> list[frozenset[str]]:
    """The pair's singular or plural, so a pair judged wrong takes it along (*meters*, *metre*)."""
    first, second = sorted(pair)
    if first.endswith("s") and second.endswith("s"):
        return [frozenset({first[:-1], second[:-1]})]
    return [frozenset({f"{first}s", f"{second}s"})]


def main() -> None:
    """Apply the rules, report, and write the map if the audit passes."""
    proposal = pl.read_parquet(sys.argv[1])
    key = json.loads((REVIEW / "key.json").read_text())
    agents = agent_verdicts()
    author: dict[str, str] = json.loads(Path(sys.argv[2]).read_text())
    agreed = sum(author.get(i) == agents[i] for i in author)
    sys.stdout.write(
        f"calibration: the agents agree with the author on {agreed} of {len(author)} "
        f"(rule: at least {CALIBRATION_AGREEMENT} of 30)\n"
    )
    for i in sorted(author):
        if author[i] != agents[i]:
            sys.stdout.write(
                f"  {key[i]['form']} -> {key[i]['standard']}: "
                f"author {author[i]}, agents {agents[i]}\n"
            )
    if len(author) < 30 or agreed < CALIBRATION_AGREEMENT:
        raise SystemExit("the calibration does not pass: the author judges the sample")
    sample = [i for i, entry in key.items() if "sample" in entry["sets"]]
    wrong_sample = [i for i in sample if agents[i] == "wrong" or key[i]["form"] in AUTHOR_CALLS]
    sys.stdout.write(
        f"sample: {len(wrong_sample)} of {len(sample)} wrong (rule: at most {SAMPLE_MAX_WRONG}): "
        + ", ".join(f"{key[i]['form']}->{key[i]['standard']}" for i in wrong_sample)
        + "\n"
    )
    wrong = {key[i]["form"] for i in key if agents[i] == "wrong"} | AUTHOR_CALLS
    census_wrong = sorted(wrong - {key[i]["form"] for i in wrong_sample})
    sys.stdout.write(f"census, fixed one by one: {census_wrong}\n")
    if len(wrong_sample) > SAMPLE_MAX_WRONG:
        raise SystemExit("the sample does not pass: the failing source is reviewed in full")
    pairs = [frozenset(pair) for pair in proposal.select("form", "standard").iter_rows()]
    wrong_pairs = {
        pair for pair, form in zip(pairs, proposal["form"], strict=True) if form in wrong
    }
    family = wrong_pairs | {
        number_pair for pair in wrong_pairs for number_pair in other_number(pair)
    }
    dropped = [pair in family for pair in pairs]
    kept = proposal.filter(~pl.Series(dropped)).sort("form")
    extra = sorted(set(proposal.filter(pl.Series(dropped))["form"]) - wrong)
    sys.stdout.write(f"left out with their singular or plural judged wrong: {extra}\n")
    spellings = dict(zip(kept["form"], kept["standard"], strict=True))
    lemmas = [word.lemma for word in load_words(read_catalog()) if not word.is_phrase]
    away = mapped_away(spellings, lemmas)
    if away:
        raise SystemExit(f"word-table lemmas mapped away: {away}")
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        handle.write(HEADER)
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["form", "standard", "source"])
        writer.writerows(kept.select("form", "standard", "source").iter_rows())
    sys.stdout.write(
        f"{kept.height} pairs written to {OUT} ({proposal.height - kept.height} left out)\n"
    )


if __name__ == "__main__":
    main()
