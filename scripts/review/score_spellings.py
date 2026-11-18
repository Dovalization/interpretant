"""Score the spelling audit by `docs/spellings.md`'s rules and write `curated/spellings.tsv`.

Reads the key and the agent verdicts in `data/review/spellings/` (the two judges' agreed verdict,
else the adjudicator's) and the author's calibration answers as the page's store exports them
(one JSON file per blind id). The calibration passes if the agents agree with the author on at
least 27 of the 30 ("not sure" counts as a disagreement); then the sample passes if at most 3 of
its 150 pairs are wrong. The map is the proposal minus every pair judged wrong in the sample or
the census. Usage: `python scripts/review/score_spellings.py PROPOSAL.parquet AUTHOR_DIR`.
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
        for path in sorted((REVIEW / "verdicts").glob(f"judge_{judge}_batch_*.json")):
            found |= {i: v["verdict"] for i, v in json.loads(path.read_text()).items()}
        return found

    first, second = load("A"), load("B")
    settled = {
        i: v["verdict"]
        for i, v in json.loads((REVIEW / "verdicts" / "opus_adjudication.json").read_text()).items()
    }
    return {i: first[i] if first[i] == second[i] else settled[i] for i in first}


def author_answers(folder: Path) -> dict[str, str]:
    """The author's calibration answers, by blind id."""
    answers: dict[str, str] = {}
    for path in sorted(folder.rglob("*.json")):
        document = json.loads(path.read_text())
        answers[path.stem] = document.get("data", document)["verdict"]
    return answers


def main() -> None:
    """Apply the rules, report, and write the map if the audit passes."""
    proposal = pl.read_parquet(sys.argv[1])
    key = json.loads((REVIEW / "key.json").read_text())
    agents = agent_verdicts()
    author = author_answers(Path(sys.argv[2]))
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
    wrong_sample = [i for i in sample if agents[i] == "wrong"]
    sys.stdout.write(
        f"sample: {len(wrong_sample)} of {len(sample)} wrong (rule: at most {SAMPLE_MAX_WRONG}): "
        + ", ".join(f"{key[i]['form']}->{key[i]['standard']}" for i in wrong_sample)
        + "\n"
    )
    wrong = {key[i]["form"] for i in key if agents[i] == "wrong"}
    census_wrong = sorted(wrong - {key[i]["form"] for i in wrong_sample})
    sys.stdout.write(f"census, fixed one by one: {census_wrong}\n")
    if len(wrong_sample) > SAMPLE_MAX_WRONG:
        raise SystemExit("the sample does not pass: the failing source is reviewed in full")
    kept = proposal.filter(~pl.col("form").is_in(sorted(wrong))).sort("form")
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
    sys.stdout.write(f"{kept.height} pairs written to {OUT} ({len(wrong)} judged wrong left out)\n")


if __name__ == "__main__":
    main()
