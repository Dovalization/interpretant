"""Score the compound audit by `docs/compounds.md`'s rules and write `curated/compounds.tsv`.

Reads the key and the agent verdicts in `data/review/compounds/verdicts/` (the two judges' agreed
verdict, else the adjudicator's) and the author's calibration answers (a JSON object of blind id
to verdict). The calibration passes if the agents agree with the author on at least 27 of the 30
("not sure" counts as a disagreement); then the sample passes if at most 3 of its 150 pairs are
wrong. The map is the proposal minus every pair judged wrong in the sample or the census, minus
the pairs of the same kind whose form only inflects a wrong one (*re-create*, *re-created*,
*re-creating*), and minus any pair that would write a word of the table away. Usage:
`python scripts/review/score_compounds.py PROPOSAL.parquet AUTHOR.json`.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import polars as pl

from interpretant.catalog.compounds import load_compounds
from interpretant.catalog.config import read_catalog
from interpretant.catalog.words import load_words
from interpretant.corpus.compound_spelling import build_compound_map, compounds_written_away

REVIEW = Path("data/review/compounds")
VERDICTS = REVIEW / "verdicts"
OUT = Path("curated/compounds.tsv")
CALIBRATION_AGREEMENT = 27
SAMPLE_MAX_WRONG = 3
INFLECTIONS = ("s", "es", "d", "ed", "ing")
HEADER = """\
# The compound map (decision agenda 3.4; phase step 3; docs/compounds.md).
# Columns form<TAB>standard<TAB>source. Each spelling of a two-part compound is written as its
# family's most frequent form over kept works (source frequency), or as a word-table lemma (source
# word_table). A form with a space is a spaced pair, merged only as a noun-noun compound; any other
# form is one token, rewritten whole.
# Built by scripts/review/score_compounds.py from the audited proposal; do not edit by hand.
"""


def agent_verdicts() -> dict[str, str]:
    """Each blind id's final agent verdict: the judges' agreed one, else the adjudicator's."""

    def load(judge: str) -> dict[str, str]:
        found: dict[str, str] = {}
        for path in sorted(VERDICTS.glob(f"judge_{judge}_batch_*.json")):
            found |= {i: v["verdict"] for i, v in json.loads(path.read_text()).items()}
        return found

    first, second = load("A"), load("B")
    settled = {
        i: v["verdict"]
        for i, v in json.loads((VERDICTS / "opus_adjudication.json").read_text()).items()
    }
    return {i: first[i] if first[i] == second[i] else settled[i] for i in first}


def inflects(form: str, other: str) -> bool:
    """Whether ``other`` only inflects ``form`` (*re-create*, *re-creating*; *data bases*)."""
    stem = form[:-1] if form.endswith("e") else form
    return other != form and other.startswith(stem) and other[len(stem) :] in INFLECTIONS


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
    wrong_sample = [i for i in sample if agents[i] == "wrong"]
    sys.stdout.write(
        f"sample: {len(wrong_sample)} of {len(sample)} wrong (rule: at most {SAMPLE_MAX_WRONG}): "
        + ", ".join(f"{key[i]['form']}->{key[i]['standard']}" for i in wrong_sample)
        + "\n"
    )
    wrong = {key[i]["form"] for i in key if agents[i] == "wrong"}
    census_wrong = sorted(wrong - {key[i]["form"] for i in wrong_sample})
    sys.stdout.write(f"census, fixed one by one ({len(census_wrong)}): {census_wrong}\n")
    if len(wrong_sample) > SAMPLE_MAX_WRONG:
        raise SystemExit("the sample does not pass: the failing kind is reviewed in full")
    kinds = dict(zip(proposal["form"], proposal["kind"], strict=True))
    along = sorted(
        other
        for other in kinds
        if other not in wrong
        and any(inflects(form, other) and kinds[form] == kinds[other] for form in wrong)
    )
    sys.stdout.write(f"left out with an inflection judged wrong: {along}\n")
    kept = proposal.filter(~pl.col("form").is_in(sorted(wrong | set(along)))).sort("form")
    compounds = build_compound_map(kept.select("form", "standard").iter_rows())
    catalog = read_catalog()
    away = set(compounds_written_away(compounds, [word.lemma for word in load_words(catalog)]))
    if away:
        sys.stdout.write(f"pairs dropped for writing {len(away)} word-table words away\n")
        kept = kept.filter(
            ~pl.col("form").is_in(sorted(away))
            & ~pl.col("form").map_elements(
                lambda form: any(word in away for word in form.split(" ")),
                return_dtype=pl.Boolean,
            )
        )
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        handle.write(HEADER)
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["form", "standard", "source"])
        writer.writerows(kept.select("form", "standard", "source").iter_rows())
    written = load_compounds(catalog)
    sys.stdout.write(
        f"{len(written)} pairs written to {OUT} ({proposal.height - kept.height} left out); "
        f"by kind {dict(kept.group_by('kind').len().iter_rows())}\n"
    )


if __name__ == "__main__":
    main()
