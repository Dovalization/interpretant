"""Two name rules for capitalised kept targets, measured on the audits' judged items.

After GLiNER failed to separate names from terms written in capitals (`gliner_names.py`,
`docs/usages_audit.md`), two rules that use other evidence were set before they ran (the author,
2026-11-21). Both read the target's capitalised run: from the target, the words on either side that
are capitalised or are the links a name may hold (*and*, *of*, *for*, *the*, *in*, *on*, *to*,
*&*, *-*, as in `usages/entities.py`), stopping at any other word, number or sign, with links
trimmed from both ends. A run of the target alone never makes a name.

- **The lower-case check** (Mikheev 2002, *Periods, Capitalized Words, etc.*: a capitalised word
  is a common word if the corpus writes it in lower case elsewhere): the run is counted over every
  work's text, ignoring case, and a name is a run seen at least 3 times, under a tenth of them in
  lower case. A run seen fewer than 3 times is left as a use; the variant counting it as a name is
  reported beside.
- **Name words:** the run holds a publication's or an organisation's word (`NAME_WORDS`), as the
  instrument rule's run holds a test's.

Items: every kept item of the audit folders given whose target is capitalised and does not open its
sentence, with the judges' final verdict as the answer. A rule (or the two together) is worth
adopting if it removes at least 5 wrong items and at least twice as many wrong items as right ones;
an adopted rule is then confirmed by a fresh precision draw. The report holds counts only. Run from
the repository root on the machine holding the audits' items and the corpus:
`python scripts/evidence/capitalised_runs.py OUT.json AUDIT_DIR...`.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from interpretant.catalog.abstracts import load_abstracts_part
from interpretant.catalog.config import read_catalog

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "review"))
from score_usages import final_verdicts, load_items

OPEN, CLOSE = "⟨t⟩", "⟨/t⟩"
OPENING = " \"'‘“([-…"  # what may precede a sentence's first word (`usages/traits.py`)
TOKEN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*|\d+|[^\w\s]")
LINKS = frozenset({"and", "of", "for", "the", "in", "on", "to", "&", "-"})
NAME_WORDS = frozenset(
    {
        "Journal", "Proceedings", "Conference", "Symposium", "Workshop", "Congress", "Society",
        "Association", "Institute", "Institution", "Laboratory", "Laboratories", "Center",
        "Centre", "University", "College", "Department", "Foundation", "Council", "Committee",
        "Corporation", "Company", "Inc", "Ltd", "Press", "Quarterly", "Bulletin", "Transactions",
        "Magazine", "Handbook", "Encyclopedia",
    }
)  # fmt: skip
MIN_SEEN = 3
MAX_LOWER_SHARE = 0.1
MIN_REMOVED = 5
MIN_RATIO = 2


def capitalised_run(sentence: str) -> tuple[str, list[str]] | None:
    """The target's text and its capitalised run, if the target is capitalised mid-sentence."""
    before, rest = sentence.split(OPEN, 1)
    target, after = rest.split(CLOSE, 1)
    if not target[:1].isupper() or not before.strip(OPENING):
        return None
    left, right = TOKEN.findall(before), TOKEN.findall(after)

    def extend(tokens: list[str]) -> list[str]:
        run = []
        for token in tokens:
            if not (token[:1].isupper() or token.lower() in LINKS):
                break
            run.append(token)
        while run and run[-1].lower() in LINKS:
            run.pop()
        return run

    run = [*reversed(extend(list(reversed(left)))), *TOKEN.findall(target), *extend(right)]
    return target, run


def run_text(run: list[str]) -> str:
    """The run as one lower-case string, words joined by spaces and hyphens kept tight."""
    return " ".join(run).replace(" - ", "-").replace(" -", "-").replace("- ", "-").lower()


def corpus_counts(runs: set[str]) -> dict[str, tuple[int, int]]:
    """Each run's occurrences over every work's text, ignoring case, and those in lower case."""
    catalog = read_catalog()
    pattern = re.compile(
        r"(?<!\w)(?:"
        + "|".join(re.escape(run).replace(r"\ ", r"\s+") for run in sorted(runs, key=len)[::-1])
        + r")(?!\w)"
    )
    seen: Counter[str] = Counter()
    lower: Counter[str] = Counter()
    parts = sorted(path.stem for path in catalog.usages.path.glob("*.parquet"))
    for index, part in enumerate(parts, 1):
        for text in load_abstracts_part(catalog, part, ["text"])["text"].drop_nulls():
            folded = text.lower()
            for match in pattern.finditer(folded):
                found = re.sub(r"\s+", " ", match.group())
                seen[found] += 1
                lower[found] += text[match.start() : match.end()] == match.group()
        sys.stderr.write(f"counting runs: {index}/{len(parts)} parts\n")
    return {run: (seen[run], lower[run]) for run in runs}


def score(rows: list[dict[str, Any]], rule: str) -> dict[str, Any]:
    """Wrong items removed against right items lost, and whether the rule meets the bar."""
    removed = sum(row[rule] and row["wrong"] for row in rows)
    lost = sum(row[rule] and not row["wrong"] for row in rows)
    return {
        "wrong_removed": removed,
        "wrong_total": sum(row["wrong"] for row in rows),
        "right_lost": lost,
        "right_total": sum(not row["wrong"] for row in rows),
        "meets_bar": removed >= MIN_REMOVED and removed >= MIN_RATIO * lost,
    }


def main() -> None:
    """Apply both rules to the capitalised kept items and write the report."""
    out, audits = Path(sys.argv[1]), [Path(arg) for arg in sys.argv[2:]]
    rows: list[dict[str, Any]] = []
    for audit in audits:
        key = json.loads((audit / "key.json").read_text())
        final, _, _ = final_verdicts(audit, key)
        items = load_items(audit)
        for item_id, entry in key.items():
            if entry["kind"] != "kept":
                continue
            found = capitalised_run(items[item_id]["sentence"])
            if found is None:
                continue
            target, run = found
            words = [t for t in run if t.lower() not in LINKS]
            longer = len(words) > len([t for t in TOKEN.findall(target) if t.lower() not in LINKS])
            rows.append(
                {
                    "audit": audit.name,
                    "wrong": final[item_id] == "wrong",
                    "run": run_text(run) if longer else None,
                    "name_word": longer and any(t in NAME_WORDS for t in run),
                }
            )
    counts = corpus_counts({row["run"] for row in rows if row["run"]})
    for row in rows:
        seen, lower = counts.get(row["run"], (0, 0)) if row["run"] else (0, 0)
        rarely_lower = lower < MAX_LOWER_SHARE * seen
        row["lower_case"] = bool(row["run"]) and seen >= MIN_SEEN and rarely_lower
        row["lower_case_or_unseen"] = bool(row["run"]) and (seen < MIN_SEEN or rarely_lower)
        row["either"] = row["lower_case"] or row["name_word"]
    rules = ["lower_case", "lower_case_or_unseen", "name_word", "either"]
    report = {
        "items": dict(Counter(row["audit"] for row in rows)),
        "runs_longer_than_the_target": {
            "wrong": sum(bool(row["run"]) and row["wrong"] for row in rows),
            "right": sum(bool(row["run"]) and not row["wrong"] for row in rows),
        },
        "rules": {rule: score(rows, rule) for rule in rules},
        "by_audit": {
            name: {rule: score(group, rule) for rule in rules}
            for name in sorted({row["audit"] for row in rows})
            if (group := [row for row in rows if row["audit"] == name])
        },
    }
    out.write_text(json.dumps(report, indent=1) + "\n")
    sys.stdout.write(f"{len(rows)} capitalised kept items; written to {out}\n")


if __name__ == "__main__":
    main()
