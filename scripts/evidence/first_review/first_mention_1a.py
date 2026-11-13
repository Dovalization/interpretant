"""Check 1a's items: how many come from abstracts with several mentions, and does position
(first sentence, first-of-several) relate to the author's 'unclear' answers?"""

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts")
from check_known_presence import CASES, SAMPLING, usages  # noqa: E402

from interpretant.drift.senses import (  # noqa: E402
    _URL,
    _SENTENCE_END,
    cased_abstracts,
    drop_shared,
    word_pattern,
)

OUT = Path("data/diagnostics")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as table:
        return list(csv.DictReader(table, delimiter="\t"))


blind = {row["item"]: row for row in read(OUT / "check_1a_blind.tsv")}
key = {row["item"]: row for row in read(OUT / "check_1a_key.tsv")}
labels = {row["item"]: row["label"].strip().lower() for row in read(OUT / "check_1a_labels.tsv")}

rows = []
for case in CASES:
    word, root, field, early, late, _ = CASES[case]
    cells = drop_shared({(field, d): cased_abstracts(root, field, d) for d in (early, late)})
    texts = {wid: text for docs in cells.values() for wid, text in docs}
    from interpretant.drift.senses import sample_occurrences

    found = sample_occurrences(cells, word, seed=1, sampling=SAMPLING)
    by_context = {}
    for occurrence in found:
        shown = (
            f"{occurrence.context[:occurrence.start]}⟦"
            f"{occurrence.context[occurrence.start:occurrence.end]}⟧"
            f"{occurrence.context[occurrence.end:]}"
        ).replace("\t", " ")
        by_context[(occurrence.decade, shown)] = occurrence
    pattern = word_pattern(word, SAMPLING.plural)
    for item, key_row in key.items():
        if key_row["case"] != case or item not in labels:
            continue
        occurrence = by_context.get((int(key_row["decade"]), blind[item]["context"]))
        if occurrence is None:
            rows.append((item, case, None, None, labels[item]))
            continue
        text = _URL.sub(" ", texts[occurrence.work_id])
        matches = list(pattern.finditer(text))
        first_end = next(iter(_SENTENCE_END.finditer(text)), None)
        in_opening = first_end is None or matches[0].start() < first_end.end()
        rows.append((item, case, len(matches), in_opening, labels[item]))

print(f"items: {len(rows)}, unmatched: {sum(r[2] is None for r in rows)}")
matched = [r for r in rows if r[2] is not None]
multi = [r for r in matched if r[2] > 1]
print(f"from abstracts with several mentions: {len(multi)} of {len(matched)} "
      f"({len(multi) / len(matched):.0%})")


def table(name: str, groups: dict[str, list]) -> None:
    print(f"\n{name}")
    for group, members in groups.items():
        counts = Counter(r[4] for r in members)
        total = len(members)
        print(f"  {group:<28} n={total:<4} " + "  ".join(
            f"{label}={counts[label]} ({counts[label] / total:.0%})"
            for label in ("yes", "no", "unclear")
        ))


table("by number of mentions", {
    "one mention": [r for r in matched if r[2] == 1],
    "several mentions (first)": multi,
})
table("by opening sentence", {
    "in opening sentence": [r for r in matched if r[3]],
    "later sentence": [r for r in matched if not r[3]],
})
table("several mentions, by opening sentence", {
    "in opening sentence": [r for r in multi if r[3]],
    "later sentence": [r for r in multi if not r[3]],
})
