"""Whether a second name recognizer tells names from uses among capitalised kept targets.

The usages audit's redraw (`docs/usages_audit.md`) found most kept errors on targets capitalised
mid-sentence: names the transformer's recognizer missed (cited titles, organisations, products,
named tests), mixed with real uses (headings, terms written in capitals). This test, set before it
was run (the author, 2026-11-21), asks whether GLiNER (Zaratiana et al. 2024), a recognizer that
takes its labels as plain words, separates the two well enough to justify a rule.

- **Items:** every kept item of the audit folders given whose target is capitalised and does not
  open its sentence (as shown to the judges), with the judges' final verdict as the answer: wrong
  is a name or another error, right is a real use.
- **Model:** `urchade/gliner_large-v2.1` through `gliner==0.2.30`, threshold 0.5 (the default),
  on the item's sentence as the judges saw it, without the marks.
- **Labels,** the codebook's names (`curated/judging/usages/prompt.md`): person, organisation,
  location, product, event, law, named test or scale, title of a cited work, named system or
  model.
- **The rule tested:** a target is inside a name when a found entity covers it and holds another
  word that is not a stop word (the generic-term rule's reading of a name, `usages/entities.py`).
  The looser rule, any covering entity, is reported beside it.
- **Worth a rule only if** it catches at least half of the wrong items and removes at least twice
  as many wrong items as right ones. A rule adopted on this would still be confirmed by a fresh
  precision draw.

The report holds counts only: no sentence or entity text. Run from the repository root on the
machine holding the audits' items:
`uv run --with gliner==0.2.30 python scripts/evidence/gliner_names.py OUT.json AUDIT_DIR...`.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from gliner import GLiNER
from spacy.lang.en.stop_words import STOP_WORDS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "review"))
from score_usages import final_verdicts, load_items

MODEL = "urchade/gliner_large-v2.1"
THRESHOLD = 0.5
LABELS = [
    "person",
    "organisation",
    "location",
    "product",
    "event",
    "law",
    "named test or scale",
    "title of a cited work",
    "named system or model",
]
OPEN, CLOSE = "⟨t⟩", "⟨/t⟩"
OPENING = " \"'‘“([-…"  # what may precede a sentence's first word (`usages/traits.py`)
WORD = re.compile(r"[^\W\d_]+")
MIN_SHARE_CAUGHT = 0.5
MIN_RATIO = 2


def capitalised_target(sentence: str) -> tuple[str, int, int] | None:
    """The unmarked sentence and the target's span, if the target is capitalised mid-sentence."""
    before, rest = sentence.split(OPEN, 1)
    target, after = rest.split(CLOSE, 1)
    if not target[:1].isupper() or not before.strip(OPENING):
        return None
    return before + target + after, len(before), len(before) + len(target)


def covering(entities: list[dict[str, Any]], text: str, start: int, end: int) -> dict[str, Any]:
    """The entities covering the target: any, and those holding another content word."""
    found = [e for e in entities if e["start"] <= start and e["end"] >= end]
    strict = [
        e
        for e in found
        if any(
            word.lower() not in STOP_WORDS
            for word in WORD.findall(text[e["start"] : start] + " " + text[end : e["end"]])
        )
    ]
    return {"any": found, "strict": strict}


def score(rows: list[dict[str, Any]], rule: str) -> dict[str, Any]:
    """Wrong items caught against right items lost, and whether the rule meets the bar."""
    wrong = [row for row in rows if row["wrong"]]
    right = [row for row in rows if not row["wrong"]]
    caught = sum(bool(row[rule]) for row in wrong)
    lost = sum(bool(row[rule]) for row in right)
    labels = Counter(row[rule][0]["label"] for row in rows if row[rule])
    return {
        "wrong_caught": caught,
        "wrong_total": len(wrong),
        "right_lost": lost,
        "right_total": len(right),
        "by_label": dict(labels.most_common()),
        "meets_bar": caught >= MIN_SHARE_CAUGHT * len(wrong) and caught >= MIN_RATIO * lost,
    }


def main() -> None:
    """Run the recognizer over the capitalised kept items and write the report."""
    out, audits = Path(sys.argv[1]), [Path(arg) for arg in sys.argv[2:]]
    model = GLiNER.from_pretrained(MODEL)
    model.to("cuda")
    rows: list[dict[str, Any]] = []
    for audit in audits:
        key = json.loads((audit / "key.json").read_text())
        final, _, _ = final_verdicts(audit, key)
        items = load_items(audit)
        for item_id, entry in key.items():
            if entry["kind"] != "kept":
                continue
            found = capitalised_target(items[item_id]["sentence"])
            if found is None:
                continue
            text, start, end = found
            entities = model.predict_entities(text, LABELS, threshold=THRESHOLD)
            rows.append(
                {"audit": audit.name, "wrong": final[item_id] == "wrong"}
                | covering(entities, text, start, end)
            )
    report = {
        "model": MODEL,
        "threshold": THRESHOLD,
        "labels": LABELS,
        "items": dict(Counter(row["audit"] for row in rows)),
        "strict": score(rows, "strict"),
        "any": score(rows, "any"),
        "by_audit": {
            name: {"strict": score(group, "strict"), "any": score(group, "any")}
            for name in sorted({row["audit"] for row in rows})
            if (group := [row for row in rows if row["audit"] == name])
        },
    }
    out.write_text(json.dumps(report, indent=1) + "\n")
    sys.stdout.write(f"{len(rows)} capitalised kept items; written to {out}\n")


if __name__ == "__main__":
    main()
