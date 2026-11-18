"""Candidate merge rules for the sentence view: cuts that look wrong (`docs/sentence_view.md`).

The sentence view's cut precision failed its bar (2026-11-18: 7 wrong of 314). The wrong cuts fell
after *p. 8* inside brackets, inside spaced OCR ellipses, at a stray OCR period and after bare list
numbers. This script scans every cut between two consecutive sentences of a kept work and counts
the cuts each candidate pattern matches, overall and per decade, with seeded examples of each
(the last characters before the cut and the first after), so each pattern's precision can be
judged before it becomes a rule. The examples quote corpus text: they stay out of git. Run from the
repository root on the run machine:
`python scripts/evidence/sentence_cuts.py OUT [PROCESSES]`.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from rich.console import Console

from interpretant.catalog.abstracts import abstracts_parts, load_abstracts_part
from interpretant.catalog.config import CatalogConfig, read_catalog
from interpretant.cli.progress import LiveProgress
from interpretant.pipelines.workers import map_parts

SEED = 20261130
EXAMPLES = 40
CONTEXT = 70
ABBREVIATIONS = frozenset(
    [
        "p",
        "pp",
        "vol",
        "vols",
        "no",
        "nos",
        "fig",
        "figs",
        "eq",
        "eqs",
        "ed",
        "eds",
        "al",
        "eg",
        "ie",
        "cf",
        "vs",
        "ca",
        "approx",
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "st",
        "jr",
        "sr",
        "inc",
        "ltd",
        "co",
        "dept",
        "univ",
        "ch",
        "chap",
        "sec",
        "secs",
        "ref",
        "refs",
        "viz",
        "op",
        "cit",
        "ibid",
        "resp",
        "esp",
        "etc",
    ]
)
_LIST_NUMBER = re.compile(r"(?:^|[\s(:])(?:\d{1,2}|[ivxlc]{1,5}|[a-zA-Z])\s?\.$")
_INITIAL = re.compile(r"(?:^|\s)[A-Z]\.$")
_ELLIPSIS = re.compile(r"\.\s\.$|^\.\s\.|^\.\s*$")
_LAST_WORD = re.compile(r"([A-Za-z][A-Za-z.]*)\.$")
_TERMINAL = ".!?:;\"'”’)]»"


def patterns(before: str, after: str) -> list[str]:
    """The candidate patterns a cut between ``before`` and ``after`` matches."""
    found = []
    head = after.lstrip("\"'“‘([")
    if not any(character.isalpha() for character in after):
        found.append("after_has_no_letter")
    if not any(character.isalpha() for character in before):
        found.append("before_has_no_letter")
    if head[:1].islower():
        found.append("after_starts_lowercase")
    if after[:1] in ":;,)]}":
        found.append("after_starts_with_punctuation")
    if after[:1].isdigit():
        found.append("after_starts_with_digit")
    if _LIST_NUMBER.search(before):
        found.append("before_ends_with_list_number")
    if _INITIAL.search(before):
        found.append("before_ends_with_initial")
    word = _LAST_WORD.search(before)
    if word and word.group(1).replace(".", "").lower() in ABBREVIATIONS:
        found.append("before_ends_with_abbreviation")
    if _ELLIPSIS.search(before) or _ELLIPSIS.search(after):
        found.append("spaced_ellipsis")
    if before.count("(") > before.count(")") or before.count("[") > before.count("]"):
        found.append("before_has_open_bracket")
    if before.rstrip()[-1:] not in _TERMINAL:
        found.append("before_has_no_end_punctuation")
    if sum(1 for piece in before.split() if any(c.isalpha() for c in piece)) <= 2:
        found.append("before_is_short")
    return found


def scan_part(part: str, catalog: CatalogConfig) -> dict[str, Any]:
    """One part's cuts: counts per pattern and decade, and the smallest-keyed examples of each."""
    abstracts = load_abstracts_part(
        catalog, part, ["work_id", "decade", "text", "sentences", "dropped_reason"]
    )
    counts: Counter[str] = Counter()
    examples: dict[str, list[tuple[str, str, int, str, str]]] = {}
    for work_id, decade, text, sentences, dropped in abstracts.iter_rows():
        if dropped is not None:
            continue
        for number in range(1, len(sentences)):
            before = text[sentences[number - 1]["start"] : sentences[number - 1]["end"]]
            after = text[sentences[number]["start"] : sentences[number]["end"]]
            found = patterns(before, after)
            counts["cuts"] += 1
            counts[f"cuts:{decade}"] += 1
            counts["cuts_with_any"] += bool(found)
            for name in found:
                counts[name] += 1
                counts[f"{name}:{decade}"] += 1
                key = hashlib.sha256(f"{SEED}:{name}:{work_id}:{number}".encode()).hexdigest()
                kept = examples.setdefault(name, [])
                kept.append((key, work_id, number, before[-CONTEXT:], after[:CONTEXT]))
                if len(kept) > 4 * EXAMPLES:
                    examples[name] = sorted(kept)[:EXAMPLES]
    return {
        "counts": dict(counts),
        "examples": {name: sorted(rows)[:EXAMPLES] for name, rows in examples.items()},
    }


def main() -> None:
    """Scan every part and write the counts and the examples."""
    out = Path(sys.argv[1])
    processes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    out.mkdir(parents=True, exist_ok=True)
    catalog = read_catalog()
    parts = abstracts_parts(catalog)
    with LiveProgress(Console()) as live:
        live.stage("scanning cuts", total=len(parts))
        results = map_parts(
            scan_part, dict.fromkeys(parts, catalog), process_count=processes, progress=live
        )
    counts: Counter[str] = Counter()
    examples: dict[str, list[Any]] = {}
    for result in results.values():
        counts.update(result["counts"])
        for name, rows in result["examples"].items():
            examples[name] = sorted(examples.get(name, []) + rows)[:EXAMPLES]
    (out / "counts.json").write_text(json.dumps(dict(sorted(counts.items())), indent=1) + "\n")
    (out / "examples.json").write_text(json.dumps(examples, indent=1, ensure_ascii=False) + "\n")
    names = sorted({name.split(":")[0] for name in counts} - {"cuts", "cuts_with_any"})
    sys.stdout.write(f"{counts['cuts']:,} cuts, {counts['cuts_with_any']:,} match a pattern\n")
    for name in sorted(names, key=lambda n: -counts[n]):
        sys.stdout.write(f"  {name}: {counts[name]:,} ({counts[name] / counts['cuts']:.2%})\n")


if __name__ == "__main__":
    main()
