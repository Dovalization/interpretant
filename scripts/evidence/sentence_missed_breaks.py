"""Candidate split rules for the sentence view: breaks the spaCy pass missed.

The fresh audit's boundary sample (2026-11-18) found target sentences holding two sentences, most
where spaCy did not cut after a capital letter or a short acronym with a period, as if it were an
initial (*over a VP. Also, we propose*, *in Volume V. This work*). This script finds, inside every
sentence of the view with no exclusion, each place where such a token is followed by a common
sentence opener, and counts them per decade, with seeded examples, so the split's precision can be
judged before it becomes a rule (`docs/sentence_view.md`). The examples quote corpus text: they
stay out of git. Run from the repository root on the run machine:
`python scripts/evidence/sentence_missed_breaks.py OUT [PROCESSES]`.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl
from rich.console import Console

from interpretant.catalog.abstracts import abstracts_parts, load_abstracts_part
from interpretant.catalog.config import CatalogConfig, read_catalog
from interpretant.catalog.parts import part_path
from interpretant.cli.progress import LiveProgress
from interpretant.pipelines.workers import map_parts

SEED = 20261206
EXAMPLES = 60
CONTEXT = 60
OPENERS = (
    "The|This|These|That|Those|A|An|In|We|It|Its|Our|Also|However|Thus|Such|There|They|He|She"
    "|Finally|Moreover|Here|Furthermore|Although|When|Then|As|For|One|Two|Both|Each|No"
)
_SPLIT = re.compile(rf"(?<![A-Za-z.])([A-Z]{{1,3}})\.\s+(?=(?:{OPENERS})\b)")


def scan_part(part: str, catalog: CatalogConfig) -> dict[str, Any]:
    """One part's candidate splits: counts per decade and token, and smallest-keyed examples."""
    abstracts = load_abstracts_part(
        catalog, part, ["work_id", "decade", "text", "dropped_reason"]
    ).filter(pl.col("dropped_reason").is_null())
    texts = dict(zip(abstracts["work_id"], abstracts["text"], strict=True))
    decades = dict(zip(abstracts["work_id"], abstracts["decade"], strict=True))
    view = pl.read_parquet(
        part_path(catalog.sentence_view.path, part, ".parquet"),
        columns=["work_id", "sentence_number", "start", "end", "exclusion"],
    ).filter(pl.col("exclusion").is_null())
    counts: Counter[str] = Counter()
    examples: list[tuple[str, str, int, str]] = []
    for work_id, number, start, end, _ in view.iter_rows():
        sentence = texts[work_id][start:end]
        counts["sentences"] += 1
        for match in _SPLIT.finditer(sentence):
            decade = decades[work_id]
            counts["splits"] += 1
            counts[f"splits:{decade}"] += 1
            counts[f"token_length:{len(match.group(1))}"] += 1
            key = hashlib.sha256(f"{SEED}:{work_id}:{number}:{match.start()}".encode()).hexdigest()
            shown = sentence[max(0, match.start() - CONTEXT) : match.end() + CONTEXT]
            examples.append((key, work_id, decade, shown))
            if len(examples) > 4 * EXAMPLES:
                examples = sorted(examples)[:EXAMPLES]
    return {"counts": dict(counts), "examples": sorted(examples)[:EXAMPLES]}


def main() -> None:
    """Scan every part and write the counts and the examples."""
    out = Path(sys.argv[1])
    processes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    out.mkdir(parents=True, exist_ok=True)
    catalog = read_catalog()
    parts = abstracts_parts(catalog)
    with LiveProgress(Console()) as live:
        live.stage("scanning sentences", total=len(parts))
        results = map_parts(
            scan_part, dict.fromkeys(parts, catalog), process_count=processes, progress=live
        )
    counts: Counter[str] = Counter()
    examples: list[Any] = []
    for result in results.values():
        counts.update(result["counts"])
        examples = sorted(examples + result["examples"])[:EXAMPLES]
    (out / "counts.json").write_text(json.dumps(dict(sorted(counts.items())), indent=1) + "\n")
    (out / "examples.json").write_text(json.dumps(examples, indent=1, ensure_ascii=False) + "\n")
    sys.stdout.write(
        f"{counts['splits']:,} candidate splits in {counts['sentences']:,} sentences\n"
    )


if __name__ == "__main__":
    main()
