"""A fix's check: the usages one word gains from one tag, drawn and judged (`docs/usages_audit.md`).

After a failed bar a rule may be fixed for one word without redrawing the audit (the author,
2026-11-21): the usages the fix brings into scope are drawn, judged like the precision items, and
pass with at most 2 errors in 20, the precision screen's bar per word.

`draw OUT WORD POS` draws 20 drawn in-scope usages of WORD tagged POS, the smallest audit keys
under seed 20261308, and writes them as one batch of kept items with their key (the items quote
corpus sentences and stay out of git). `score OUT` reads the judges' and the adjudicator's verdicts,
writes `results.json` and says whether the fix passes. Run from the repository root:
`python scripts/review/usages_fix_check.py draw OUT WORD POS` or `... score OUT`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import polars as pl

from interpretant.catalog.abstracts import load_abstracts_part
from interpretant.catalog.config import read_catalog
from interpretant.catalog.words import load_words
from interpretant.pipelines.draw_usages_audit import load_usages_with_decades
from interpretant.review.usages_audit import build_with_keys, render_marked_sentence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_usages import final_verdicts

CHECK_SEED = 20261308
SIZE = 20
MOST_ERRORS = 2


def draw(out: Path, lemma: str, pos: str) -> None:
    """Draw the gained usages and write them as items with their key."""
    catalog = read_catalog()
    word = next(word for word in load_words(catalog) if word.lemma == lemma)
    usages = load_usages_with_decades(catalog).filter(
        (pl.col("word") == lemma) & (pl.col("pos") == pos) & pl.col("in_scope") & pl.col("drawn")
    )
    chosen = build_with_keys(usages, CHECK_SEED).sort("key").head(SIZE).to_dicts()
    texts: dict[str, str] = {}
    for part in sorted({row["part"] for row in chosen}):
        ids = sorted(row["work_id"] for row in chosen if row["part"] == part)
        frame = load_abstracts_part(catalog, part, ["work_id", "text"])
        texts |= dict(frame.filter(pl.col("work_id").is_in(ids)).iter_rows())
    random.Random(CHECK_SEED).shuffle(chosen)
    out.mkdir(parents=True, exist_ok=True)
    key = {}
    with (out / "batch_00.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(chosen):
            item_id = f"F{index:03d}"
            item = {
                "id": item_id,
                "kind": "kept",
                "word": lemma,
                "parts_of_speech": list(word.pos),
                "sentence": render_marked_sentence(texts[row["work_id"]], row),
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            key[item_id] = {
                "kind": "kept",
                "work_id": row["work_id"],
                "word": word.reported_name,
                "target_start": row["target_start"],
                "target_end": row["target_end"],
                "part": row["part"],
            }
    (out / "key.json").write_text(json.dumps(key, indent=1) + "\n")
    sys.stdout.write(f"{len(key)} of {usages.height} gained usages drawn into {out}\n")


def score(out: Path) -> None:
    """Count the errors among the judged items and write whether the fix passes."""
    key = json.loads((out / "key.json").read_text())
    final, agreed, unsettled = final_verdicts(out, key)
    if unsettled:
        raise SystemExit(f"{len(unsettled)} items still need the adjudicator")
    errors = sum(verdict == "wrong" for verdict in final.values())
    results = {
        "judged": len(final),
        "judges_agreed": agreed,
        "errors": errors,
        "bar": MOST_ERRORS,
        "passes": errors <= MOST_ERRORS,
    }
    (out / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    verdict = "passes" if results["passes"] else "fails"
    sys.stdout.write(f"{errors} errors in {len(final)} (bar {MOST_ERRORS}): {verdict}\n")


def main() -> None:
    """Draw the check's items or score them."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    drawing = commands.add_parser("draw")
    drawing.add_argument("out", type=Path)
    drawing.add_argument("word")
    drawing.add_argument("pos")
    scoring = commands.add_parser("score")
    scoring.add_argument("out", type=Path)
    options = parser.parse_args()
    if options.command == "draw":
        draw(options.out, options.word, options.pos)
    else:
        score(options.out)


if __name__ == "__main__":
    main()
