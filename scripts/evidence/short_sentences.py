"""How many usages sit in short sentences, per word and decade, for the short-sentence decision.

The SemEval-2020 and DWUG benchmarks take uses from sentences of at least 10 tokens (Schlechtweg et
al. 2020, 2021). Step 6 records each usage's sentence length (its words holding a letter,
`corpus/sentence_view.word_count`) and leaves the decision to the data
(`docs/preprocessing_phase.md`, step 6). For each threshold this script counts the eligible usages
(in scope, no exclusion) in shorter sentences, and what leaving them out of the draw would do to
the draws: a work and word whose drawn usage is short either draws another usage or loses its draw.
Held-out words are counted together, never named. Run from the repository root:
`python scripts/evidence/short_sentences.py OUT.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import polars as pl

from interpretant.catalog.config import read_catalog
from interpretant.catalog.words import load_words

THRESHOLDS = (5, 8, 10)


def usages_with_decades() -> pl.DataFrame:
    """The eligible usages of every saved part, with the part's decade."""
    catalog = read_catalog()
    frames = []
    for path in sorted(catalog.usages.path.glob("*.parquet")):
        decade = int(path.stem.rsplit("_", 1)[1])
        frames.append(
            pl.read_parquet(
                path, columns=["work_id", "word", "sentence_word_count", "in_scope", "drawn"]
            )
            .filter(pl.col("in_scope"))
            .with_columns(pl.lit(decade).alias("decade"))
        )
    return pl.concat(frames)


def draw_effect(usages: pl.DataFrame, threshold: int) -> dict[str, int]:
    """How many work-and-word draws stay, move to a longer sentence, or are lost."""
    short = pl.col("sentence_word_count") < threshold
    per_draw = usages.group_by("work_id", "word").agg(
        (pl.col("drawn") & short).any().alias("drawn_short"),
        (~short).any().alias("has_long"),
    )
    moved = per_draw.filter(pl.col("drawn_short") & pl.col("has_long")).height
    lost = per_draw.filter(pl.col("drawn_short") & ~pl.col("has_long")).height
    return {"draws": per_draw.height, "move": moved, "lost": lost}


def main() -> None:
    """Count short-sentence usages per threshold, word and decade, and write them."""
    out = Path(sys.argv[1])
    held_out = {word.lemma for word in load_words(read_catalog()) if word.held_out}
    usages = usages_with_decades()
    results: dict[str, Any] = {"eligible_usages": usages.height, "thresholds": {}}
    for threshold in THRESHOLDS:
        short = pl.col("sentence_word_count") < threshold
        by_word = (
            usages.group_by("word")
            .agg(pl.len().alias("usages"), short.sum().alias("short"))
            .with_columns((pl.col("short") / pl.col("usages")).round(4).alias("share"))
        )
        by_decade = (
            usages.group_by("decade")
            .agg(pl.len().alias("usages"), short.sum().alias("short"))
            .with_columns((pl.col("short") / pl.col("usages")).round(4).alias("share"))
            .sort("decade")
        )
        visible = by_word.filter(~pl.col("word").is_in(held_out)).sort("share", descending=True)
        results["thresholds"][str(threshold)] = {
            "short_usages": int(usages.select(short.sum()).item()),
            "share": round(usages.select(short.mean()).item(), 4),
            "draws": draw_effect(usages, threshold),
            "by_decade": by_decade.to_dicts(),
            "by_word_highest": visible.head(10).to_dicts(),
            "held_out_words": len(held_out),
        }
        top = results["thresholds"][str(threshold)]
        sys.stdout.write(
            f"under {threshold} words: {top['short_usages']} usages ({top['share']:.2%}); "
            f"draws {top['draws']}\n"
        )
    out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
