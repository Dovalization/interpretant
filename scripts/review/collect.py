"""Review pages for the sense lists, stage 1: every sampled usage of the lists' words.

The author reviews draft sense lists and cue words on a published page (docs/plan_sense_labels.md,
"Measuring shares on this corpus"). Three stages build it:

    uv run python scripts/review/collect.py data/labels/review/raw/usages.jsonl [LISTS.toml]
    uv run python scripts/review/build.py data/labels/review/raw/usages.jsonl \
        data/labels/review/raw/vectors.npz data/labels/review/raw/previews.json [LISTS.toml]
    uv run python scripts/review/render.py data/labels/review/raw/previews.json \
        data/labels/review/raw/review.html "Sense list review"

LISTS.toml defaults to data/sense_lists/corpus_draft.toml. Stage 1 reads every field and decade of
the grid (at most 150 abstracts per cell, seed 1, round C′'s quality rules) once, so cue changes
only rerun stages 2 and 3. Everything under raw/ holds abstract text and stays uncommitted.
"""

import json
import sys
import tomllib
from pathlib import Path

from interpretant.drift.senses import Sampling, cased_abstracts, sample_occurrences

GRID = Path("data/raw/openalex")
words = list(
    tomllib.loads(
        Path(sys.argv[2] if len(sys.argv) > 2 else "data/sense_lists/corpus_draft.toml").read_text()
    )
)
fields = sorted(p.name for p in GRID.iterdir() if p.is_dir())
with open(sys.argv[1], "w", encoding="utf-8") as out:
    for field in fields:
        for decade in range(1950, 2030, 10):
            try:
                cell = {(field, decade): cased_abstracts(GRID, field, decade)}
            except Exception:
                continue
            for word in words:
                for o in sample_occurrences(
                    cell, word, seed=1, sampling=Sampling(cap=150, plural=True, quality=True)
                ):
                    out.write(
                        json.dumps(
                            {
                                "word": word,
                                "field": field,
                                "decade": decade,
                                "text": o.context,
                                "start": o.start,
                                "end": o.end,
                            }
                        )
                        + "\n"
                    )
        print(field, "done", flush=True)
