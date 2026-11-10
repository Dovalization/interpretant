"""Measure how many of the sense rounds' abstracts a field classifier places in another field.

For each decade, round S's classifier (TF-IDF and logistic regression, seed 1's training split
of the equal-size grid) scores every abstract of the planned sense cells that contains the
word, singular or plural; abstracts it was trained on are left out. An abstract is *flagged*
when the classifier gives another field a probability of at least ``FLAG``. Flags mix
mislabelled abstracts with real cross-field work, so they are read, and used as a robustness
check in round D, never as a filter. Writes the flagged abstracts to
data/drift/label_noise.parquet and prints the share flagged per cell.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

from interpretant.corpus.lemmas import LemmaMap
from interpretant.corpus.openalex import OpenAlexSource
from interpretant.corpus.phrases import PhraseModel
from interpretant.drift.separability import decade_samples, fit_fields, split

sys.path.insert(0, str(Path(__file__).parent))
from fetch_sense_words import PLAN, WORDS_DIR  # noqa: E402

TEXTS = Path("data/phrased/openalex/texts")
GRID = Path("data/raw/openalex")
OUTPUT = Path("data/drift/label_noise.parquet")
FLAG = 0.8


def main() -> None:
    lemmas = LemmaMap.load(Path("data/processed/lemmas.tsv"))
    phrases = PhraseModel.load(Path("data/lemmatised/phrases.model"))
    decades = sorted({decade for _, cell_decades in PLAN.values() for decade in cell_decades})
    classifiers = {}
    trained: dict[int, set[str]] = {}
    for decade in decades:
        train, _ = split(decade_samples(TEXTS, decade, seed=1))
        classifiers[decade] = fit_fields(train)
        trained[decade] = {text for texts in train.values() for text in texts}
        print(f"{decade}s classifier fitted on {sum(map(len, train.values())):,} abstracts")

    flagged: list[dict[str, object]] = []
    for word, (fields, cell_decades) in PLAN.items():
        raw_dir = WORDS_DIR / word if (WORDS_DIR / word).exists() else GRID
        pattern = re.compile(rf"\b{re.escape(word)}s?\b")
        print(f"\n{word} ({'word-targeted' if raw_dir != GRID else 'grid'}): share flagged")
        for field in fields:
            row = []
            for decade in cell_decades:
                cell = OpenAlexSource(raw_dir, field).cell(decade)
                kept = [
                    (work_id, line)
                    for work_id, document in zip(cell.ids, cell.documents, strict=True)
                    if pattern.search(document)
                    for line in [phrases.apply(lemmas.apply(document))]
                    if line not in trained[decade]
                ]
                if not kept:
                    row.append(f"{decade}s    -")
                    continue
                model = classifiers[decade]
                probabilities = model.predict_proba([line for _, line in kept])
                classes = [str(label) for label in model.classes_]
                own = classes.index(field)
                count = 0
                for (work_id, line), row_probabilities in zip(kept, probabilities, strict=True):
                    best = int(row_probabilities.argmax())
                    if best != own and row_probabilities[best] >= FLAG:
                        count += 1
                        flagged.append(
                            {
                                "word": word,
                                "field": field,
                                "decade": decade,
                                "work_id": work_id,
                                "assigned": classes[best],
                                "probability": float(row_probabilities[best]),
                                "own_probability": float(row_probabilities[own]),
                                "text": line,
                            }
                        )
                row.append(f"{decade}s {count / len(kept):>5.1%} of {len(kept):>4}")
            print(f"  {field:14}  " + "   ".join(row), flush=True)

    table = pd.DataFrame(flagged)
    table.to_parquet(OUTPUT, index=False)
    print(f"\n{len(table):,} flagged abstracts → {OUTPUT}")
    destinations = table.groupby(["word", "field"]).assigned.agg(
        lambda column: ", ".join(f"{k} {v}" for k, v in column.value_counts().head(3).items())
    )
    print("\nWhere flagged abstracts go (top three):")
    print(destinations.to_string())


if __name__ == "__main__":
    main()
