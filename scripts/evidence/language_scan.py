"""Which languages occur in the corpus: lingua with all its languages on 50,000 abstracts.

Decision agenda 2.4: lingua's candidates were English and four languages the old checks looked
for, and other languages were pushed into one of them. This scan reads run 1's abstracts table
(its text and sentence offsets), takes the whole-abstract verdict of every sampled abstract, and
labels each sentence of the abstracts not confidently English, with the minimum relative
distance the pass uses, so too-close calls count as undetermined. Its counts set the candidate
list in params.yaml.

Run: uv run python scripts/evidence/language_scan.py [SAMPLE_SIZE]
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from collections.abc import Sequence

import polars as pl
from lingua import Language, LanguageDetector, LanguageDetectorBuilder

TABLE = "data/processed/abstracts/*.parquet"
SEED = 0
ENGLISH_THRESHOLD = 0.9
MINIMUM_RELATIVE_DISTANCE = 0.1
BATCH = 2_000


def detector() -> LanguageDetector:
    return (
        LanguageDetectorBuilder.from_all_languages()
        .with_minimum_relative_distance(MINIMUM_RELATIVE_DISTANCE)
        .with_preloaded_language_models()
        .build()
    )


def in_batches(texts: Sequence[str], label: str) -> list[list[str]]:
    batches = [list(texts[start : start + BATCH]) for start in range(0, len(texts), BATCH)]
    print(f"{label}: {len(texts):,} texts in {len(batches)} batches", flush=True)
    return batches


def report(label: str, counts: Counter[str], total: int) -> None:
    print(f"\n{label} ({total:,})")
    for code, count in counts.most_common():
        print(f"  {code:4} {count:>8,}  {count / total:7.3%}")


def code(language: Language | None) -> str:
    return "und" if language is None else language.iso_code_639_1.name.lower()


def main(sample_size: int) -> None:
    started = time.monotonic()
    table = (
        pl.scan_parquet(TABLE).select("text", "sentences").collect().sample(sample_size, seed=SEED)
    )
    lingua = detector()
    print(f"loaded {table.height:,} abstracts and lingua in {time.monotonic() - started:.0f} s")

    whole: Counter[str] = Counter()
    unsure_rows: list[int] = []
    texts = table["text"].to_list()
    for index, batch in enumerate(in_batches(texts, "whole abstracts")):
        for offset, values in enumerate(
            lingua.compute_language_confidence_values_in_parallel(batch)
        ):
            top = values[0] if values else None
            language = code(top.language) if top else "und"
            whole[language] += 1
            if language != "en" or (top is not None and top.value < ENGLISH_THRESHOLD):
                unsure_rows.append(index * BATCH + offset)
        print(f"  batch {index + 1}: {time.monotonic() - started:.0f} s", flush=True)

    sentences = [
        text[sentence["start"] : sentence["end"]]
        for text, offsets in zip(
            table["text"].gather(unsure_rows).to_list(),
            table["sentences"].gather(unsure_rows).to_list(),
            strict=True,
        )
        for sentence in offsets
        if any(character.isalpha() for character in text[sentence["start"] : sentence["end"]])
    ]
    by_sentence: Counter[str] = Counter()
    for index, batch in enumerate(in_batches(sentences, "sentences of unsure abstracts")):
        by_sentence.update(code(found) for found in lingua.detect_languages_in_parallel_of(batch))
        print(f"  batch {index + 1}: {time.monotonic() - started:.0f} s", flush=True)

    report("Whole-abstract top language", whole, table.height)
    print(f"\nnot confidently English: {len(unsure_rows):,} abstracts")
    report("Sentence languages in those abstracts", by_sentence, len(sentences))
    print(f"\ndone in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50_000)
