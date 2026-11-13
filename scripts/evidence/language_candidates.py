"""How well each candidate list labels sentences, against lingua with all its languages.

Decision agenda 2.4, after scripts/evidence/language_scan.py: the same 50,000 abstracts; for the
abstracts not confidently English, each sentence is labelled by lingua with all its languages
(the reference) and by each candidate list, both with the pass's minimum relative distance.
Counted per list: sentences the reference reads as another language that the list reads as
English (foreign text kept), sentences the reference reads as English that the list reads as
another language (English lost), and the time per 10,000 sentences.

Run: uv run python scripts/evidence/language_candidates.py
"""

from __future__ import annotations

import time

import polars as pl
from lingua import IsoCode639_1, Language, LanguageDetectorBuilder

TABLE = "data/processed/abstracts/*.parquet"
SEED = 0
SAMPLE = 50_000
ENGLISH_THRESHOLD = 0.9
DISTANCE = 0.1
# The scan's languages with at least 0.5% and at least 0.1% of the sentences of abstracts not
# confidently English; tl, yo and so were nearly all misreadings, la about half.
OVER_HALF_PERCENT = "en fr ko id de es la ja pt zh ru ms fi tr nl uk it bs hr ar".split()
OVER_TENTH_PERCENT = [
    *OVER_HALF_PERCENT,
    *"af cs sv ca fa et sl nb da hu az pl tl bg is sk ro".split(),
]
LISTS = {
    "0.5%": OVER_HALF_PERCENT,
    "0.5% without la": [code for code in OVER_HALF_PERCENT if code != "la"],
    "0.1%": OVER_TENTH_PERCENT,
    "0.1% without la, tl": [code for code in OVER_TENTH_PERCENT if code not in {"la", "tl"}],
    "old five": "en de es fr pt".split(),
}


def code(language: Language | None) -> str:
    """The ISO 639-1 code, or und when lingua gave no verdict."""
    return "und" if language is None else language.iso_code_639_1.name.lower()


def main() -> None:
    """Label the unsure abstracts' sentences with each list and print the disagreements."""
    table = pl.scan_parquet(TABLE).select("text", "sentences").collect().sample(SAMPLE, seed=SEED)
    reference = (
        LanguageDetectorBuilder.from_all_languages()
        .with_minimum_relative_distance(DISTANCE)
        .with_preloaded_language_models()
        .build()
    )
    values = reference.compute_language_confidence_values_in_parallel(table["text"].to_list())
    unsure = [
        row
        for row, found in enumerate(values)
        if not found or code(found[0].language) != "en" or found[0].value < ENGLISH_THRESHOLD
    ]
    sentences = [
        text[offset["start"] : offset["end"]]
        for text, offsets in zip(
            table["text"].gather(unsure).to_list(),
            table["sentences"].gather(unsure).to_list(),
            strict=True,
        )
        for offset in offsets
        if any(character.isalpha() for character in text[offset["start"] : offset["end"]])
    ]
    truth = [code(found) for found in reference.detect_languages_in_parallel_of(sentences)]
    foreign = sum(label not in {"en", "und"} for label in truth)
    english = truth.count("en")
    print(f"{len(sentences):,} sentences: reference {english:,} English, {foreign:,} other")
    for name, codes in LISTS.items():
        languages = [Language.from_iso_code_639_1(getattr(IsoCode639_1, c.upper())) for c in codes]
        detector = (
            LanguageDetectorBuilder.from_languages(*languages)
            .with_minimum_relative_distance(DISTANCE)
            .with_preloaded_language_models()
            .build()
        )
        started = time.monotonic()
        labels = [code(found) for found in detector.detect_languages_in_parallel_of(sentences)]
        seconds = time.monotonic() - started
        kept = sum(t not in {"en", "und"} and label == "en" for t, label in zip(truth, labels))
        lost = sum(t == "en" and label not in {"en", "und"} for t, label in zip(truth, labels))
        print(
            f"{name:22} {len(codes):2} languages: foreign kept as English {kept:4} "
            f"({kept / foreign:.1%}), English lost {lost:4} ({lost / english:.1%}), "
            f"{seconds / len(sentences) * 10_000:.2f} s per 10,000 sentences"
        )


if __name__ == "__main__":
    main()
