"""Score round K (K1-K4): does HCI's register move survive inside fixed venues?

Committed before it was run. Rebuilds the HCI 1990s and 2020s cells with their work ids,
restricts them to the venues present in both decades (the venue-stable cell HCI_v), and
scores E(X -> HCI_v) and E(X -> HCI_all) in one run per seed with the corrected design at
100,000 training tokens and 500 held-out abstracts. If HCI_v does not qualify in a decade the
round is reported as untestable and no excess is computed.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from pathlib import Path

from interpretant.corpus.lemmas import LemmaMap
from interpretant.corpus.openalex import OpenAlexSource
from interpretant.corpus.phrases import PhraseModel
from interpretant.drift.perplexity import (
    MIN_VOCABULARY_COUNT,
    KneserNeyBigram,
    read_cell,
    remove_shared,
    training_sample,
)

RAW = Path("data/raw/openalex")
TEXTS = Path("data/phrased/openalex/texts")
SEEDS = (1, 2, 3, 4, 5)
EARLY, LATE = 1990, 2020
BUDGET = 100_000
HELD_OUT = 500
SHARED_MIN = 100
MODELS = ("psychology", "sociology", "ai")


def phrased_lines(decade: int) -> tuple[list[str], list[str]]:
    """The HCI cell's documents as the lemmatised, phrased lines the models read, with ids."""
    cell = OpenAlexSource(RAW, "hci").cell(decade)
    lemmas = LemmaMap.load(Path("data/processed/lemmas.tsv"))
    phrases = PhraseModel.load(Path("data/lemmatised/phrases.model"))
    lines = [phrases.apply(lemmas.apply(document)) for document in cell.documents]
    _, on_disk = read_cell(TEXTS / f"hci_{decade}.txt")
    if set(lines) != set(on_disk):
        raise SystemExit(f"rebuilt {decade}s lines differ from {TEXTS}/hci_{decade}.txt")
    return lines, cell.ids


def sources(decade: int) -> dict[str, str | None]:
    path = RAW / "hci" / f"sources_{decade}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {row["id"]: row["source_id"] for row in rows}


def source_names(decade: int) -> dict[str, str]:
    path = RAW / "hci" / f"sources_{decade}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {row["source_id"]: row["source_name"] for row in rows if row["source_id"]}


def split(lines: list[str], seed: int, held_out: int) -> tuple[list[str], list[str]]:
    shuffled = list(lines)
    random.Random(seed).shuffle(shuffled)
    return shuffled[held_out:], shuffled[:held_out]


def rates_and_counts(
    model: KneserNeyBigram, own: KneserNeyBigram, held_out: list[str]
) -> tuple[dict[str, float], dict[str, int]]:
    other_bits, counts = model.word_bits(held_out)
    own_bits, _ = own.word_bits(held_out)
    return {w: (other_bits[w] - own_bits[w]) / counts[w] for w in counts}, dict(counts)


def report(name: str, passed: bool, detail: str) -> None:
    print(f"\n{name}: {'PASSED' if passed else 'FALSIFIED'}\n  {detail}")


def main() -> None:
    lines = {decade: phrased_lines(decade) for decade in (EARLY, LATE)}
    venue_of = {decade: sources(decade) for decade in (EARLY, LATE)}
    names = {**source_names(EARLY), **source_names(LATE)}
    counts = {
        decade: Counter(venue_of[decade].get(work_id) for work_id in lines[decade][1])
        for decade in (EARLY, LATE)
    }
    shared = sorted(
        source
        for source in set(counts[EARLY]) & set(counts[LATE])
        if source and counts[EARLY][source] >= SHARED_MIN and counts[LATE][source] >= SHARED_MIN
    )
    stable = {
        decade: [
            line
            for line, work_id in zip(*lines[decade], strict=True)
            if venue_of[decade].get(work_id) in shared
        ]
        for decade in (EARLY, LATE)
    }

    print("Shared venues (at least 100 cleaned documents in both decades):")
    for source in shared:
        print(f"  {names.get(source, source)}: {counts[EARLY][source]} / {counts[LATE][source]}")
    for decade in (EARLY, LATE):
        total = len(lines[decade][0])
        print(
            f"{decade}s: {total} cleaned documents, {len(stable[decade])} in shared venues "
            f"({len(stable[decade]) / total:.1%}), {counts[decade][None]} without a source, "
            f"{len(counts[decade]) - 1} distinct sources"
        )
    growth = {
        source: counts[LATE][source] / len(lines[LATE][0])
        - counts[EARLY][source] / len(lines[EARLY][0])
        for source in set(counts[EARLY]) | set(counts[LATE])
        if source
    }
    print(
        "Sources that grew most as a share of the cell:",
        [names.get(s, s) for s in sorted(growth, key=growth.__getitem__, reverse=True)[:10]],
    )

    # Qualification, before any excess is computed.
    for decade in (EARLY, LATE):
        pool_tokens = min(
            sum(len(line.split()) for line in split(stable[decade], seed, HELD_OUT)[0])
            for seed in SEEDS
        )
        print(f"{decade}s HCI_v: smallest training pool over seeds {pool_tokens:,} tokens")
        if pool_tokens < BUDGET:
            print(
                f"\nUNTESTABLE: HCI_v in the {decade}s has fewer than {BUDGET:,} training tokens "
                "after holding out 500 abstracts; no excess computed."
            )
            return

    excess: dict[tuple[int, int, str, str], float] = {}
    mix_rate: dict[tuple[str, int], tuple[float, float, float]] = {}
    per_word: dict[tuple[str, int, int], tuple[dict[str, float], dict[str, int]]] = {}
    for seed in SEEDS:
        for decade in (EARLY, LATE):
            contents = dict(read_cell(path) for path in sorted(TEXTS.glob(f"*_{decade}.txt")))
            contents = remove_shared(contents)
            kept = set(contents["hci"])
            texts = {field: split(docs, seed, HELD_OUT) for field, docs in contents.items()}
            texts["hci_v"] = split(
                [line for line in stable[decade] if line in kept], seed, HELD_OUT
            )
            samples = {
                field: training_sample(pool, BUDGET)
                for field, (pool, _) in texts.items()
                if sum(len(line.split()) for line in pool) >= BUDGET
            }
            vocabulary_counts: Counter[str] = Counter()
            for sample in samples.values():
                for line in sample:
                    vocabulary_counts.update(line.split())
            vocabulary = {w for w, c in vocabulary_counts.items() if c >= MIN_VOCABULARY_COUNT}
            models = {f: KneserNeyBigram(vocabulary).fit(s) for f, s in samples.items()}
            for text in ("hci", "hci_v"):
                held_out = texts[text][1]
                own = models[text].cross_entropy(held_out)
                for model_field in MODELS:
                    excess[(seed, decade, model_field, text)] = (
                        models[model_field].cross_entropy(held_out) - own
                    )
                    if text == "hci_v" and model_field != "ai":
                        per_word[(model_field, seed, decade)] = rates_and_counts(
                            models[model_field], models["hci_v"], held_out
                        )
        for model_field in ("psychology", "sociology"):
            rates_early, counts_early = per_word[(model_field, seed, EARLY)]
            rates_late, counts_late = per_word[(model_field, seed, LATE)]
            n_early, n_late = sum(counts_early.values()), sum(counts_late.values())
            mean_early = sum(rates_early[w] * c for w, c in counts_early.items()) / n_early
            mean_late = sum(rates_late[w] * c for w, c in counts_late.items()) / n_late
            counterfactual = (
                sum(rates_early.get(w, rates_late[w]) * c for w, c in counts_late.items()) / n_late
            )
            mix_rate[(model_field, seed)] = (
                mean_early - mean_late,
                mean_early - counterfactual,
                counterfactual - mean_late,
            )

    def fall(seed: int, model_field: str, text: str) -> float:
        return excess[(seed, EARLY, model_field, text)] - excess[(seed, LATE, model_field, text)]

    def mean_fall(model_field: str, text: str) -> float:
        return statistics.fmean(fall(seed, model_field, text) for seed in SEEDS)

    print("\nFalls 1990s -> 2020s, per seed (bits per token):")
    for model_field in MODELS:
        for text in ("hci_v", "hci"):
            print(
                f"  {model_field:>10} -> {text:5}: "
                + ", ".join(f"{fall(seed, model_field, text):.3f}" for seed in SEEDS)
                + f"   mean {mean_fall(model_field, text):.3f}"
            )

    k1 = all(fall(seed, m, "hci_v") > 0 for m in ("psychology", "sociology") for seed in SEEDS)
    report(
        "K1 fall survives inside fixed venues",
        k1,
        "see the per-seed falls above (psychology -> hci_v, sociology -> hci_v)",
    )
    k2_lines, k2 = [], True
    for model_field in ("psychology", "sociology"):
        hits = sum(
            1
            for seed in SEEDS
            if mix_rate[(model_field, seed)][1] > mix_rate[(model_field, seed)][2]
        )
        k2 &= hits >= 4
        k2_lines.append(
            f"{model_field}: mix > rate in {hits} of 5; (fall, mix, rate) "
            + str([tuple(round(v, 3) for v in mix_rate[(model_field, seed)]) for seed in SEEDS])
        )
    report("K2 still the word mix", k2, "\n  ".join(k2_lines))
    ratios = {m: mean_fall(m, "hci_v") / mean_fall(m, "hci") for m in ("psychology", "sociology")}
    report(
        "K3 most of the effect is the field's",
        all(ratio >= 0.5 for ratio in ratios.values()),
        "mean fall on HCI_v / mean fall on HCI_all: "
        + str({m: round(r, 2) for m, r in ratios.items()}),
    )
    report(
        "K4 computing still does not gain",
        mean_fall("ai", "hci_v") < mean_fall("psychology", "hci_v") / 3,
        f"AI mean fall on HCI_v {mean_fall('ai', 'hci_v'):.3f} vs psychology's "
        f"{mean_fall('psychology', 'hci_v'):.3f}",
    )


if __name__ == "__main__":
    main()
