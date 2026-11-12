"""Round E (E1-E6): a classifier on XL-LEXEME trained from cue words, on DWUG EN's half B.

Committed before it was run (docs/predictions_labels.md). Steps:

    uv run python scripts/score_labels_classifier.py encode   # XL-LEXEME vectors, stored
    uv run python scripts/score_labels_classifier.py label    # seeds, classifiers, labels
    uv run python scripts/score_labels_classifier.py score

Labels go to data/labels/round_e/ as DWUG identifiers and labels only; vectors stay local.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
import tomllib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import score_labels_benchmark as round_y
import score_labels_induced as round_q

from interpretant.drift.classifier import classify, cue_sense, example_usage
from interpretant.drift.labels import OTHER, UNCLEAR, load_sense_lists
from interpretant.drift.pairs import Usage, XLLexeme
from interpretant.predictions import report

OUT = Path("data/labels/round_e")
CUES = Path("data/sense_lists/dwug_en_cues.toml")
VECTORS = OUT / "vectors.npz"
REFERENCE = {"sonnet": 0.693, "jev": 0.800}


def examples() -> dict[str, Usage]:
    """Each sense's invented example as a usage, by ``lemma|sense``."""
    lists = load_sense_lists(round_y.LISTS)
    found = {}
    for lemma, senses in lists.items():
        for sense in senses.senses:
            usage = example_usage(senses.word, sense.example)
            if usage is not None:
                found[f"{lemma}|{sense.key}"] = usage
    return found


def encode() -> None:
    """XL-LEXEME vectors of every kept DWUG usage and every example, stored once."""
    corpus = sorted(round_y.usages().items())
    extra = sorted(examples().items())
    keys = [key for key, _ in corpus] + [f"example|{key}" for key, _ in extra]
    usages = [usage for _, (_, _, usage) in corpus] + [usage for _, usage in extra]
    vectors = XLLexeme().encode(usages)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(VECTORS, keys=np.array(keys), vectors=vectors)
    print(f"{len(keys)} vectors stored")


def vectors() -> dict[str, np.ndarray]:
    stored = np.load(VECTORS)
    return {str(key): row for key, row in zip(stored["keys"], stored["vectors"], strict=True)}


def write(name: str, labels: dict[str, str]) -> None:
    with (OUT / f"{name}_labels.tsv").open("w", encoding="utf-8", newline="") as table:
        writer = csv.writer(table, delimiter="\t")
        writer.writerow(["usage", "label"])
        writer.writerows(sorted(labels.items()))


def label() -> None:
    """Seeds from half A, a classifier per word, labels for half B; and examples alone."""
    found = round_y.usages()
    half = round_q.halves()
    lists = load_sense_lists(round_y.LISTS)
    cues = tomllib.loads(CUES.read_text(encoding="utf-8"))
    vector_of = vectors()
    full: dict[str, str] = {}
    alone: dict[str, str] = {}
    seeds_per_sense: dict[str, int] = {}
    for lemma, senses in sorted(lists.items()):
        keys_a = sorted(k for k in found if found[k][0] == lemma and half[k] == "A")
        keys_b = sorted(k for k in found if found[k][0] == lemma and half[k] == "B")
        cue_seeds = [(k, cue_sense(found[k][2], cues[lemma])) for k in keys_a]
        cue_seeds = [(k, s) for k, s in cue_seeds if s is not None]
        example_seeds = [
            (f"example|{lemma}|{s.key}", s.key)
            for s in senses.senses
            if f"example|{lemma}|{s.key}" in vector_of
        ]
        for sense in senses.senses:
            seeds_per_sense[f"{lemma}|{sense.key}"] = sum(s == sense.key for _, s in cue_seeds)
        targets = np.array([vector_of[k] for k in keys_b])
        for seeds, store in ((cue_seeds + example_seeds, full), (example_seeds, alone)):
            predicted = classify(
                np.array([vector_of[k] for k, _ in seeds]), [s for _, s in seeds], targets
            )
            store.update(zip(keys_b, predicted, strict=True))
    OUT.mkdir(parents=True, exist_ok=True)
    write("classifier", full)
    write("examples_only", alone)
    (OUT / "seeds_per_sense.json").write_text(json.dumps(seeds_per_sense, indent=1), "utf-8")
    print(f"{len(full)} half-B usages labelled; {sum(seeds_per_sense.values())} cue seeds")


def score() -> None:
    lists = load_sense_lists(round_y.LISTS)
    half = round_q.halves()
    labels = round_q.read(OUT / "classifier_labels.tsv")
    rho, partial, rand = round_q.measures(labels, lists)
    report(
        "E1 the classifier ranks the benchmark's changes",
        rho >= 0.70,
        f"Spearman {rho:.3f} on half B (needed 0.70; Sonnet 0.693, JEV 0.800, gold 0.966)",
    )
    report(
        "E2 it does about as well as a language model labeller",
        rho >= REFERENCE["sonnet"] - 0.10,
        f"Spearman {rho:.3f} (needed {REFERENCE['sonnet'] - 0.10:.3f}, Sonnet's 0.693 minus 0.10)",
    )
    report(
        "E3 it tracks change beyond variety",
        partial >= 0.40,
        f"partial Spearman {partial:.3f} (needed 0.40; gold 0.849)",
    )
    for name in ("sonnet", "jev"):
        reference = round_q.read(round_y.OUT / f"{name}_labels.tsv")
        shared = sorted(labels)
        same = statistics.mean(labels[k] == reference[k] for k in shared)
        fitted = [k for k in shared if reference[k] not in (OTHER, UNCLEAR)]
        same_fitted = statistics.mean(labels[k] == reference[k] for k in fitted)
        report(
            f"E4 agreement with {name}",
            "descriptive",
            f"same label on {same:.1%} of {len(shared)} half-B usages; {same_fitted:.1%} of the "
            f"{len(fitted)} where {name} chose a listed sense",
        )
    seeds = json.loads((OUT / "seeds_per_sense.json").read_text(encoding="utf-8"))
    report(
        "E5 seeds",
        "descriptive",
        f"{sum(seeds.values())} cue seeds over {len(seeds)} senses; "
        f"{statistics.mean(v >= 3 for v in seeds.values()):.0%} of senses have three or more; "
        f"median {statistics.median(seeds.values()):.0f}",
    )
    alone = round_q.read(OUT / "examples_only_labels.tsv")
    rho_alone, _, _ = round_q.measures(alone, lists)
    report(
        "E6 examples alone",
        "descriptive",
        f"Spearman {rho_alone:.3f} on half B with one invented example per sense",
    )
    report("E (grouping)", "descriptive", f"mean ARI on half B {rand:.3f}")
    print(f"half B: {sum(v == 'B' for v in half.values())} usages")


if __name__ == "__main__":
    {"encode": encode, "label": label, "score": score}[sys.argv[1]]()
