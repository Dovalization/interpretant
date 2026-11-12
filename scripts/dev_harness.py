"""Score classifier variants on all the development data at once (spent rounds and Claude's
labels), reading vectors from the shared stores. Development only: nothing here is a test.

    uv run python scripts/dev_harness.py '{"encoder": "xl-lexeme", "weight": "sqrt"}' [LIST]

Training: each word's cue-seeded usages in the main store (the lists' example sentences are
not in the store and are left out) (every grid and word-targeted usage,
quality rules version 2), with every development text left out. Development sets:

- mine: Claude's 300 blind labels (round U's development set);
- U: round U's 240 usages, where the two labellers agreed or the author chose one;
- A: round A's 2,040 target-word usages, Sonnet's labels as a second opinion (silver);
- T: check 2b's 100 pairs, the author's judgements against "same sense" and probability overlap.
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
import tomllib
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
import pilot_labellers as pilot  # type: ignore[import-not-found]

from interpretant.drift.classifier import (
    cue_first,
    cue_sense,
    normalised,
    out_of_scope,
    sqrt_balanced,
)
from interpretant.drift.pairs import Usage

STORE = Path("data/store")
DEFAULT_LIST = Path("data/sense_lists/corpus_v2.toml")
WORDS = ("attention", "prompt", "platform", "network", "component", "having")
STORES = {
    ("xl-lexeme", "main"): (STORE / "index.json", STORE / "vectors"),
    ("xl-lexeme", "dev"): (STORE / "index_xl-lexeme_dev.json", STORE / "vectors_xl-lexeme_dev"),
    ("deberta", "main"): (STORE / "index_deberta_main.json", STORE / "vectors_deberta_main"),
    ("deberta", "dev"): (STORE / "index_deberta_dev.json", STORE / "vectors_deberta_dev"),
}


@cache
def store(encoder: str, which: str) -> tuple[dict[str, list[object]], Path]:
    index_path, shards = STORES[(encoder, which)]
    return json.loads(index_path.read_text(encoding="utf-8")), shards


def vectors(encoder: str, which: str, keys: list[str]) -> np.ndarray:
    index, shards = store(encoder, which)
    by_shard: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for position, key in enumerate(keys):
        shard, row = index[key]
        by_shard[str(shard)].append((position, int(row)))  # type: ignore[call-overload]
    out = np.zeros((len(keys), 1024), dtype=np.float32)
    for shard, pairs in by_shard.items():
        data = np.load(shards / shard)["vectors"]
        for position, row in pairs:
            out[position] = data[row]
    return out


@cache
def dev_rows() -> tuple[dict[str, object], ...]:
    with gzip.open(STORE / "dev_usages.jsonl.gz", "rt", encoding="utf-8") as source:
        return tuple(json.loads(line) for line in source)


@cache
def training_rows(word: str) -> tuple[dict[str, object], ...]:
    dev_texts = {str(r["text"]) for r in dev_rows()}
    found, seen = [], set()
    with gzip.open(STORE / "usages.jsonl.gz", "rt", encoding="utf-8") as source:
        for line in source:
            if f'"word": "{word}"' not in line:
                continue
            r = json.loads(line)
            if r["word"] != word or r["text"] in seen or r["text"] in dev_texts:
                continue
            seen.add(r["text"])
            found.append(r)
    return tuple(found)


@cache
def known_decades() -> dict[str, int]:
    """Round A's usage ids and the decade of each usage's cell."""
    with open("data/labels/known/raw/measured.jsonl", encoding="utf-8") as source:
        return {r["id"]: int(r["decade"]) for r in map(json.loads, source)}


def usage(row: dict[str, object]) -> Usage:
    return Usage(str(row["text"]), int(row["start"]), int(row["end"]))  # type: ignore[call-overload]


def gold() -> dict[str, dict[str, str]]:
    """Reference labels by development set and usage id."""
    mine = json.loads(Path("data/labels/round_u/raw/dev_labels_claude.json").read_text())

    def read(path: str) -> dict[str, dict[str, str]]:
        with open(path, encoding="utf-8") as table:
            return {r["usage"]: r for r in csv.DictReader(table, delimiter="\t")}

    classifier = read("data/labels/round_u/classifier_labels.tsv")
    sonnet = read("data/labels/round_u/sonnet_labels.tsv")
    choices = read("data/labels/round_u/judgements.tsv")
    key = json.loads(Path("data/labels/round_u/judging_key.json").read_text())
    u: dict[str, str] = {}
    for uid in classifier:
        c, s = classifier[uid]["label"], sonnet[uid]["label"]
        if c == s:
            u[uid] = c
        elif uid in choices and choices[uid]["choice"] in ("first", "second"):
            winner = key[uid][0 if choices[uid]["choice"] == "first" else 1]
            u[uid] = c if winner == "classifier" else s
    a = {r["usage"]: r["label"] for r in read("data/labels/known/sonnet_labels.tsv").values()}
    return {"mine": mine, "U": u, "A": a}


def features(which: str, keys: list[str], names: list[str]) -> np.ndarray:
    """Each encoder's vectors, normalised, side by side."""
    return np.hstack([normalised(vectors(name, which, keys)) for name in names])


def decade_weights(senses: list[str], decades: list[int], power: float) -> np.ndarray:
    """Per-usage weights under which each sense's decades count more evenly: a usage weighs
    its (sense, decade) cell's size to the power ``-power`` (1 makes the decades equal, 0.5
    halfway), rescaled so each sense keeps its total weight. Training only; never a feature."""
    cells = Counter(zip(senses, decades, strict=True))
    raw = np.array([cells[(s, d)] ** -power for s, d in zip(senses, decades, strict=True)])
    totals: Counter[str] = Counter(senses)
    mass: dict[str, float] = defaultdict(float)
    for sense, weight in zip(senses, raw, strict=True):
        mass[sense] += float(weight)
    return np.array([w * totals[s] / mass[s] for s, w in zip(senses, raw, strict=True)])


def tempered(proba: np.ndarray, temperature: float) -> np.ndarray:
    """Probabilities with their log-odds divided by ``temperature`` (labels unchanged)."""
    scaled = np.log(np.clip(proba, 1e-12, 1.0)) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    out: np.ndarray = exp / exp.sum(axis=1, keepdims=True)
    return out


def fit_temperature(proba: np.ndarray, truth: np.ndarray) -> float:
    """The temperature that minimises held-out log loss."""

    def loss(log_t: float) -> float:
        p = tempered(proba, float(np.exp(log_t)))
        return float(-np.log(np.clip(p[np.arange(len(truth)), truth], 1e-12, 1.0)).mean())

    return float(np.exp(minimize_scalar(loss, bounds=(-3, 3), method="bounded").x))


def probe(
    names: list[str],
    train_keys: list[str],
    y: list[str],
    dev_keys: list[str],
    class_weight: object,
    c: float,
    sample_weight: np.ndarray | None = None,
    which: str = "dev",
) -> tuple[np.ndarray, np.ndarray]:
    """A logistic regression on the named encoders' vectors: probabilities and classes for
    ``dev_keys``, read from the ``which`` store."""
    model = LogisticRegression(C=c, class_weight=class_weight, max_iter=3000)  # type: ignore[arg-type]
    model.fit(normalised(features("main", train_keys, names)), y, sample_weight=sample_weight)
    return model.predict_proba(normalised(features(which, dev_keys, names))), model.classes_


def fit(
    keys: list[str],
    labels: list[str],
    decades: list[int],
    targets: list[str],
    which: str,
    groups: list[list[str]],
    class_weight: object,
    c: float,
    power: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One probe per encoder group, averaged: probabilities and classes for ``targets``."""
    weight = decade_weights(labels, decades, power) if power else None
    outputs = [probe(g, keys, labels, targets, class_weight, c, weight, which) for g in groups]
    return sum(o[0] for o in outputs) / len(outputs), outputs[0][1]


def run(variant: dict[str, object], list_path: Path) -> dict[str, object]:
    encoder = str(variant.get("encoder", "xl-lexeme"))
    table = tomllib.loads(list_path.read_text(encoding="utf-8"))
    references = gold()
    external: dict[str, dict[str, Any]] = {}
    if "predictions" in variant:
        external = json.loads(Path(str(variant["predictions"])).read_text())["predictions"]
    results: dict[str, Counter[str]] = defaultdict(Counter)
    probabilities: dict[str, dict[str, float]] = {}
    labels_t: dict[str, str] = {}
    temperatures: dict[str, float] = {}
    lm_shares: dict[str, float] = {}
    losses: dict[str, list[float]] = defaultdict(list)
    for word in WORDS:
        cues = {s["key"]: s["cues"] for s in table[word]["senses"]}
        scope = list(table[word].get("out_of_scope", []))
        seeds = [
            (r, s)
            for r in training_rows(word)
            if not out_of_scope(usage(r), scope) and (s := cue_sense(usage(r), cues))
        ]
        items = [r for r in dev_rows() if r["word"] == word and not out_of_scope(usage(r), scope)]
        if "predictions" in variant:
            chosen = [external[f"{r['set']}|{r['id']}"] for r in items]
            senses_order = [s["key"] for s in table[word]["senses"]]
            classes = np.array(senses_order)
            proba = np.array(
                [[c["probabilities"].get(s, 0.0) for s in senses_order] for c in chosen]
            )
            predicted = [str(c["label"]) for c in chosen]
        else:
            y = [s for _, s in seeds]
            weights = {"balanced": "balanced", "none": None}.get(str(variant.get("weight")))
            class_weight = sqrt_balanced(y) if variant.get("weight") == "sqrt" else weights
            train_keys = [str(r["key"]) for r, _ in seeds]
            dev_keys = [str(r["key"]) for r in items]
            parts = ["xl-lexeme", "deberta"] if encoder in ("concat", "ensemble") else [encoder]

            c = float(str(variant.get("C", 1.0)))
            power = float(str(variant.get("decades", 0)))
            decades = [int(r["decade"]) for r, _ in seeds]  # type: ignore[call-overload]

            groups = [parts[:1], parts[1:]] if encoder == "ensemble" else [parts]
            fitting = {"groups": groups, "class_weight": class_weight, "c": c, "power": power}
            proba, classes = fit(train_keys, y, decades, dev_keys, "dev", **fitting)
            if variant.get("calibrate"):
                rng = np.random.default_rng(2026)
                held = rng.random(len(y)) < 0.1
                kept = [i for i in range(len(y)) if not held[i]]
                out = [i for i in range(len(y)) if held[i]]
                held_proba, held_classes = fit(
                    [train_keys[i] for i in kept],
                    [y[i] for i in kept],
                    [decades[i] for i in kept],
                    [train_keys[i] for i in out],
                    "main",
                    **fitting,
                )
                position = {str(k): n for n, k in enumerate(held_classes)}
                usable = [n for n, i in enumerate(out) if y[i] in position]
                temperature = fit_temperature(
                    held_proba[usable], np.array([position[y[out[n]]] for n in usable])
                )
                temperatures[word] = round(temperature, 3)
                proba = tempered(proba, temperature)
            predicted = [str(classes[int(p.argmax())]) for p in proba]
        if variant.get("cue_first", True):
            predicted = cue_first([usage(r) for r in items], cues, predicted)
        if word == "prompt":
            for decade in (2000, 2020):
                chosen_labels = [
                    label
                    for r, label in zip(items, predicted, strict=True)
                    if r["set"] == "A" and known_decades().get(str(r["id"])) == decade
                ]
                share = chosen_labels.count("language_model_input") / max(1, len(chosen_labels))
                lm_shares[f"prompt A {decade}s"] = round(share, 2)
        for r, label, p in zip(items, predicted, proba, strict=True):
            dataset, uid = str(r["set"]), str(r["id"])
            if dataset == "T":
                labels_t[uid] = label
                probabilities[uid] = {str(c): float(v) for c, v in zip(classes, p, strict=True)}
                continue
            truth = references[dataset].get(uid)
            if truth is None:
                continue
            results[dataset]["n"] += 1
            results[dataset]["right"] += label == truth
            column = {str(k): n for n, k in enumerate(classes)}
            if truth in column:
                losses[dataset].append(-float(np.log(max(float(p[column[truth]]), 1e-12))))
            results[f"{dataset}:{word}"]["n"] += 1
            results[f"{dataset}:{word}"]["right"] += label == truth
    judged = pilot.judgements(False)
    same, overlap, judgement = [], [], []
    for row in (r for r in pilot.key_rows() if not r["repeat_of"]):
        j = judged[row["item"]]
        if j <= 0:
            continue
        a, b = (
            pilot.usage_id(
                row["word"], Usage(row[f"text_{s}"], int(row[f"start_{s}"]), int(row[f"end_{s}"]))
            )
            for s in "ab"
        )
        if a not in labels_t or b not in labels_t:
            continue
        same.append(int(labels_t[a] == labels_t[b]))
        overlap.append(
            sum(probabilities[a][k] * probabilities[b].get(k, 0.0) for k in probabilities[a])
        )
        judgement.append(j)
    summary: dict[str, object] = {
        name: round(c["right"] / c["n"], 3) for name, c in results.items() if ":" not in name
    }
    summary["log loss"] = {name: round(float(np.mean(v)), 3) for name, v in losses.items()}
    summary["language model share (Sonnet 0.03 and 0.91)"] = lm_shares
    if temperatures:
        summary["temperatures"] = temperatures
    summary["T same sense"] = round(float(spearmanr(judgement, same).statistic), 3)
    summary["T overlap"] = round(float(spearmanr(judgement, overlap).statistic), 3)
    summary["by word"] = {
        name: f"{c['right']}/{c['n']}" for name, c in sorted(results.items()) if ":" in name
    }
    return summary


if __name__ == "__main__":
    spec = json.loads(sys.argv[1])
    path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_LIST
    print(json.dumps({"variant": spec, "list": str(path), **run(spec, path)}, indent=1))
