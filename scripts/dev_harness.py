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
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
import pilot_labellers as pilot  # type: ignore[import-not-found]  # noqa: E402

from interpretant.drift.classifier import (  # noqa: E402
    cue_first,
    cue_sense,
    normalised,
    out_of_scope,
    sqrt_balanced,
)
from interpretant.drift.pairs import Usage  # noqa: E402

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
            x = vectors(encoder, "main", [str(r["key"]) for r, _ in seeds])
            y = [s for _, s in seeds]
            weights = {"balanced": "balanced", "none": None}.get(str(variant.get("weight")))
            class_weight = sqrt_balanced(y) if variant.get("weight") == "sqrt" else weights
            model = LogisticRegression(
                C=float(str(variant.get("C", 1.0))), class_weight=class_weight, max_iter=2000
            )
            model.fit(normalised(x), y)
            dx = vectors(encoder, "dev", [str(r["key"]) for r in items])
            proba = model.predict_proba(normalised(dx))
            classes = model.classes_
            predicted = [str(classes[int(p.argmax())]) for p in proba]
        if variant.get("cue_first", True):
            predicted = cue_first([usage(r) for r in items], cues, predicted)
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
