"""Rung 2 on this corpus: round T (T1-T3), the cue-seeded classifier against the author's pair
judgements of check 2b (docs/predictions_labels.md).

Committed before it was run. Steps:

    uv run python scripts/score_labels_corpus.py collect   # whole-corpus sample of the words
    uv run python scripts/score_labels_corpus.py encode    # XL-LEXEME vectors, cached
    uv run python scripts/score_labels_corpus.py classify  # check 2b's usages, by the classifier
    uv run --extra labels python scripts/score_labels_corpus.py sonnet
    uv run --extra labels python scripts/score_labels_corpus.py jev
    uv run python scripts/score_labels_corpus.py score

The lists are read from data/sense_lists/corpus.toml, the frozen copy of the approved draft.
Abstract text and vectors stay under data/labels/corpus/raw/, which is not committed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
import tomllib
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import pilot_labellers as pilot  # noqa: E402

from interpretant.drift.classifier import (  # noqa: E402
    boilerplate,
    cue_sense,
    example_usage,
    normalised,
    tidy_usage,
)
from interpretant.drift.labels import (  # noqa: E402
    BatchLabeller,
    Item,
    JevLabeller,
    load_sense_lists,
)
from interpretant.drift.pairs import Usage, XLLexeme  # noqa: E402
from interpretant.drift.senses import Sampling, cased_abstracts, sample_occurrences  # noqa: E402
from interpretant.predictions import report  # noqa: E402

GRID = Path("data/raw/openalex")
LISTS = Path("data/sense_lists/corpus.toml")
OUT = Path("data/labels/corpus")
RAW = OUT / "raw"
USAGES = RAW / "usages.jsonl"
VECTORS = RAW / "vectors.npz"
WORDS = ("attention", "prompt", "platform", "network", "component", "having")
REFERENCE = {"xl-lexeme": (0.456, 0.494), "second judge": (0.616, 0.657)}
SONNET = BatchLabeller(model="claude-sonnet-5-5", effort="low")
JEV = JevLabeller(model="jev-1.14.0")


def key_of(text: str, start: int, end: int) -> str:
    return hashlib.sha256(f"{start}:{end}:{text}".encode()).hexdigest()[:20]


def collect() -> None:
    """Every sampled usage of the words: every field and decade, at most 150 abstracts per cell."""
    RAW.mkdir(parents=True, exist_ok=True)
    fields = sorted(p.name for p in GRID.iterdir() if p.is_dir())
    with USAGES.open("w", encoding="utf-8") as out:
        for field in fields:
            for decade in range(1950, 2030, 10):
                try:
                    cell = {(field, decade): cased_abstracts(GRID, field, decade)}
                except FileNotFoundError:
                    continue
                for word in WORDS:
                    sampling = Sampling(cap=150, plural=True, quality=True)
                    for o in sample_occurrences(cell, word, seed=1, sampling=sampling):
                        row = {
                            "word": word,
                            "field": field,
                            "decade": decade,
                            "text": o.context,
                            "start": o.start,
                            "end": o.end,
                        }
                        out.write(json.dumps(row) + "\n")
            print(field, "done", flush=True)


def corpus_rows() -> list[dict[str, object]]:
    """The sample's usages, tidied (tidy_usage), without boilerplate sentences (found in five or
    more usages of the same word)."""
    rows = []
    for line in USAGES.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        usage = tidy_usage(Usage(row["text"], row["start"], row["end"]))
        rows.append({**row, "text": usage.text, "start": usage.start, "end": usage.end})
    repeated = {
        (word, text)
        for word in WORDS
        for text in boilerplate([str(r["text"]) for r in rows if r["word"] == word])
    }
    return [r for r in rows if (r["word"], r["text"]) not in repeated]


def gold_usages() -> dict[str, tuple[str, Usage]]:
    """Check 2b's 186 usages, by the pilot's identifiers, their text tidied."""
    return {uid: (w, tidy_usage(u)) for uid, (w, u) in pilot.usages().items()}


def encode() -> None:
    """XL-LEXEME vectors of every training candidate, every example and check 2b's usages."""
    lists = load_sense_lists(LISTS)
    cache: dict[str, np.ndarray] = {}
    if VECTORS.exists():
        stored = np.load(VECTORS)
        cache = {str(k): v for k, v in zip(stored["keys"], stored["vectors"], strict=True)}
    cues = cue_lists()
    pending: dict[str, Usage] = {}
    for row in corpus_rows():
        usage = Usage(str(row["text"]), int(row["start"]), int(row["end"]))  # type: ignore[call-overload]
        if cue_sense(usage, cues[str(row["word"])]) is not None:
            pending.setdefault(key_of(usage.text, usage.start, usage.end), usage)
    for senses in lists.values():
        for sense in senses.senses:
            example = example_usage(senses.word, sense.example)
            if example is not None:
                pending.setdefault(key_of(example.text, example.start, example.end), example)
    for _, usage in gold_usages().values():
        pending.setdefault(key_of(usage.text, usage.start, usage.end), usage)
    todo = {k: u for k, u in pending.items() if k not in cache}
    if todo:
        cache.update(zip(todo, XLLexeme().encode(list(todo.values())), strict=True))
        np.savez(VECTORS, keys=np.array(list(cache)), vectors=np.array(list(cache.values())))
    print(f"{len(todo)} new vectors, {len(cache)} cached")


def cue_lists() -> dict[str, dict[str, list[str]]]:
    table = tomllib.loads(LISTS.read_text(encoding="utf-8"))
    return {w: {s["key"]: s["cues"] for s in body["senses"]} for w, body in table.items()}


def classify() -> None:
    """Train each word's classifier without check 2b's usages; label and score those usages."""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    lists = load_sense_lists(LISTS)
    cues = cue_lists()
    stored = np.load(VECTORS)
    cache = {str(k): v for k, v in zip(stored["keys"], stored["vectors"], strict=True)}
    gold = gold_usages()
    gold_keys = {key_of(u.text, u.start, u.end) for _, u in gold.values()}
    gold_texts = {u.text for _, u in gold.values()}
    labels: dict[str, str] = {}
    probabilities: dict[str, dict[str, float]] = {}
    seeds_used: dict[str, int] = {}
    for word in WORDS:
        x, y = [], []
        seen: set[str] = set()
        for row in corpus_rows():
            if row["word"] != word or row["text"] in gold_texts or row["text"] in seen:
                continue
            seen.add(str(row["text"]))
            usage = Usage(str(row["text"]), int(row["start"]), int(row["end"]))  # type: ignore[call-overload]
            sense = cue_sense(usage, cues[word])
            key = key_of(usage.text, usage.start, usage.end)
            if sense is not None and key not in gold_keys:
                x.append(cache[key])
                y.append(sense)
        seeds_used[word] = len(y)
        for sense in lists[word].senses:
            example = example_usage(lists[word].word, sense.example)
            if example is not None:
                x.append(cache[key_of(example.text, example.start, example.end)])
                y.append(sense.key)
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
        model.fit(normalised(np.array(x)), y)
        targets = [(uid, u) for uid, (w, u) in gold.items() if w == word]
        vectors = normalised(np.array([cache[key_of(u.text, u.start, u.end)] for _, u in targets]))
        for (uid, _), row_p in zip(targets, model.predict_proba(vectors), strict=True):
            probabilities[uid] = {
                str(c): float(p) for c, p in zip(model.classes_, row_p, strict=True)
            }
            labels[uid] = str(model.classes_[int(np.argmax(row_p))])
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "round_t_classifier_labels.tsv").open("w", encoding="utf-8", newline="") as table:
        writer = csv.writer(table, delimiter="\t")
        writer.writerow(["usage", "label", "probabilities"])
        for uid in sorted(labels):
            writer.writerow([uid, labels[uid], json.dumps(probabilities[uid])])
    print(f"{len(labels)} usages labelled; training sentences per word: {seeds_used}")


def language_model(name: str) -> None:
    lists = load_sense_lists(LISTS)
    work = [(lists[w], Item(uid, u)) for uid, (w, u) in sorted(gold_usages().items())]
    labeller = {"sonnet": SONNET, "jev": JEV}[name]
    labels = labeller.label(work, RAW / f"round_t_{name}")  # type: ignore[attr-defined]
    with (OUT / f"round_t_{name}_labels.tsv").open("w", encoding="utf-8", newline="") as table:
        writer = csv.writer(table, delimiter="\t")
        writer.writerow(["usage", "label"])
        writer.writerows(sorted(labels.items()))
    print(f"{name}: {len(labels)} labels")


def read(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8") as table:
        return {row["usage"]: row for row in csv.DictReader(table, delimiter="\t")}


def score() -> None:
    lists = load_sense_lists(LISTS)
    classifier = read(OUT / "round_t_classifier_labels.tsv")
    others = {
        n: read(OUT / f"round_t_{n}_labels.tsv")
        for n in ("sonnet", "jev")
        if (OUT / f"round_t_{n}_labels.tsv").exists()
    }
    rows = [r for r in pilot.key_rows() if not r["repeat_of"]]
    results: dict[str, dict[bool, float]] = {}
    for corrected in (False, True):
        judged = pilot.judgements(corrected)
        pairs = [r for r in rows if judged[r["item"]] > 0]
        ids = [
            tuple(
                pilot.usage_id(
                    r["word"], Usage(r[f"text_{s}"], int(r[f"start_{s}"]), int(r[f"end_{s}"]))
                )
                for s in ("a", "b")
            )
            for r in pairs
        ]
        values = [judged[r["item"]] for r in pairs]

        def same(table: dict[str, dict[str, str]], word: str, a: str, b: str) -> float:
            listed = {s.key for s in lists[word].senses}
            return float(table[a]["label"] == table[b]["label"] and table[a]["label"] in listed)

        def overlap(a: str, b: str) -> float:
            pa = json.loads(classifier[a]["probabilities"])
            pb = json.loads(classifier[b]["probabilities"])
            return sum(pa[s] * pb.get(s, 0.0) for s in pa)

        measured = {
            "classifier, same sense": [
                same(classifier, r["word"], a, b) for r, (a, b) in zip(pairs, ids, strict=True)
            ],
            "classifier, probability overlap": [overlap(a, b) for a, b in ids],
        }
        for name, table in others.items():
            measured[f"{name}, same sense"] = [
                same(table, r["word"], a, b) for r, (a, b) in zip(pairs, ids, strict=True)
            ]
        for name, scores in measured.items():
            results.setdefault(name, {})[corrected] = float(spearmanr(scores, values).statistic)
        label = "with the correction pass" if corrected else "original judgements"
        print(f"{label} ({len(pairs)} pairs):")
        for name in measured:
            print(f"  {name}: Spearman {results[name][corrected]:.3f}")
    report(
        "T1 labels keep the encoder's agreement with the author",
        results["classifier, probability overlap"][False] >= REFERENCE["xl-lexeme"][0],
        f"probability overlap {results['classifier, probability overlap'][False]:.3f} "
        f"(needed {REFERENCE['xl-lexeme'][0]}, XL-LEXEME's raw similarity; second judge "
        f"{REFERENCE['second judge'][0]})",
    )
    if "sonnet, same sense" in results:
        bar = results["sonnet, same sense"][False] - 0.10
        report(
            "T2 the classifier reads the corpus about as well as a language model labeller",
            results["classifier, same sense"][False] >= bar,
            f"classifier {results['classifier, same sense'][False]:.3f}, Sonnet "
            f"{results['sonnet, same sense'][False]:.3f} (needed {bar:.3f})",
        )
    for name, table in others.items():
        shared = sorted(classifier)
        agreement = statistics.mean(classifier[k]["label"] == table[k]["label"] for k in shared)
        report(
            f"T3 agreement with {name}",
            "descriptive",
            f"same sense on {agreement:.1%} of {len(shared)}",
        )


if __name__ == "__main__":
    steps = {"collect": collect, "encode": encode, "classify": classify, "score": score}
    if sys.argv[1] in ("sonnet", "jev"):
        language_model(sys.argv[1])
    else:
        steps[sys.argv[1]]()
