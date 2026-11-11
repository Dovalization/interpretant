"""Rung 2, second attempt: round U (U1-U4), the revised classifier against Sonnet, with the
author judging their disagreements blind (docs/predictions_labels.md).

Committed before it was run. Steps:

    uv run python scripts/score_labels_head_to_head.py draw      # 240 test usages
    uv run python scripts/score_labels_head_to_head.py encode    # XL-LEXEME vectors, cached
    uv run python scripts/score_labels_head_to_head.py classify  # the revised classifier
    uv run --extra labels python scripts/score_labels_head_to_head.py sonnet
    uv run python scripts/score_labels_head_to_head.py page      # the blind judging page
    uv run python scripts/score_labels_head_to_head.py score     # after the author has judged

The lists are data/sense_lists/corpus_v2.toml. Abstract text, vectors, the page and its key stay
under data/labels/round_u/raw/, which is not committed until the author has judged.
The author's choices are exported from the page to data/labels/round_u/judgements.tsv.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import random
import sys
import tomllib
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pilot_labellers as pilot  # noqa: E402

from interpretant.drift.classifier import (  # noqa: E402
    boilerplate,
    cue_first,
    cue_sense,
    example_usage,
    normalised,
    out_of_scope,
    sqrt_balanced,
    tidy_usage,
)
from interpretant.drift.labels import BatchLabeller, Item, load_sense_lists  # noqa: E402
from interpretant.drift.pairs import Usage, XLLexeme  # noqa: E402
from interpretant.drift.senses import Sampling, cased_abstracts, sample_occurrences  # noqa: E402
from interpretant.predictions import report  # noqa: E402

GRID = Path("data/raw/openalex")
LISTS = Path("data/sense_lists/corpus_v2.toml")
SAMPLE = Path("data/labels/corpus/raw/usages.jsonl")
T_VECTORS = Path("data/labels/corpus/raw/vectors.npz")
OUT = Path("data/labels/round_u")
RAW = OUT / "raw"
EXTRA = RAW / "prompt_uncapped.jsonl"
TEST = RAW / "test.jsonl"
VECTORS = RAW / "vectors.npz"
DEV = OUT / "dev_labels.tsv"
WORDS = ("attention", "prompt", "platform", "network", "component", "having")
PER_WORD = 40
SONNET = BatchLabeller(model="claude-sonnet-5-5", effort="low")
NONE_FITS = "None of the listed meanings fits."


def key_of(text: str, start: int, end: int) -> str:
    return hashlib.sha256(f"{start}:{end}:{text}".encode()).hexdigest()[:20]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:20]


@cache
def rows() -> list[dict[str, object]]:
    """Round T's whole-corpus sample plus prompt uncapped in its two capped cells, tidied, without
    boilerplate or out-of-scope usages."""
    if not EXTRA.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        seen = {
            json.loads(line)["text"]
            for line in SAMPLE.read_text(encoding="utf-8").splitlines()
            if json.loads(line)["word"] == "prompt"
        }
        with EXTRA.open("w", encoding="utf-8") as out:
            for field in ("ai", "compsci"):
                cell = {(field, 2020): cased_abstracts(GRID, field, 2020)}
                sampling = Sampling(cap=100_000, plural=True, quality=True)
                for o in sample_occurrences(cell, "prompt", seed=1, sampling=sampling):
                    if o.context not in seen:
                        row = {"word": "prompt", "field": field, "decade": 2020}
                        row |= {"text": o.context, "start": o.start, "end": o.end}
                        out.write(json.dumps(row) + "\n")
    table = tomllib.loads(LISTS.read_text(encoding="utf-8"))
    tidied = []
    for path in (SAMPLE, EXTRA):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            usage = tidy_usage(Usage(row["text"], row["start"], row["end"]))
            tidied.append({**row, "text": usage.text, "start": usage.start, "end": usage.end})
    repeated = {
        (word, text)
        for word in WORDS
        for text in boilerplate([str(r["text"]) for r in tidied if r["word"] == word])
    }
    return [
        r
        for r in tidied
        if (r["word"], r["text"]) not in repeated
        and not out_of_scope(
            as_usage(r),
            table[str(r["word"])].get("out_of_scope", []),
        )
    ]


def excluded_texts() -> set[str]:
    """Hashes of check 2b's 186 texts and the development set's 300."""
    gold = {text_hash(tidy_usage(u).text) for _, u in pilot.usages().values()}
    with DEV.open(encoding="utf-8") as table:
        dev = {row["text_hash"] for row in csv.DictReader(table, delimiter="\t")}
    return gold | dev


def draw() -> None:
    """40 distinct usages per word, uniformly from the sample's distinct sentences."""
    skip = excluded_texts()
    RAW.mkdir(parents=True, exist_ok=True)
    with TEST.open("w", encoding="utf-8") as out:
        for word in WORDS:
            distinct: dict[str, dict[str, object]] = {}
            for row in rows():
                text = str(row["text"])
                if row["word"] == word and text_hash(text) not in skip:
                    distinct.setdefault(text, row)
            ordered = sorted(distinct.values(), key=lambda r: str(r["text"]))
            for n, row in enumerate(random.Random(f"round-u:{word}").sample(ordered, PER_WORD)):
                out.write(json.dumps({"id": f"{word}-{n:02d}", **row}) + "\n")
    print(f"{PER_WORD * len(WORDS)} test usages drawn")


def test_items() -> list[dict[str, object]]:
    return [json.loads(line) for line in TEST.read_text(encoding="utf-8").splitlines()]


def as_usage(row: dict[str, object]) -> Usage:
    return Usage(str(row["text"]), int(row["start"]), int(row["end"]))  # type: ignore[call-overload]


def cue_lists() -> dict[str, dict[str, list[str]]]:
    table = tomllib.loads(LISTS.read_text(encoding="utf-8"))
    return {w: {s["key"]: s["cues"] for s in body["senses"]} for w, body in table.items()}


def training(word: str, cues: dict[str, list[str]]) -> list[tuple[Usage, str]]:
    """Cue-seeded training usages of one word, without the test texts or check 2b's."""
    skip = {str(r["text"]) for r in test_items()}
    skip |= {tidy_usage(u).text for _, u in pilot.usages().values()}
    seeds, seen = [], set()
    for row in rows():
        text = str(row["text"])
        if row["word"] != word or text in skip or text in seen:
            continue
        seen.add(text)
        usage = as_usage(row)
        sense = cue_sense(usage, cues)
        if sense is not None:
            seeds.append((usage, sense))
    return seeds


def encode() -> None:
    """XL-LEXEME vectors of every training usage, every example and the test usages."""
    lists, cues = load_sense_lists(LISTS), cue_lists()
    cache: dict[str, np.ndarray] = {}
    for path in (T_VECTORS, VECTORS):
        if path.exists():
            stored = np.load(path)
            cache |= {str(k): v for k, v in zip(stored["keys"], stored["vectors"], strict=True)}
    pending: dict[str, Usage] = {}
    for word in WORDS:
        usages = [u for u, _ in training(word, cues[word])]
        usages += [as_usage(r) for r in test_items() if r["word"] == word]
        for sense in lists[word].senses:
            example = example_usage(lists[word].word, sense.example)
            if example is not None:
                usages.append(example)
        for usage in usages:
            pending.setdefault(key_of(usage.text, usage.start, usage.end), usage)
    todo = {k: u for k, u in pending.items() if k not in cache}
    if todo:
        cache |= dict(zip(todo, XLLexeme().encode(list(todo.values())), strict=True))
    needed = {k: cache[k] for k in pending}
    np.savez(VECTORS, keys=np.array(list(needed)), vectors=np.array(list(needed.values())))
    print(f"{len(todo)} new vectors, {len(needed)} kept")


def classify() -> None:
    """Square-root balanced logistic regression per word; cue-first labels for the test usages."""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    lists, cues = load_sense_lists(LISTS), cue_lists()
    stored = np.load(VECTORS)
    cache: dict[str, np.ndarray] = {
        str(k): np.asarray(v) for k, v in zip(stored["keys"], stored["vectors"], strict=True)
    }

    def vector(usage: Usage) -> np.ndarray:
        return cache[key_of(usage.text, usage.start, usage.end)]

    labels: dict[str, str] = {}
    seeds_used: dict[str, int] = {}
    for word in WORDS:
        seeds = training(word, cues[word])
        seeds_used[word] = len(seeds)
        x = [vector(u) for u, _ in seeds]
        y = [s for _, s in seeds]
        for sense in lists[word].senses:
            example = example_usage(lists[word].word, sense.example)
            if example is not None:
                x.append(vector(example))
                y.append(sense.key)
        model = LogisticRegression(C=1.0, class_weight=sqrt_balanced(y), max_iter=1000)
        model.fit(normalised(np.array(x)), y)
        items = [r for r in test_items() if r["word"] == word]
        usages = [as_usage(r) for r in items]
        predicted = [
            str(p) for p in model.predict(normalised(np.array([vector(u) for u in usages])))
        ]
        for row, label in zip(items, cue_first(usages, cues[word], predicted), strict=True):
            labels[str(row["id"])] = label
    write_labels("classifier", labels)
    print(f"{len(labels)} usages labelled; training usages per word: {seeds_used}")


def write_labels(name: str, labels: dict[str, str]) -> None:
    with (OUT / f"{name}_labels.tsv").open("w", encoding="utf-8", newline="") as table:
        writer = csv.writer(table, delimiter="\t")
        writer.writerow(["usage", "label"])
        writer.writerows(sorted(labels.items()))


def read_labels(name: str) -> dict[str, str]:
    with (OUT / f"{name}_labels.tsv").open(encoding="utf-8") as table:
        return {row["usage"]: row["label"] for row in csv.DictReader(table, delimiter="\t")}


def sonnet() -> None:
    lists = load_sense_lists(LISTS)
    work = [(lists[str(r["word"])], Item(str(r["id"]), as_usage(r))) for r in test_items()]
    write_labels("sonnet", SONNET.label(work, RAW / "sonnet"))
    print("sonnet: labelled")


def page() -> None:
    """The judging page and its key: each disagreement with the two labels' definitions in a
    seeded random order, with nothing on the page saying which labeller gave which."""
    lists = load_sense_lists(LISTS)
    classifier, sonnet_labels = read_labels("classifier"), read_labels("sonnet")
    rng = random.Random("round-u:order")
    items, key = [], {}
    for row in test_items():
        uid, word = str(row["id"]), str(row["word"])
        pair = {"classifier": classifier[uid], "sonnet": sonnet_labels[uid]}
        if pair["classifier"] == pair["sonnet"]:
            continue
        order = ["classifier", "sonnet"]
        rng.shuffle(order)
        meaning = {s.key: s.definition for s in lists[word].senses}
        usage = as_usage(row)
        items.append(
            {
                "id": uid,
                "word": word,
                "before": usage.text[: usage.start],
                "target": usage.text[usage.start : usage.end],
                "after": usage.text[usage.end :],
                "options": [meaning.get(pair[name], NONE_FITS) for name in order],
            }
        )
        key[uid] = order
    rng.shuffle(items)
    (RAW / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    template = (Path(__file__).parent / "review" / "judge_template.html").read_text(
        encoding="utf-8"
    )
    embedded = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    (RAW / "judging.html").write_text(template.replace("__ITEMS__", embedded), encoding="utf-8")
    print(f"{len(items)} disagreements on the page ({html.escape(str(RAW / 'judging.html'))})")


def score() -> None:
    classifier, sonnet_labels = read_labels("classifier"), read_labels("sonnet")
    key = json.loads((RAW / "key.json").read_text(encoding="utf-8"))
    with (OUT / "judgements.tsv").open(encoding="utf-8") as table:
        choices = {row["usage"]: row["choice"] for row in csv.DictReader(table, delimiter="\t")}
    word_of = {str(r["id"]): str(r["word"]) for r in test_items()}
    agree = [u for u in classifier if classifier[u] == sonnet_labels[u]]
    outcome: dict[str, str] = {}
    for uid, order in key.items():
        choice = choices.get(uid, "")
        if choice in ("first", "second"):
            outcome[uid] = order[0 if choice == "first" else 1]
        elif choice in ("both", "neither"):
            outcome[uid] = choice
    decided = [u for u, o in outcome.items() if o in ("classifier", "sonnet")]
    wins = sum(outcome[u] == "classifier" for u in decided)
    share = wins / len(decided) if decided else float("nan")
    print(f"{len(key)} disagreements; {len(outcome)} judged; {len(decided)} decided for one label")
    report(
        "U1 where they differ, the classifier is about as good as Sonnet",
        None if len(decided) < 20 else share >= 0.40,
        f"classifier chosen in {wins} of {len(decided)} ({share:.1%}; needed 40%); "
        f"both fit {sum(o == 'both' for o in outcome.values())}, "
        f"neither {sum(o == 'neither' for o in outcome.values())}",
    )
    by_word = Counter(word_of[u] for u in agree)
    report(
        "U2 agreement",
        "descriptive",
        f"{len(agree)} of {len(classifier)} ({len(agree) / len(classifier):.1%}); per word "
        + ", ".join(f"{w} {by_word[w]}/{PER_WORD}" for w in WORDS),
    )
    both = sum(o == "both" for o in outcome.values())
    estimate = {
        name: (len(agree) + sum(outcome[u] == name for u in decided) + both) / len(classifier)
        for name in ("classifier", "sonnet")
    }
    report(
        "U3 rough accuracy (agreements counted right; an upper estimate)",
        "descriptive",
        f"classifier {estimate['classifier']:.1%}, Sonnet {estimate['sonnet']:.1%}",
    )
    per_word: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for uid in decided:
        per_word[word_of[uid]][0] += outcome[uid] == "classifier"
        per_word[word_of[uid]][1] += 1
    report(
        "U4 per word (classifier chosen / decided)",
        "descriptive",
        ", ".join(f"{w} {per_word[w][0]}/{per_word[w][1]}" for w in WORDS),
    )


if __name__ == "__main__":
    steps = {
        "draw": draw,
        "encode": encode,
        "classify": classify,
        "sonnet": sonnet,
        "page": page,
        "score": score,
    }
    steps[sys.argv[1]]()
