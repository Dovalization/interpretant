"""Stages 2 and 3 for the review page: cue matches across the corpus, and two checks with
XL-LEXEME that point at doubtful training sentences.

- Leave-one-cue-out: for each cue word with at least 3 matches, a classifier trained without that
  cue's sentences labels them; a cue whose sentences mostly land elsewhere is suspect.
- Out-of-fold doubt: every matched sentence is predicted by a classifier trained on the other four
  fifths; those placed in another sense, or with a low probability for their own, are listed first.

Usage text is tidied and boilerplate sentences are dropped first, as in rounds T and A. Vectors
are cached by usage, so a rerun after a cue change encodes only newly caught sentences.

    uv run python scripts/review/build.py USAGES.jsonl VECTORS.npz PREVIEWS.json [LISTS.toml]
"""

import hashlib
import json
import random
import sys
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from interpretant.drift.classifier import (
    boilerplate,
    cue_sense,
    example_usage,
    normalised,
    tidy_usage,
)
from interpretant.drift.labels import shown
from interpretant.drift.pairs import Usage, XLLexeme

LABEL = {
    "ai": "AI",
    "hci": "HCI",
    "compsci": "computer science",
    "cogpsych": "cognitive psychology",
    "histphilsci": "history and philosophy of science",
}
usages_path, vectors_path, out_path = (Path(a) for a in sys.argv[1:4])
draft = tomllib.loads(
    Path(sys.argv[4] if len(sys.argv) > 4 else "data/sense_lists/corpus_draft.toml").read_text()
)
rows = []
for line in usages_path.read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    u = tidy_usage(Usage(r["text"], r["start"], r["end"]))
    rows.append({**r, "text": u.text, "start": u.start, "end": u.end})
_repeated = {
    (w, x)
    for w in {r["word"] for r in rows}
    for x in boilerplate([r["text"] for r in rows if r["word"] == w])
}
rows = [r for r in rows if (r["word"], r["text"]) not in _repeated]
print(f"{len(_repeated)} boilerplate sentences dropped", flush=True)


def key_of(text: str, start: int, end: int) -> str:
    return hashlib.sha256(f"{start}:{end}:{text}".encode()).hexdigest()[:20]


# Stage 2: cue matches
seeds: dict[str, dict[str, list[dict]]] = {}
for word, body in draft.items():
    cues = {s["key"]: s["cues"] for s in body["senses"]}
    by_sense: dict[str, list[dict]] = defaultdict(list)
    seen_texts: set[str] = set()
    for row in rows:
        if row["word"] != word or row["text"] in seen_texts:
            continue
        seen_texts.add(row["text"])
        usage = Usage(row["text"], row["start"], row["end"])
        sense = cue_sense(usage, cues)
        if sense:
            matched = [c for c in cues[sense] if cue_sense(usage, {sense: [c]}) == sense]
            by_sense[sense].append(
                {**row, "cues": matched, "key": key_of(row["text"], row["start"], row["end"])}
            )
    seeds[word] = by_sense

# Encode every seed and example once (cached across runs)
cache: dict[str, np.ndarray] = {}
if vectors_path.exists():
    stored = np.load(vectors_path)
    cache = {str(k): v for k, v in zip(stored["keys"], stored["vectors"], strict=True)}
pending: dict[str, Usage] = {}
for word, body in draft.items():
    for s in body["senses"]:
        ex = example_usage(body["word"], s["example"])
        if ex is not None:
            k = key_of(ex.text, ex.start, ex.end)
            if k not in cache:
                pending[k] = ex
        for seed in seeds[word][s["key"]]:
            if seed["key"] not in cache:
                pending[seed["key"]] = Usage(seed["text"], seed["start"], seed["end"])
if pending:
    vectors = XLLexeme().encode(list(pending.values()))
    cache.update(zip(pending, vectors, strict=True))
    np.savez(vectors_path, keys=np.array(list(cache)), vectors=np.array(list(cache.values())))
print(f"{len(pending)} new vectors, {len(cache)} cached", flush=True)


def fit(x: np.ndarray, y: list[str]) -> LogisticRegression | None:
    if len(set(y)) < 2:
        return None
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    model.fit(normalised(x), y)
    return model


out = {}
for word, body in draft.items():
    senses = [s["key"] for s in body["senses"]]
    examples = []
    for s in body["senses"]:
        ex = example_usage(body["word"], s["example"])
        if ex is not None:
            examples.append((key_of(ex.text, ex.start, ex.end), s["key"]))
    all_seeds = [(seed, sense) for sense in senses for seed in seeds[word][sense]]

    # Out-of-fold doubt
    rng = random.Random(f"2026:{word}")
    order = list(range(len(all_seeds)))
    rng.shuffle(order)
    fold_of = {index: position % 5 for position, index in enumerate(order)}
    doubt: dict[str, tuple[str, float]] = {}
    for fold in range(5):
        train = [i for i in range(len(all_seeds)) if fold_of[i] != fold]
        test = [i for i in range(len(all_seeds)) if fold_of[i] == fold]
        if not test:
            continue
        x = np.array(
            [cache[all_seeds[i][0]["key"]] for i in train] + [cache[k] for k, _ in examples]
        )
        y = [all_seeds[i][1] for i in train] + [s for _, s in examples]
        model = fit(x, y)
        if model is None:
            continue
        probabilities = model.predict_proba(
            normalised(np.array([cache[all_seeds[i][0]["key"]] for i in test]))
        )
        classes = list(model.classes_)
        for i, row in zip(test, probabilities, strict=True):
            own = all_seeds[i][1]
            p_own = float(row[classes.index(own)]) if own in classes else 0.0
            doubt[all_seeds[i][0]["key"]] = (classes[int(np.argmax(row))], p_own)

    # Leave-one-cue-out
    cue_scores: dict[str, dict[str, list]] = defaultdict(dict)
    for sense in senses:
        for cue in sorted({c for seed in seeds[word][sense] for c in seed["cues"]}):
            held = [seed for seed in seeds[word][sense] if cue in seed["cues"]]
            if len(held) < 3:
                continue
            held_keys = {seed["key"] for seed in held}
            train = [(seed, s) for seed, s in all_seeds if seed["key"] not in held_keys]
            x = np.array(
                [cache[seed["key"]] for seed, _ in train] + [cache[k] for k, _ in examples]
            )
            y = [s for _, s in train] + [s for _, s in examples]
            model = fit(x, y)
            if model is None:
                continue
            predicted = model.predict(normalised(np.array([cache[seed["key"]] for seed in held])))
            elsewhere = Counter(str(p) for p in predicted if p != sense)
            cue_scores[sense][cue] = [
                len(held),
                float(np.mean(predicted == sense)),
                elsewhere.most_common(1)[0][0] if elsewhere else "",
            ]

    entries = []
    for s in body["senses"]:
        caught = seeds[word][s["key"]]
        counts = Counter(seed["field"] for seed in caught)
        per_field: dict[str, list[dict]] = defaultdict(list)
        for seed in caught:
            per_field[seed["field"]].append(seed)
        pick_rng = random.Random(f"2026:{word}:{s['key']}")
        for items in per_field.values():
            pick_rng.shuffle(items)
        fields_order = sorted(per_field, key=lambda f: -len(per_field[f]))
        picked: list[dict] = []
        while len(picked) < 40 and any(per_field[f] for f in fields_order):
            for f in fields_order:
                if per_field[f] and len(picked) < 40:
                    picked.append(per_field[f].pop())

        def view(seed: dict, doubt: dict = doubt) -> dict:
            usage = Usage(seed["text"], seed["start"], seed["end"])
            score = doubt.get(seed["key"])
            return {
                "text": shown(usage, limit=80),
                "source": f"{LABEL.get(seed['field'], seed['field'])} · {seed['decade']}s",
                "cues": seed["cues"],
                "p_own": round(score[1], 2) if score else None,
                "predicted": score[0] if score else None,
            }

        flagged = [
            seed
            for seed in caught
            if seed["key"] in doubt
            and (doubt[seed["key"]][0] != s["key"] or doubt[seed["key"]][1] < 0.5)
        ]
        doubtful = sorted(
            (
                seed
                for seed in caught
                if seed["key"] in doubt
                and (doubt[seed["key"]][0] != s["key"] or doubt[seed["key"]][1] < 0.5)
            ),
            key=lambda seed: doubt[seed["key"]][1],
        )[:10]
        entries.append(
            {
                **s,
                "seeds": len(caught),
                "by_field": [[LABEL.get(f, f), n] for f, n in counts.most_common()],
                "cue_scores": cue_scores.get(s["key"], {}),
                "doubtful": [view(seed) for seed in doubtful],
                "n_doubtful": len(flagged),
                "doubtful_to": dict(
                    Counter(
                        doubt[seed["key"]][0]
                        for seed in flagged
                        if doubt[seed["key"]][0] != s["key"]
                    )
                ),
                "confidence": round(
                    float(
                        np.mean([doubt[seed["key"]][1] for seed in caught if seed["key"] in doubt])
                    ),
                    2,
                )
                if any(seed["key"] in doubt for seed in caught)
                else None,
                "previews": [view(seed) for seed in picked],
            }
        )
    out[word] = {"usages": sum(1 for row in rows if row["word"] == word), "senses": entries}
    print(word, {e["key"]: (e["seeds"], len(e["doubtful"])) for e in entries}, flush=True)
out_path.write_text(json.dumps(out, indent=1))
